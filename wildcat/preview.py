# -*- coding: utf-8 -*-
"""Asynchronous preview / validation worker.

Training only snapshots the weights (a device-to-device copy of a few MB) and goes on; full-frame
inference, OCIO display transform and PNG encoding run in this worker, preferably on a second GPU
(e.g. 5090 trains, 4090/4060 renders previews).  If the worker is still busy when the next preview
is due, that preview is skipped instead of stalling training.
"""
import gc
import os
import queue
import threading
import traceback

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from . import color, models
from .log import log


def _run_padded(net, x, amp_dtype, channels_last):
    """x: (1,C,h,w) on device -> (1,Cout,h,w); pads to a multiple of 16 with edge replication."""
    h, w = x.shape[-2:]
    ph, pw = (16 - h % 16) % 16, (16 - w % 16) % 16
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
    is_cuda = x.is_cuda
    if channels_last and is_cuda:
        x = x.contiguous(memory_format=torch.channels_last)
    with torch.autocast("cuda" if is_cuda else "cpu", dtype=amp_dtype or torch.float32,
                        enabled=amp_dtype is not None and is_cuda):
        return net(x)[:, :, :h, :w].float()


def infer_image(net, x, amp_dtype=None, channels_last=False, tile=1024, overlap=128):
    """Full-frame inference with automatic fallback to overlapped tiles when VRAM runs out
    (large plates on 16 GB cards). Tiles keep `overlap` px of context on every side."""
    try:
        return _run_padded(net, x, amp_dtype, channels_last)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
    _, _, h, w = x.shape
    out = None
    step = tile
    for y0 in range(0, h, step):
        for x0 in range(0, w, step):
            y1, x1 = min(h, y0 + step), min(w, x0 + step)
            ya, xa = max(0, y0 - overlap), max(0, x0 - overlap)
            yb, xb = min(h, y1 + overlap), min(w, x1 + overlap)
            o = _run_padded(net, x[:, :, ya:yb, xa:xb], amp_dtype, channels_last)
            if out is None:
                out = torch.empty((1, o.shape[1], h, w), device=x.device, dtype=torch.float32)
            out[:, :, y0:y1, x0:x1] = o[:, :, y0 - ya:y0 - ya + (y1 - y0), x0 - xa:x0 - xa + (x1 - x0)]
    return out


@torch.no_grad()
def forward_full(net, frame, device, amp_dtype=None, channels_last=False):
    """Full-frame inference (edge-replicate padding to a multiple of 16, closest to Nuke Inference)."""
    x = torch.from_numpy(np.asarray(frame)).to(device).float().permute(2, 0, 1)[None]
    return infer_image(net, x, amp_dtype, channels_last)


class PreviewWorker:
    def __init__(self, trainer, device, amp_dtype=None):
        self.t = trainer
        self.device = torch.device(device)
        self.amp_dtype = amp_dtype
        net, _ = models.build_generator(trainer.mo, cudnn_pad=True)
        self.net = net.to(self.device).eval()
        self.q = queue.Queue(maxsize=1)
        self.busy = threading.Event()
        self.th = threading.Thread(target=self._run, daemon=True)
        self._stream = torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        self._frames_dev = None
        self._disp_cache = {}
        self._yield_flag = os.path.join(self.t.run_dir, ".yield_preview_gpu")
        self.th.start()

    # --------------------------------------------------------------- API (training thread)
    def submit(self, step, snapshot, ready_event, do_preview, do_val, crops=None, monitor=None, done=None):
        """snapshot: dict name->tensor (copy on training device). Returns False if skipped.
        monitor: the quality monitor when it has booked this step (wildcat/quality.py) - it is run after the job.
        done: a threading.Event that is set when the job (and the monitor's evaluation) has finished."""
        job = (step, snapshot, ready_event, do_preview, do_val, crops, monitor, done)
        if do_val or monitor is not None:
            # validation results are logged and shown in the UI: always queue (blocking if needed)
            self.q.put(job)
            return True
        if self.busy.is_set() or self.q.full():
            if done is not None:
                done.set()
            return False
        self.q.put(job)
        return True

    def drain(self):
        self.q.put(None)
        self.th.join()

    # --------------------------------------------------------------- worker thread
    def _frames(self):
        if self._frames_dev is None:
            frames = []
            budget = None
            if self.device.type == "cuda":
                free, total = torch.cuda.mem_get_info(self.device)
                budget = free * 0.5
            used = 0
            for a in self.t.src_full:
                nbytes = a.size * 4
                if budget is not None and used + nbytes < budget:
                    frames.append(torch.from_numpy(np.asarray(a, dtype=np.float32)).to(self.device))
                    used += nbytes
                else:
                    frames.append(a)
            self._frames_dev = frames
        return self._frames_dev

    def _maybe_yield_preview_gpu(self):
        """Release the secondary preview GPU back to the training GPU when another task needs that card."""
        train_dev = torch.device(self.t.device)
        if self.device == train_dev or not os.path.isfile(self._yield_flag):
            return
        old_dev = self.device
        try:
            if self._stream is not None:
                self._stream.synchronize()
        except Exception:
            pass
        self._frames_dev = None
        self.net = self.net.to(train_dev).eval()
        self.device = train_dev
        self._stream = torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        if old_dev.type == "cuda":
            gc.collect()
            try:
                with torch.cuda.device(old_dev):
                    torch.cuda.empty_cache()
            except Exception:
                pass
        try:
            os.remove(self._yield_flag)
        except OSError:
            pass
        log("secondary GPU needed by another training task; preview/val switched to %s" % self.device)

    def _run(self):
        while True:
            try:
                job = self.q.get(timeout=0.5)
            except queue.Empty:
                if self.device != torch.device(self.t.device) and os.path.isfile(self._yield_flag):
                    try:
                        with self.t.gpu_lock:
                            self._maybe_yield_preview_gpu()
                    except Exception:
                        log("preview worker yield error:\n%s" % traceback.format_exc())
                continue
            if job is None:
                return
            self.busy.set()
            step, snapshot, ready_event, do_preview, do_val, crops, monitor, done = job
            job = None
            try:
                with self.t.gpu_lock:
                    self._maybe_yield_preview_gpu()
                    rendered = self._do_infer(step, snapshot, ready_event, do_preview, do_val)
                snapshot = None
                if monitor is not None:
                    # quality monitor: self.net holds this step's weights.  It runs before the pictures of the job are
                    # written (the training thread may be waiting for it, see Trainer._preview) and outside the lock
                    # above - only each inference takes it (_monitor_infer), never a whole evaluation.
                    booked, monitor = monitor, None
                    booked.run(step, self._monitor_infer, charge_budget=(done is not None))
                if done is not None:
                    done.set()
                with self.t.gpu_lock:
                    self._do_write(step, rendered, do_preview, crops)
            except Exception:  # never kill training because of a preview, but always log full stack
                log("preview worker error:\n%s" % traceback.format_exc())
            finally:
                if monitor is not None:  # the job failed before the evaluation: give the booking back
                    monitor.skipped(step)
                self.busy.clear()
                if done is not None:
                    done.set()

    def _monitor_infer(self, frame):
        """quality monitor: one full frame (HxWx3 float32) through the weights the last job loaded (eval mode, no_grad,
        FP16 on CUDA for fast full-frame sweep, FP32 on CPU for bit-reproducible tests).  Lock held per frame only."""
        mon_amp = torch.float16 if self.device.type == "cuda" else self.amp_dtype
        with self.t.gpu_lock:
            with (torch.cuda.stream(self._stream) if self._stream is not None else _Null()):
                return forward_full(self.net, frame, self.device, mon_amp)[0].permute(1, 2, 0).cpu().numpy()

    def _do_infer(self, step, snapshot, ready_event, do_preview, do_val):
        """load the snapshot into this worker's network, validate / render the keyframes; returns what to write."""
        if ready_event is not None:
            ready_event.synchronize()
        ctx = torch.cuda.stream(self._stream) if self._stream is not None else _Null()
        rendered = []
        mses = []
        with ctx:
            # weights copy + every kernel below run on this worker's stream, in order
            sd = {k: v.to(self.device, non_blocking=False) for k, v in snapshot.items()}
            del snapshot
            self.net.load_state_dict(sd, strict=True)
            if self._stream is not None:
                self._stream.synchronize()
            # (a job of the quality monitor alone - neither validation nor preview - only needs the weights loaded)
            pairs = zip(self.t.pair_names, self._frames(), self.t.tgt_full) if (do_val or do_preview) else ()
            for i, (name, a, b) in enumerate(pairs):
                fr = a if isinstance(a, torch.Tensor) else a
                out = self._infer(fr)
                o = out[0].permute(1, 2, 0)
                if do_val:
                    gt = torch.from_numpy(np.asarray(b, dtype=np.float32)).to(self.device)
                    mses.append(float(torch.mean((o - gt) ** 2).item()))
                if do_preview:
                    rendered.append((name, i, o.cpu().numpy()))
        if do_val:
            log("@step %d val(MSE): %s" % (step, "  ".join("f%d %.6f" % (i + 1, m) for i, m in enumerate(mses))))
        return rendered

    def _do_write(self, step, rendered, do_preview, crops=None):
        """the pictures of a job: full-frame previews and the contact sheet of training crops."""
        if do_preview:
            self._write_previews(step, rendered)
        if crops is not None:
            with (torch.cuda.stream(self._stream) if self._stream is not None else _Null()):
                self._write_contact_sheet(step, crops)

    def _pad_crops_to_three(self, x, y, step, scale):
        """Ensure at least 3 crop pairs for a complete 3x3 Nuke-style Contact Sheet without touching global RNG."""
        if x.shape[0] >= 3 or not self.t.src_full:
            return x[:3], y[:3]
        rng = np.random.RandomState((int(step) * 2654435761 + 17) & 0xFFFFFFFF)
        ch, cw = int(x.shape[-2]), int(x.shape[-1])
        xs, ys = [x], [y]
        need = 3 - int(x.shape[0])
        n_frames = len(self.t.src_full)
        sc = max(1, int(scale))
        for _ in range(need):
            fi = int(rng.randint(0, n_frames))
            sf = torch.from_numpy(np.asarray(self.t.src_full[fi], dtype=np.float32)).to(self.device).permute(2, 0, 1)[None]
            tf = torch.from_numpy(np.asarray(self.t.tgt_full[fi], dtype=np.float32)).to(self.device).permute(2, 0, 1)[None]
            if sc > 1:
                sh = max(1, sf.shape[-2] // sc)
                sw = max(1, sf.shape[-1] // sc)
                sf = F.interpolate(sf, size=(sh, sw), mode="bicubic", align_corners=True)
                tf = F.interpolate(tf, size=(sh, sw), mode="bicubic", align_corners=True)
            pad_h = max(0, ch - sf.shape[-2])
            pad_w = max(0, cw - sf.shape[-1])
            if pad_h or pad_w:
                sf = F.pad(sf, (0, pad_w, 0, pad_h), mode="replicate")
                tf = F.pad(tf, (0, pad_w, 0, pad_h), mode="replicate")
            y0 = int(rng.randint(0, max(1, sf.shape[-2] - ch + 1)))
            x0 = int(rng.randint(0, max(1, sf.shape[-1] - cw + 1)))
            xs.append(sf[:, :, y0 : y0 + ch, x0 : x0 + cw])
            ys.append(tf[:, :, y0 : y0 + ch, x0 : x0 + cw])
        return torch.cat(xs, dim=0)[:3], torch.cat(ys, dim=0)[:3]

    def _write_contact_sheet(self, step, crops):
        """CopyCat-style 3x3 contact sheet of training crops: 3 rows = crops, 3 columns =
        input | ground truth | output. Writes both Nuke's raw Training_*.<step>.png and an
        OCIO display-referred sheet under contact_sup/ for accurate HDR Web UI viewing."""
        if len(crops) == 3:
            x_raw, y_raw, scale = crops
        else:
            x_raw, y_raw, scale = crops[0], crops[1], 1
        x = x_raw.to(self.device).float()
        y = y_raw.to(self.device).float()
        x, y = self._pad_crops_to_three(x, y, step, scale)
        ch, cw = int(x.shape[-2]), int(x.shape[-1])
        ph, pw = (16 - ch % 16) % 16, (16 - cw % 16) % 16
        x_in = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x
        is_cuda = self.device.type == "cuda"
        with torch.no_grad(), torch.autocast("cuda" if is_cuda else "cpu", dtype=self.amp_dtype or torch.float32,
                                             enabled=self.amp_dtype is not None and is_cuda):
            o = self.net(x_in)[:, :, :ch, :cw].float()

        side = int(self.t.crop_size)
        rows_raw = []
        rows_disp = []
        for i in range(x.shape[0]):
            tiles_raw = []
            tiles_disp = []
            for t_chw in (x[i], y[i], o[i]):
                if t_chw.shape[-2] != side or t_chw.shape[-1] != side:
                    t_up = F.interpolate(t_chw[None], size=(side, side), mode="nearest")[0]
                else:
                    t_up = t_chw
                t_rgb = t_up[:3] if t_up.shape[0] >= 3 else t_up[:1].repeat(3, 1, 1)
                tiles_raw.append(t_rgb)
                hwc = t_up.permute(1, 2, 0).cpu().numpy()
                tiles_disp.append(self._to_disp(hwc))
            rows_raw.append(torch.cat(tiles_raw, dim=2))
            rows_disp.append(np.concatenate(tiles_disp, axis=1))

        # 1. Nuke-compatible Training_YYMMDD_HHMMSS.<step>.png in run_dir
        sheet = torch.cat(rows_raw, dim=1).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        enc = np.where(sheet <= 0.0031308, sheet * 12.92, 1.055 * np.power(sheet, 1 / 2.4) - 0.055)
        path = "%s.%d.png" % (self.t.nuke_prefix, step)
        tmp = path + ".tmp.png"
        color.imwrite_unicode(tmp, cv2.cvtColor((np.clip(enc, 0, 1) * 65535).astype(np.uint16), cv2.COLOR_RGB2BGR))
        os.replace(tmp, path)

        # 2. OCIO display-managed 3x3 Contact Sheet for Web UI live preview
        disp_sheet = np.concatenate(rows_disp, axis=0)
        cdir = os.path.join(self.t.run_dir, "contact_sup")
        os.makedirs(cdir, exist_ok=True)
        cpath = os.path.join(cdir, "step_%06d_s%d.png" % (step, max(1, int(scale))))
        ctmp = cpath + ".tmp.png"
        color.imwrite_unicode(ctmp, cv2.cvtColor((np.clip(disp_sheet, 0, 1) * 65535).astype(np.uint16), cv2.COLOR_RGB2BGR))
        os.replace(ctmp, cpath)

    def _infer(self, fr):
        if isinstance(fr, torch.Tensor):
            with torch.no_grad():
                return infer_image(self.net, fr.permute(2, 0, 1)[None].float(), self.amp_dtype)
        return forward_full(self.net, fr, self.device, self.amp_dtype)

    def _to_disp(self, arr):
        t = self.t
        arr = np.asarray(arr, dtype=np.float32)
        if arr.shape[2] == 1:
            return np.clip(np.repeat(arr, 3, axis=2), 0.0, 1.0)
        if t.use_ocio_display_view:
            srgb = color.acescg_to_display_srgb(
                arr[..., :3], src_space=t.working_colorspace, view_mode=t.view_transform,
                ocio_config_path=t.ocio_config,
            )
        else:
            srgb = np.clip(arr[..., :3], 0.0, 1.0)
        if arr.shape[2] >= 4:
            alpha = np.clip(arr[..., 3:4], 0.0, 1.0)
            if float(np.min(alpha)) < 0.995:
                h, w = srgb.shape[:2]
                yy = (np.arange(h, dtype=np.int32) // 16)[:, None]
                xx = (np.arange(w, dtype=np.int32) // 16)[None, :]
                chk = np.where(((yy + xx) & 1) == 0, 0.11, 0.165).astype(np.float32)[..., None]
                bg_mask = alpha[..., 0] < 0.01
                rgb_fg = (srgb * alpha) if (bool(bg_mask.any()) and float(np.mean(srgb[bg_mask])) > 0.02) else srgb
                return np.clip(rgb_fg + (1.0 - alpha) * chk, 0.0, 1.0)
        return srgb

    def _write_previews(self, step, rendered):
        t = self.t
        psize = t.io.get("preview_size", 0)
        for name, i, o in rendered:
            if name not in self._disp_cache:
                self._disp_cache[name] = (self._to_disp(t.src_full[i]), self._to_disp(t.tgt_full[i]))
            a_vis, b_vis = self._disp_cache[name]
            trip = np.concatenate([a_vis, self._to_disp(o), b_vis], 1)
            s = psize / max(trip.shape[:2]) if psize > 0 else 1.0
            if s < 1.0:
                trip = cv2.resize(trip, (int(trip.shape[1] * s), int(trip.shape[0] * s)), interpolation=cv2.INTER_AREA)
            pdir = os.path.join(t.run_dir, "preview_sup", name)
            os.makedirs(pdir, exist_ok=True)
            path = os.path.join(pdir, "step_%06d.png" % step)
            tmp = path + ".tmp.png"
            color.imwrite_unicode(tmp, cv2.cvtColor((np.clip(trip, 0, 1) * 65535).astype(np.uint16), cv2.COLOR_RGB2BGR))
            os.replace(tmp, path)


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
