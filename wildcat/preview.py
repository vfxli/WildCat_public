# -*- coding: utf-8 -*-
"""Asynchronous preview / validation worker.

Training only snapshots the weights (a device-to-device copy of a few MB) and goes on; full-frame
inference, OCIO display transform and PNG encoding run in this worker, preferably on a second GPU
(e.g. 5090 trains, 4090/4060 renders previews).  If the worker is still busy when the next preview
is due, that preview is skipped instead of stalling training.
"""
import os
import queue
import threading

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
        self.th.start()

    # --------------------------------------------------------------- API (training thread)
    def submit(self, step, snapshot, ready_event, do_preview, do_val, crops=None):
        """snapshot: dict name->tensor (copy on training device). Returns False if skipped."""
        if do_val:
            # validation results are logged and shown in the UI: always queue (blocking if needed)
            self.q.put((step, snapshot, ready_event, do_preview, do_val, crops))
            return True
        if self.busy.is_set() or self.q.full():
            return False
        self.q.put((step, snapshot, ready_event, do_preview, do_val, crops))
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

    def _run(self):
        while True:
            job = self.q.get()
            if job is None:
                return
            self.busy.set()
            try:
                with self.t.gpu_lock:
                    self._do(*job)
            except Exception as e:  # never kill training because of a preview
                log("preview worker error: %s" % e)
            finally:
                self.busy.clear()

    def _do(self, step, snapshot, ready_event, do_preview, do_val, crops=None):
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
            for i, (name, a, b) in enumerate(zip(self.t.pair_names, self._frames(), self.t.tgt_full)):
                fr = a if isinstance(a, torch.Tensor) else a
                out = self._infer(fr)
                o = out[0].permute(1, 2, 0)
                if do_val:
                    gt = torch.from_numpy(np.asarray(b, dtype=np.float32)).to(self.device)
                    mses.append(float(torch.mean((o - gt) ** 2).item()))
                if do_preview:
                    rendered.append((name, i, o.cpu().numpy()))
        if do_val:
            log("val(MSE): " + "  ".join("f%d %.6f" % (i + 1, m) for i, m in enumerate(mses)))
        if do_preview:
            self._write_previews(step, rendered)
        if crops is not None:
            with (torch.cuda.stream(self._stream) if self._stream is not None else _Null()):
                self._write_contact_sheet(step, crops)

    def _write_contact_sheet(self, step, crops):
        """CopyCat-style contact sheet of the current training crops: rows = crops, columns =
        input | ground truth | output, plain sRGB encoding, 16-bit PNG."""
        x, y = crops
        x = x.to(self.device).float()
        y = y.to(self.device).float()
        is_cuda = self.device.type == "cuda"
        with torch.no_grad(), torch.autocast("cuda" if is_cuda else "cpu", dtype=self.amp_dtype or torch.float32,
                                             enabled=self.amp_dtype is not None and is_cuda):
            o = self.net(x).float()
        side = int(self.t.crop_size)
        rows = []
        for i in range(x.shape[0]):
            tiles = []
            for t in (x[i], y[i], o[i]):
                t = t[:3] if t.shape[0] >= 3 else t[:1].repeat(3, 1, 1)
                if t.shape[-1] != side:
                    t = F.interpolate(t[None], size=(side, side), mode="nearest")[0]
                tiles.append(t)
            rows.append(torch.cat(tiles, dim=2))
        sheet = torch.cat(rows, dim=1).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        enc = np.where(sheet <= 0.0031308, sheet * 12.92, 1.055 * np.power(sheet, 1 / 2.4) - 0.055)
        path = "%s.%d.png" % (self.t.nuke_prefix, step)
        tmp = path + ".tmp.png"
        cv2.imwrite(tmp, cv2.cvtColor((np.clip(enc, 0, 1) * 65535).astype(np.uint16), cv2.COLOR_RGB2BGR))
        os.replace(tmp, path)

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
            return color.acescg_to_display_srgb(
                arr[..., :3], src_space=t.working_colorspace, view_mode=t.view_transform,
                ocio_config_path=t.ocio_config,
            )
        return np.clip(arr[..., :3], 0.0, 1.0)

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
            cv2.imwrite(tmp, cv2.cvtColor((np.clip(trip, 0, 1) * 65535).astype(np.uint16), cv2.COLOR_RGB2BGR))
            os.replace(tmp, path)


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
