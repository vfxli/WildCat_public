# -*- coding: utf-8 -*-
"""WildCat image sequence loading, Nuke-CopyCat-equivalent sampling and crop batching.

Sampling semantics (CopyCat-compatible):
  * frames are drawn **without replacement**: an endless stream of shuffled epoch permutations
    of the dataset, consumed ``batch_size`` frames per step (a batch may straddle two epochs);
  * crop origins are uniform over a 4-pixel full-resolution grid anchored at the bottom edge
    (1 / 2 / 4 stage pixels at 1/4, 1/2, 1/1); no flips / rotations / colour augmentation;
  * multi-resolution stages downscale whole frames with **bicubic, align_corners=True**.
"""
import glob
import os
import threading
import queue

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from . import color


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def _read(p, input_colorspace="auto", working_colorspace="ACEScg", ocio_config_path="", is_data_pass=False):
    im = color.read_image_float(
        p,
        input_colorspace=input_colorspace,
        working_colorspace=working_colorspace,
        ocio_config_path=ocio_config_path,
        is_data_pass=is_data_pass,
    )
    if im.ndim == 2:
        im = im[..., None].repeat(3, 2)
    elif im.shape[2] == 1:
        im = im.repeat(3, 2)
    return im


def expand_pattern(pattern):
    if os.path.isdir(pattern):
        exr_files = sorted(glob.glob(os.path.join(pattern, "*.exr")) + glob.glob(os.path.join(pattern, "*.EXR")))
        if exr_files:
            return exr_files
        files = []
        for ext in ("*.png", "*.tif", "*.tiff", "*.jpg", "*.jpeg"):
            files.extend(glob.glob(os.path.join(pattern, ext)))
        return sorted(files)
    return sorted(glob.glob(pattern.replace("####", "*")))


def mem_available_bytes():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return None


def _probe_frame_bytes(path, channels):
    try:
        im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if im is not None:
            return int(im.shape[0]) * int(im.shape[1]) * int(channels)
    except Exception:
        pass
    return None


def load_sequence(
    pattern,
    channels,
    input_colorspace="auto",
    working_colorspace="ACEScg",
    ocio_config_path="",
    dtype="float32",
    ram_fraction=0.80,
):
    """Load a frame sequence as a list of HxWxC numpy arrays.

    dtype: 'float32' (default) or 'auto' -> float16 when every value round-trips through half
    exactly (16-bit half EXR, 8-bit PNG/JPG ...), which halves host RAM with zero loss.
    A host-RAM guard refuses to load sequences that would exhaust system memory.
    """
    files = expand_pattern(pattern)
    if not files:
        raise IOError("no frames match: " + pattern)
    px = _probe_frame_bytes(files[0], channels)
    avail = mem_available_bytes()
    if px and avail:
        need = px * len(files) * (2 if dtype == "auto" else 4)
        if need > avail * ram_fraction:
            raise MemoryError(
                "Sequence %s needs ~%.1f GB host RAM but only %.1f GB is available. "
                "Reduce frame count / resolution." % (pattern, need / 1e9, avail / 1e9)
            )
    is_data = int(channels) == 1
    frames = []
    for p in files:
        im = _read(p, input_colorspace, working_colorspace, ocio_config_path, is_data)
        if im.shape[2] < channels:
            if channels == 1:
                im = im[..., :1]
            else:
                raise ValueError("%s has %d channels, need %d" % (p, im.shape[2], channels))
        a = np.ascontiguousarray(im[..., :channels], dtype=np.float32)
        if dtype == "auto":
            h = a.astype(np.float16)
            if np.array_equal(h.astype(np.float32), a, equal_nan=True):
                a = h
        frames.append(a)
    return files, frames


# --------------------------------------------------------------------------------------
# Multi-resolution pyramid (Nuke: bicubic, align_corners=True)
# --------------------------------------------------------------------------------------
PYRAMID_MODE = "bicubic"


def stage_size(h, w, scale_div):
    if scale_div == 1:
        return h, w
    return max(32, int(h / float(scale_div))), max(32, int(w / float(scale_div)))


def downscale_tensor(t, scale_div, mode=PYRAMID_MODE):
    """t: (N,C,H,W) float32 tensor -> downscaled (N,C,h,w)."""
    if scale_div == 1:
        return t
    nh, nw = stage_size(t.shape[-2], t.shape[-1], scale_div)
    return F.interpolate(t, size=(nh, nw), mode=mode, align_corners=True)


def make_pyramid(frames, scale_div, device="cpu", mode=PYRAMID_MODE):
    """Downscale every frame (numpy HxWxC) by integer divisor; returns list of HxWxC float32 numpy."""
    if scale_div == 1:
        return [np.asarray(f, dtype=np.float32) for f in frames]
    out = []
    for f in frames:
        t = torch.from_numpy(np.asarray(f, dtype=np.float32)).permute(2, 0, 1)[None].to(device)
        r = downscale_tensor(t, scale_div, mode)
        out.append(np.ascontiguousarray(r[0].permute(1, 2, 0).float().cpu().numpy()))
    return out


# --------------------------------------------------------------------------------------
# Nuke-equivalent sampler: epoch permutations, uniform crop offsets
# --------------------------------------------------------------------------------------
class EpochSampler:
    """grid: crop-origin spacing in stage pixels. CopyCat places crops on a 4-pixel grid of the
    full-resolution frame measured from the bottom edge (Nuke's y-up origin), i.e. 1 / 2 / 4 stage
    pixels at 1/4, 1/2, 1/1 resolution. grid=1 everywhere = free placement (not CopyCat)."""

    def __init__(self, n_frames, seed=0):
        self.n = int(n_frames)
        self.rng = np.random.default_rng(int(seed))
        self._buf = np.empty(0, dtype=np.int64)
        self.grid = 1

    def frames(self, count):
        while self._buf.size < count:
            self._buf = np.concatenate([self._buf, self.rng.permutation(self.n)])
        out, self._buf = self._buf[:count], self._buf[count:]
        return out

    def plan(self, steps, bs, sizes, crop):
        """Return int64 array (steps, bs, 3) of (frame, y0, x0) for `steps` consecutive steps.
        sizes: list of (h, w) of the current stage's frames; crop: stage crop size."""
        idx = self.frames(steps * bs).reshape(steps, bs)
        hs = np.array([s[0] for s in sizes], dtype=np.int64)[idx]
        ws = np.array([s[1] for s in sizes], dtype=np.int64)[idx]
        cs = np.minimum(np.minimum(hs, ws), crop)
        cs = np.maximum(32, (cs // 16) * 16)
        g = max(1, int(self.grid))
        ky = (self.rng.random((steps, bs)) * (np.maximum(hs - cs, 0) // g + 1)).astype(np.int64)
        kx = (self.rng.random((steps, bs)) * (np.maximum(ws - cs, 0) // g + 1)).astype(np.int64)
        if g == 1:
            y0, x0 = ky, kx
        else:
            y0 = np.maximum(hs - cs, 0) - g * ky  # grid anchored at the bottom edge (Nuke y-up)
            x0 = g * kx
        return np.stack([idx, y0, x0], axis=-1)


def stage_crop_size(crop, scale_div, h, w):
    c = max(32, (int(crop // scale_div) // 16) * 16)
    c = min(c, h, w)
    return max(32, (c // 16) * 16)


# --------------------------------------------------------------------------------------
# Crop sources
# --------------------------------------------------------------------------------------
class GPUCropSource:
    """All stage frames resident in VRAM as one (N,H,W,C) tensor per side (fp16 when lossless).
    A batch is one advanced-indexing gather -> NHWC buffer == channels_last NCHW view."""

    def __init__(self, src_frames, tgt_frames, scale_div, device, store_dtype=torch.float16):
        self.device = device
        self.sizes = []
        self.uniform = len({f.shape for f in src_frames}) == 1 and len({f.shape for f in tgt_frames}) == 1
        self.store_dtype = store_dtype

        def build(frames):
            outs = []
            for f in frames:
                t = torch.from_numpy(np.asarray(f)).to(device, non_blocking=False).float()
                t = t.permute(2, 0, 1)[None]
                t = downscale_tensor(t, scale_div)
                t = t[0].permute(1, 2, 0)
                if store_dtype == torch.float16:
                    h = t.half()
                    if not torch.equal(h.float(), t):
                        raise _NotExact()
                    t = h
                outs.append(t.contiguous())
            return outs

        try:
            s, g = build(src_frames), build(tgt_frames)
        except _NotExact:
            self.store_dtype = torch.float32
            store_dtype = torch.float32
            s, g = build(src_frames), build(tgt_frames)
        self.sizes = [(t.shape[0], t.shape[1]) for t in s]
        if self.uniform:
            self.src = torch.stack(s)
            self.tgt = torch.stack(g)
            del s, g
        else:
            self.src_list, self.tgt_list = s, g

    def gather(self, plan_dev, i, bs, c, out_x, out_y):
        """plan_dev: (steps,bs,3) int64 on device. Writes NCHW(channels_last) crops into out_x/out_y."""
        p = plan_dev[i]
        if self.uniform:
            ar = torch.arange(c, device=self.device)
            rows = (p[:, 1:2] + ar)[:, :, None]
            cols = (p[:, 2:3] + ar)[:, None, :]
            fi = p[:, 0][:, None, None]
            out_x.copy_(self.src[fi, rows, cols].permute(0, 3, 1, 2))
            out_y.copy_(self.tgt[fi, rows, cols].permute(0, 3, 1, 2))
        else:
            pc = p.cpu().tolist()
            for b, (fi, y0, x0) in enumerate(pc):
                out_x[b].copy_(self.src_list[fi][y0:y0 + c, x0:x0 + c].permute(2, 0, 1))
                out_y[b].copy_(self.tgt_list[fi][y0:y0 + c, x0:x0 + c].permute(2, 0, 1))


class _NotExact(Exception):
    pass


class CPUCropSource:
    """Host-RAM fallback when the dataset does not fit in VRAM: crops are cut on the CPU into
    pinned buffers by a prefetch thread and copied asynchronously (H2D overlaps compute)."""

    def __init__(self, src_frames, tgt_frames, scale_div, device, prefetch=4):
        self.device = device
        if scale_div == 1:
            self.src = [np.asarray(f) for f in src_frames]
            self.tgt = [np.asarray(f) for f in tgt_frames]
        else:
            dev = device if torch.cuda.is_available() else "cpu"
            self.src = [a.astype(np.float16) if src_frames[0].dtype == np.float16 else a for a in make_pyramid(src_frames, scale_div, dev)]
            self.tgt = [a.astype(np.float16) if tgt_frames[0].dtype == np.float16 else a for a in make_pyramid(tgt_frames, scale_div, dev)]
        self.sizes = [f.shape[:2] for f in self.src]
        self.prefetch = prefetch
        self._q = None
        self._thread = None
        self._stop = threading.Event()

    def start(self, plan_iter, bs, c):
        """plan_iter yields (bs,3) numpy rows; a producer thread cuts crops into a ring of
        pinned host buffers; each slot is recycled only after its H2D copy has completed."""
        self.stop()
        self._stop = threading.Event()
        n_slots = self.prefetch + 2
        C = self.src[0].shape[2]
        Ct = self.tgt[0].shape[2]
        pin = torch.cuda.is_available()
        self._slots = [
            (torch.empty((bs, c, c, C), dtype=torch.float32, pin_memory=pin),
             torch.empty((bs, c, c, Ct), dtype=torch.float32, pin_memory=pin))
            for _ in range(n_slots)
        ]
        self._events = [None] * n_slots
        self._free = queue.Queue()
        for i in range(n_slots):
            self._free.put(i)
        self._q = queue.Queue(maxsize=self.prefetch)

        def work():
            try:
                for p in plan_iter:
                    k = self._free.get()
                    if self._stop.is_set():
                        return
                    ev = self._events[k]
                    if ev is not None:
                        ev.synchronize()
                    xn, yn = self._slots[k][0].numpy(), self._slots[k][1].numpy()
                    for b, (fi, y0, x0) in enumerate(p):
                        xn[b] = self.src[fi][y0:y0 + c, x0:x0 + c]
                        yn[b] = self.tgt[fi][y0:y0 + c, x0:x0 + c]
                    self._q.put(k)
            finally:
                self._q.put(None)

        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()

    def next_into(self, out_x, out_y):
        k = self._q.get()
        if k is None:
            raise StopIteration
        xb, yb = self._slots[k]
        out_x.copy_(xb.to(self.device, non_blocking=True).permute(0, 3, 1, 2))
        out_y.copy_(yb.to(self.device, non_blocking=True).permute(0, 3, 1, 2))
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            ev = torch.cuda.Event()
            ev.record()
            self._events[k] = ev
        self._free.put(k)

    def stop(self):
        self._stop.set()
        if getattr(self, "_free", None) is not None:
            for _ in range(4):
                self._free.put(0)
        if self._q is not None:
            try:
                while True:
                    self._q.get_nowait()
            except queue.Empty:
                pass
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None


def dataset_vram_bytes(frames_src, frames_tgt, bytes_per_val=2):
    n = 0
    for f in list(frames_src) + list(frames_tgt):
        n += int(np.prod(f.shape)) * bytes_per_val
    return n
