# -*- coding: utf-8 -*-
"""A shot seen through its keyframes: the plate's motion, and the keyframe tools built on it (private edition only).

A paired dataset usually holds a handful of hand-made keyframes (source = a plate frame, target = the artist's
version of it) plus the whole plate as its evaluation sequence.  This module provides

  * ``Shot``        the plate in temporal order, the keyframes matched onto it, the plate's frame-to-frame optical
                    flow and ``link`` (bring something from one frame onto another, with a forward-backward
                    confidence).  The three tools below and the advanced module "keyframe commonality"
                    (wildcat/kfcommon.py, ``kf_common``) all stand on this.
  * ``consistency`` tool: how well neighbouring keyframes agree with each other once the plate motion is removed
                    (broad look / medium texture / fine detail), with heat maps.
  * ``coverage``    tool: which frames of the shot are too far from every keyframe (the plate changes so much on the
                    way that the keyframes say little about them) and where one more keyframe would help most.
  * ``odd_keyframes`` tool: which keyframe is the odd one, and where - what ``kf_common`` would rewrite, shown
                    without training.
  * ``propagate``   INTERNAL UTILITY AND TEST ANCHOR, not offered to users.  It carries the keyframes' edit (the log
                    ratio target / source) onto every plate frame between the first and the last keyframe and writes
                    source/ + target/ for all those frames.  It is not the remedy for inconsistent keyframes - a
                    compositor can do the same with Nuke's own optical flow, and it fails where the flow fails; the
                    remedy is ``kf_common``, which trains on the artist's keyframes only.  It is kept because its
                    output is compared frame by frame with reference datasets (tools/test_kfprop.py): that comparison
                    is what pins down the flow / carry arithmetic everything above depends on.  The web API offers it
                    to administrators only; there is no entry in the UI.

None of the tools trains anything or changes a dataset.  numpy + OpenCV only (no torch).
The numerical behaviour (DIS flow preset, half-float flow storage, chaining order, blending weights) is the one that
was validated on a real shot; tools/test_kfprop.py checks it against reference datasets.  Do not "tidy" the
arithmetic without re-running that test.

Optical flow of the plate: OpenCV DIS (the default; CPU, nothing to install).  Where the RAFT engine is installed,
``flow="raft"`` uses it (more accurate, above all in fast motion; needs a GPU and a weights file); it runs in a process
of its own so that this module and the web service never import torch, and when it cannot run the flow is computed with
DIS and the result says so (``flow_fallback``).  The public edition has the DIS engine only.
"""
import collections
import gc
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np

from . import color

EPS = 0.02  # stabiliser of the log ratio
SIG = 3.0  # sigma (px) that separates "overall look" from "detail"
CONF_PX = {"nearest": 2.0, "reference": 6.0, "detail": 6.0}  # forward-backward flow error (px) at which a carried edit is trusted ~37%
DETAIL_SIGMA = 10.0  # mode "detail": Gaussian sigma (px); structures finer than this come from the reference keyframe
REF_TOL = 0.03  # automatic reference: keyframes whose mean reach is within this of the best are candidates
SELECT_MAX_PIXELS = 700_000  # the automatic choice is a score, not a picture: larger plates are scored on a reduced picture
SELECT_MAX_KEYS = 32  # ... and on at most this many evenly spaced keyframes
MATCH_TOL = 2e-3  # fast path: same frame number is accepted immediately when no pixel differs by more than this
MATCH_MEAN_TOL = 0.020  # perceptual [0, 1) mean abs diff across the frame (accepts DWAA/PNG/JPG/HDR fp16/denoise)
MATCH_LOCAL_TOL = 0.038  # perceptual [0, 1) max abs diff after 2.5 px Gaussian blur (rejects painted targets & >= 0.5 px motion)
NUM_THREADS = 4  # OpenCV worker threads (the web service shares the machine with training jobs)
RAM_LIMIT_BYTES = 2 * 1024**3  # flow / label buffers beyond this go to a memory-mapped scratch file

# Agreement thresholds of the consistency check: (consistent, mostly consistent); below the second = inconsistent.
# One artist version carried along the flow scores about 0.95-1.0 / 0.9 / 0.75 in the three bands.
BAND_THRESHOLDS = {"broad": (0.9, 0.75), "medium": (0.75, 0.4), "fine": (0.55, 0.2)}
OFFSET_MIN_MEDIUM = 0.4  # below this the texture itself does not match and a position offset cannot be estimated

COVERAGE_THRESHOLD = 0.7  # keyframe coverage: a frame below this is "too far from every keyframe"
ODD_MIN_KEYFRAMES = 3  # "which keyframe is the odd one": two keyframes cannot say which of them is off
ODD_NOTHING = 0.01  # ... mean share of the edited area below which nothing stands out

# Optical-flow engines.  "dis" is the default and behaves exactly as before RAFT existed.
FLOW_ENGINES = ("dis",)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FRAME_NUM_RE = re.compile(r"(\d+)(?=\.[^.]+$)")


class KfError(ValueError):
    """A problem with the dataset that the user can act on.  ``code`` selects the localised message."""

    def __init__(self, code, message, **params):
        super().__init__(message)
        self.code = code
        self.params = params


class KfCancelled(Exception):
    """Raised between frames when the caller's ``cancel()`` returns True."""


def frame_num(path):
    m = _FRAME_NUM_RE.search(os.path.basename(str(path)))
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------- RAFT (optional second flow engine)
class RaftUnavailable(Exception):
    """RAFT cannot run or did not finish: ``code`` = no_weights | no_torch | no_gpu | no_memory | failed (+ params for
    the message)."""

    def __init__(self, code, **params):
        super().__init__("%s %s" % (code, params))
        self.code = code
        self.params = params




def raft_status():
    """(available, code, params): the public edition has no RAFT engine."""
    return False, "no_weights", {"file": "-"}


def _srgb(a):
    """working space (ACEScg) -> display sRGB without tone mapping, 0..1 (for the flow luma and the report picture)."""
    lin = np.clip(np.einsum("dc,hwc->hwd", color._AP1_TO_SRGB_MAT, a[..., :3]), 0, None)
    return np.clip(np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(np.maximum(lin, 1e-8), 1 / 2.4) - 0.055), 0, 1).astype(np.float32)


def _match_tone(a):
    """Perceptual [0, 1) tone curve across linear + HDR values: avoids amplifying half-float ULP noise on bright
    highlights (> 1) while still distinguishing different HDR intensities."""
    u = np.asarray(a[..., :3], dtype=np.float32)
    pos = np.maximum(u, 0.0)
    return (np.sqrt(pos / (1.0 + pos)) + 0.5 * np.clip(u, -0.5, 0.0)).astype(np.float32)


def _match_pair_err(ta, tb):
    """(mean perceptual abs diff, peak local perceptual diff after 2.5 px Gaussian smoothing)."""
    diff = ta - tb
    return float(np.abs(diff).mean()), float(np.abs(cv2.GaussianBlur(diff, (0, 0), 2.5)).max())


def _lp(z):
    return cv2.GaussianBlur(z, (0, 0), SIG)


def _warp(z, mx, my):
    return cv2.remap(z, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


_LUM = np.array([0.2722, 0.6741, 0.0537], np.float32)  # luminance weights of the working space (ACEScg)


def _carry(shot, t, k, e_k, mx, my, limit=True):
    """The edit layer ``e_k`` of frame k brought onto frame t with the sampling maps (mx, my) of ``shot.link(t, k)``.

    The log ratio is the right carrier for paint that absorbs light (blood, dirt, colour): it follows the lighting of
    the plate.  A HIGHLIGHT painted on a dark spot of frame k is a large ratio, though, and becomes a hot spot when the
    same spot is brighter in frame t (real shot, reference frame 31 onto keyframes 11 / 21: luminance 2.2-2.3 on the
    cheek where the artist never exceeds 1.0).  With ``limit`` a brightening is never carried as more than its additive
    amount; darkening and everything else stay a ratio.  Decided per pixel on luminance, so the hue of the edit is kept."""
    g = _warp(e_k, mx, my)
    if not limit:
        return g
    d = _warp((np.maximum(shot.S(k), 0) + EPS) * np.expm1(e_k), mx, my)  # the edit as an additive amount, carried the same way
    st = np.maximum(shot.S(t), 0) + EPS
    mult = st * np.exp(g)
    add = np.maximum(st + d, 0.25 * EPS)
    pick = ((d @ _LUM) > 0) & ((add @ _LUM) < (mult @ _LUM))
    return np.where(pick[..., None], np.log(add / st), g).astype(np.float32)


def _neighbours(t, keys):
    """nearest keyframe below and above t (never t itself)."""
    lo = max([k for k in keys if k < t], default=None)
    hi = min([k for k in keys if k > t], default=None)
    return lo, hi


class _Lru:
    def __init__(self, cap):
        self.cap = cap
        self.d = collections.OrderedDict()

    def get(self, key, make):
        if key in self.d:
            self.d.move_to_end(key)
            return self.d[key]
        val = make(key)
        self.d[key] = val
        while len(self.d) > self.cap:
            self.d.popitem(last=False)
        return val

    def clear(self):
        self.d.clear()


class _Store:
    """Large working buffers: in RAM while the total stays under ``RAM_LIMIT_BYTES``, else memory-mapped scratch files."""

    def __init__(self, scratch_dir=None, ram_limit=RAM_LIMIT_BYTES):
        self.scratch_dir = scratch_dir
        self.ram_limit = ram_limit
        self.used = 0
        self.files = []
        self.mapped = False

    def alloc(self, shape, dtype, tag):
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        if self.scratch_dir and self.used + nbytes > self.ram_limit:
            os.makedirs(self.scratch_dir, exist_ok=True)
            try:
                free = shutil.disk_usage(self.scratch_dir).free
            except OSError:
                free = None
            if free is not None and free < nbytes + 1024**3:
                raise KfError("no_space", "not enough free disk space for the working files (%.1f GB needed)" % (nbytes / 1e9), gb="%.1f" % (nbytes / 1e9))
            path = os.path.join(self.scratch_dir, "kf_%s_%d_%d.npy" % (tag, os.getpid(), len(self.files)))
            self.files.append(path)
            self.mapped = True
            return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=tuple(shape))
        self.used += nbytes
        return np.zeros(shape, dtype)

    def close(self):
        gc.collect()
        for p in self.files:
            try:
                os.remove(p)
            except OSError:
                pass
        self.files = []


class Progress:
    """Maps (stage, done, total) of a fixed stage plan onto one 0..1 fraction and polls the cancel flag."""

    def __init__(self, plan, callback=None, cancel=None):
        self.plan = list(plan)  # [(stage, weight), ...]
        self.callback = callback
        self.cancel = cancel
        tot = float(sum(w for _, w in self.plan)) or 1.0
        self.start = {}
        self.span = {}
        acc = 0.0
        for name, w in self.plan:
            self.start[name] = acc / tot
            self.span[name] = w / tot
            acc += w

    def tick(self, stage, done, total):
        if self.cancel is not None and self.cancel():
            raise KfCancelled()
        if self.callback is not None:
            frac = self.start.get(stage, 0.0) + self.span.get(stage, 0.0) * (float(done) / max(1, int(total)))
            self.callback(min(1.0, max(0.0, frac)), stage, int(done), int(total))


class Shot:
    """A dataset seen as a shot: the plate (evaluation sequence, in temporal order) and the keyframes matched onto it.

    Images are read exactly as the trainer reads them (``color.read_image_float`` with the same colourspace
    arguments), first three channels.  Frames are read on demand; only a few are kept in memory.
    """

    def __init__(self, source_files, target_files, plate_files, input_colorspace="ACEScg", working_colorspace="ACEScg",
                 ocio_config_path="", scratch_dir=None, ram_limit=RAM_LIMIT_BYTES, ignore_alpha=False):
        n_pair = min(len(source_files), len(target_files))
        self.source_files = list(source_files)[:n_pair]
        self.target_files = list(target_files)[:n_pair]
        self.input_colorspace = input_colorspace
        self.working_colorspace = working_colorspace
        self.ocio_config_path = ocio_config_path
        self.ignore_alpha = bool(ignore_alpha)
        if n_pair < 2:
            raise KfError("too_few_keyframes", "at least 2 keyframes are needed (found %d)" % n_pair, count=n_pair, need=2)
        if len(plate_files) < 2:
            raise KfError("no_plate", "the dataset has no evaluation sequence (the plate)")

        nums = [frame_num(p) for p in plate_files]
        self.plate_has_nums = all(n is not None for n in nums) and len(set(nums)) == len(nums)
        if self.plate_has_nums:  # temporal order = frame number
            order = sorted(range(len(plate_files)), key=lambda i: nums[i])
            self.plate_files = [plate_files[i] for i in order]
            self.plate_nums = [nums[i] for i in order]
        else:
            self.plate_files = list(plate_files)
            self.plate_nums = list(range(1, len(plate_files) + 1))
        self.n = len(self.plate_files)
        self.h = self.w = 0
        self.keys = []  # plate indices of the keyframes, ascending
        self.key_source = {}
        self.key_target = {}
        self.key_num = {}
        self.matched_by_number = 0
        self._store = _Store(scratch_dir, ram_limit)
        self._s = _Lru(6)
        self._t = _Lru(3)
        self._e = _Lru(4)
        self._fl = None
        self._f0 = 0
        self._f1 = 0
        self.yy = self.xx = None
        self.flow_engine = "dis"  # the engine the flow was actually computed with
        self.flow_fallback = None  # {"code", "params"} when RAFT was asked for and DIS had to be used instead
        self.flow_info = {}  # RAFT: scale of the working picture, GPU, peak GPU memory, seconds
        self._tmp_dir = None

    # ------------------------------------------------------------------ reading
    def _read_rgb(self, path, check_alpha=False):
        try:
            im = color.read_image_float(path, input_colorspace=self.input_colorspace, working_colorspace=self.working_colorspace,
                                        ocio_config_path=self.ocio_config_path)
        except Exception as exc:
            raise KfError("read_failed", "cannot read %s (%s)" % (os.path.basename(path), exc), name=os.path.basename(path))
        if im.ndim != 3 or im.shape[2] < 3:
            raise KfError("rgb_only", "only RGB to RGB datasets are supported (%s is not an RGB image)" % os.path.basename(path),
                          name=os.path.basename(path))
        if check_alpha and not self.ignore_alpha and im.shape[2] >= 4:
            a = im[..., 3]
            if float(a.max()) - float(a.min()) > 1e-3:
                raise KfError("alpha", "datasets with a varying alpha channel are not supported (%s)" % os.path.basename(path),
                              name=os.path.basename(path))
        return np.ascontiguousarray(im[..., :3], dtype=np.float32)

    def _check_size(self, a, path):
        if a.shape[:2] != (self.h, self.w):
            raise KfError("size", "%s is %dx%d, the keyframes are %dx%d" % (os.path.basename(path), a.shape[1], a.shape[0], self.w, self.h),
                          name=os.path.basename(path), w=int(a.shape[1]), h=int(a.shape[0]), w0=self.w, h0=self.h)
        return a

    def _plate_raw(self, t):
        return self._check_size(self._read_rgb(self.plate_files[t]), self.plate_files[t])

    def S(self, t):
        """plate frame t, rounded to half-float precision: exactly what source/ of the new dataset will hold."""
        return self._s.get(t, lambda i: np.clip(self._plate_raw(i), -65504.0, 65504.0).astype(np.float16).astype(np.float32))

    def T(self, k):
        """the artist's version (target) of keyframe k."""
        return self._t.get(k, lambda i: self._check_size(self._read_rgb(self.key_target[i], check_alpha=True), self.key_target[i]))

    def to_log(self, t, lab):
        return np.log((np.maximum(lab, 0) + EPS) / (np.maximum(self.S(t), 0) + EPS)).astype(np.float32)

    def from_log(self, t, e):
        s = self.S(t)
        return ((np.maximum(s, 0) + EPS) * np.exp(e) - EPS + np.minimum(s, 0)).astype(np.float32)

    def E(self, k):
        """edit layer of keyframe k: log ratio of the artist's version over the plate frame."""
        return self._e.get(k, lambda i: self.to_log(i, self.T(i)))

    # ------------------------------------------------------------------ keyframes -> plate frames
    def match_keyframes(self, tick=None):
        """Find the plate frame of every keyframe: fast path by file number when bit-exact, otherwise global best
        perceptual match across all plate frames (tolerating compression, bit-depth and HDR fp16 rounding while
        rejecting painted targets or neighbouring moving frames).  Raises KfError when a keyframe is not in the plate."""
        tick = tick or (lambda done, total: None)
        thumb = lambda a: cv2.resize(a, (64, 64), interpolation=cv2.INTER_AREA)
        num_to_idx = {n: i for i, n in enumerate(self.plate_nums)} if self.plate_has_nums else {}
        n_key = len(self.source_files)
        total = n_key + self.n
        matched, pending = {}, {}
        key_tones = _Lru(6)
        for i, p in enumerate(self.source_files):
            tick(i, total)
            ks = self._read_rgb(p, check_alpha=True)
            if i == 0:
                self.h, self.w = int(ks.shape[0]), int(ks.shape[1])
            else:
                self._check_size(ks, p)
            num = frame_num(p)
            j = num_to_idx.get(num) if num is not None else None
            if j is not None and float(np.abs(ks - self._plate_raw(j)).max()) < MATCH_TOL:
                matched[i] = j
            else:
                kt = _match_tone(ks)
                key_tones.get(i, lambda _idx, _kt=kt: _kt)
                pending[i] = thumb(kt)
        self.matched_by_number = len(matched)
        if len(set(matched.values())) != len(matched):
            raise KfError("duplicate_keyframe", "two keyframes are the same plate frame")
        if pending:
            used_j = set(matched.values())
            pass_cands = {i: [] for i in pending}  # up to top-2 (score, m_err, l_err, j) passing full-res tolerances
            best_thumb = {}  # i -> (thumb_score, j) across all plate frames, for diagnostic reporting on mismatch
            exact_locked = set()
            locked_frames = set()
            for j in range(self.n):
                if len(exact_locked) == len(pending):
                    break
                tick(n_key + j, total)
                pj = self._plate_raw(j)
                pt = _match_tone(pj)
                tj = thumb(pt)
                for i in sorted(pending):
                    if i in exact_locked:
                        continue
                    td = pending[i] - tj
                    t_mean = float(np.abs(td).mean())
                    t_local = float(np.abs(cv2.GaussianBlur(td, (0, 0), 1.0)).max())
                    if i not in best_thumb or t_mean < best_thumb[i][0]:
                        best_thumb[i] = (t_mean, j)
                    if t_mean <= MATCH_MEAN_TOL * 2.0 and t_local <= MATCH_LOCAL_TOL * 2.0:
                        kt = key_tones.get(i, lambda idx: _match_tone(self._read_rgb(self.source_files[idx])))
                        m_err, l_err = _match_pair_err(kt, pt)
                        if m_err <= MATCH_MEAN_TOL and l_err <= MATCH_LOCAL_TOL:
                            cands = pass_cands[i]
                            cands.append((l_err + m_err, m_err, l_err, j))
                            cands.sort(key=lambda x: (x[0], x[3]))
                            if len(cands) > 2:
                                del cands[2:]
                            if m_err < 2e-4 and l_err < 5e-4 and j not in used_j and j not in locked_frames:
                                exact_locked.add(i)
                                locked_frames.add(j)
            unmatched = []
            for i in sorted(pending):
                cands = pass_cands[i]
                if not cands:
                    unmatched.append(i)
                    continue
                best_score, _, _, best_j = cands[0]
                if best_j not in used_j:
                    matched[i] = best_j
                    used_j.add(best_j)
                elif len(cands) > 1 and cands[1][3] not in used_j and cands[1][0] <= best_score + 1e-4:
                    matched[i] = cands[1][3]
                    used_j.add(cands[1][3])
                else:
                    raise KfError("duplicate_keyframe", "two keyframes are the same plate frame")
            if unmatched:
                i0 = unmatched[0]
                name = os.path.basename(self.source_files[i0])
                cj = best_thumb[i0][1] if i0 in best_thumb else 0
                kt0 = key_tones.get(i0, lambda idx: _match_tone(self._read_rgb(self.source_files[idx])))
                m_err, l_err = _match_pair_err(kt0, _match_tone(self._plate_raw(cj)))
                c_num = self.plate_nums[cj]
                raise KfError(
                    "not_plate_frame",
                    "keyframe %s is not a frame of the plate (closest plate frame %s: mean diff %.2f%%, local diff %.1f%%)"
                    % (name, c_num, m_err * 100.0, l_err * 100.0),
                    name=name,
                    count=len(unmatched),
                )
        if len(matched) < 2:
            raise KfError("too_few_keyframes", "at least 2 keyframes are needed (found %d)" % len(matched), count=len(matched), need=2)
        self.keys = sorted(matched.values())
        self.key_source = {j: self.source_files[i] for i, j in matched.items()}
        self.key_target = {j: self.target_files[i] for i, j in matched.items()}
        self.key_num = {j: frame_num(self.source_files[i]) for i, j in matched.items()}  # number in the keyframe's file name
        self.yy, self.xx = np.mgrid[0 : self.h, 0 : self.w].astype(np.float32)
        tick(total, total)
        return self.keys

    # ------------------------------------------------------------------ optical flow of the plate
    def _g8(self, t):
        return np.clip(_srgb(self.S(t)).mean(axis=2) * 255 + 0.5, 0, 255).astype(np.uint8)


    def compute_flows(self, tick=None, engine="dis", raft_max_pixels=None):
        """frame-to-frame flow between the first and the last keyframe, both directions, kept as float16.
        engine "dis": OpenCV DIS, medium preset, finest scale 0, on an 8-bit luma of the displayed plate.
        engine "raft" (where the RAFT engine is installed): torchvision RAFT in a process of its own; when it cannot run or
        fails, DIS is used and ``flow_fallback`` says why.  ``flow_engine`` tells which one the flow comes from."""
        if engine not in FLOW_ENGINES:
            raise KfError("bad_flow", "unknown optical-flow engine %r" % (engine,), flow=str(engine)[:40])
        tick = tick or (lambda done, total: None)
        cv2.setNumThreads(NUM_THREADS)
        lo, hi = self.keys[0], self.keys[-1]
        self._f0, self._f1 = lo, hi
        self.flow_engine, self.flow_fallback, self.flow_info = "dis", None, {}
        self._fl = self._store.alloc((hi - lo, 2, self.h, self.w, 2), np.float16, "flow")
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        dis.setFinestScale(0)
        prev = self._g8(lo)
        for a in range(lo, hi):
            tick(a - lo, hi - lo)
            cur = self._g8(a + 1)
            self._fl[a - lo, 0] = dis.calc(prev, cur, None)
            self._fl[a - lo, 1] = dis.calc(cur, prev, None)
            prev = cur
        tick(hi - lo, hi - lo)

    @property
    def has_flow(self):
        return self._fl is not None

    def forget_frames(self):
        """drop the pictures kept in memory; the flow stays."""
        self._s.clear()
        self._t.clear()
        self._e.clear()

    def _flow(self, a, b):
        return np.asarray(self._fl[min(a, b) - self._f0, 0 if b > a else 1], dtype=np.float32)

    def chain_map(self, t, k, box=None):
        """for every pixel of frame t: where it is in frame k (frame-to-frame flows chained from t towards k).
        ``box`` = (y0, y1, x0, x1): only for that window of frame t - the same values as the full map has there."""
        if box is None:
            mx, my = self.xx.copy(), self.yy.copy()
        else:
            mx, my = self.xx[box[0] : box[1], box[2] : box[3]].copy(), self.yy[box[0] : box[1], box[2] : box[3]].copy()
        s = -1 if k < t else 1
        for a in range(t, k, s):
            # both components in one call: the same values as two single-channel calls, in half the time
            f = cv2.remap(self._flow(a, a + s), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            mx, my = mx + f[..., 0], my + f[..., 1]
        return mx, my

    def link(self, t, k, conf_px):
        """sampling maps that bring something from frame k onto frame t, and the forward-backward confidence 0..1."""
        mx, my = self.chain_map(t, k)
        bx0, by0 = self.chain_map(k, t)
        bx = cv2.remap(bx0, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        by = cv2.remap(by0, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        conf = np.exp(-(np.sqrt((bx - self.xx) ** 2 + (by - self.yy) ** 2) / conf_px) ** 2).astype(np.float32)
        return mx, my, conf

    def flow_report(self):
        """what a result says about the flow: the engine used and, when RAFT was asked for but not used, why."""
        out = {"flow": self.flow_engine}
        if self.flow_fallback:
            out["flow_fallback"] = dict(self.flow_fallback)
        if self.flow_engine == "raft":
            out.update({k: v for k, v in self.flow_info.items() if v is not None})
        return out

    def close(self):
        self._fl = None
        self._s.clear()
        self._t.clear()
        self._e.clear()
        self._store.close()
        if self._tmp_dir:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
            self._tmp_dir = None


def open_shot(source_files, target_files, plate_files, input_colorspace="ACEScg", working_colorspace="ACEScg", ocio_config_path="",
              scratch_dir=None, progress=None, ram_limit=RAM_LIMIT_BYTES, flow="dis", ignore_alpha=False):
    """Load a dataset as a Shot: match the keyframes onto the plate and compute the plate's optical flow
    (``flow``: "dis" or "raft", see ``Shot.compute_flows``).
    ``progress`` is a ``Progress`` whose plan contains the stages "match" and "flow"."""
    if flow not in FLOW_ENGINES:
        raise KfError("bad_flow", "unknown optical-flow engine %r" % (flow,), flow=str(flow)[:40])
    cv2.setNumThreads(NUM_THREADS)
    shot = Shot(source_files, target_files, plate_files, input_colorspace, working_colorspace, ocio_config_path, scratch_dir, ram_limit,
                ignore_alpha=ignore_alpha)
    try:
        shot.match_keyframes((lambda d, n: progress.tick("match", d, n)) if progress else None)
        shot.compute_flows((lambda d, n: progress.tick("flow", d, n)) if progress else None, engine=flow)
    except BaseException:
        shot.close()
        raise
    return shot


# ---------------------------------------------------------------------- propagation
def pick_reference(shot, conf_px=CONF_PX["reference"], tick=None):
    """Automatic reference keyframe: the one whose edit can be carried onto the largest part of the OTHER keyframes.

    reach(r -> t) = mean, over the edited area of keyframe t, of  confidence(flow between t and r) x [r has an edit
    there]  - exactly the quantity the "reference" mode depends on.  A keyframe inside fast motion or motion blur
    cannot be connected to the others by the flow, gets a low reach and is avoided.  (A confidence-WEIGHTED but not
    normalised "difference to the neighbours" would prefer exactly such a keyframe: with no confidence anywhere there
    is nothing left to differ.  That was the first rule; on the real shot it chose the keyframe in the middle of a
    fast head turn.)

      1. mean reach of every keyframe over all other keyframes;
      2. candidates: keyframes with a neighbour on both sides (all keyframes when there is none) whose mean reach is
         within ``REF_TOL`` of the best;
      3. among them the one whose overall edit differs least from its neighbouring keyframes, the difference being
         normalised by the confidence inside the edited area.

    This is a score, not a picture: it is computed in ONE pass over the frames with all keyframes swept together
    (the map "keyframe r -> current frame" is chained exactly, the map back is composed step by step), on a reduced
    picture when the plate has more than ``SELECT_MAX_PIXELS``, and on at most ``SELECT_MAX_KEYS`` evenly spaced
    keyframes.  Returns (plate index of the reference, details for the report)."""
    keys = list(shot.keys)
    if len(keys) > SELECT_MAX_KEYS:
        keys = sorted({keys[int(round(i))] for i in np.linspace(0, len(keys) - 1, SELECT_MAX_KEYS)})
    s = max(1.0, (shot.h * shot.w / float(SELECT_MAX_PIXELS)) ** 0.5)
    ws, hs = (shot.w, shot.h) if s == 1.0 else (max(8, int(round(shot.w / s))), max(8, int(round(shot.h / s))))
    sx, sy = shot.w / float(ws), shot.h / float(hs)
    yy, xx = np.mgrid[0:hs, 0:ws].astype(np.float32)
    small = (lambda a: a) if s == 1.0 else (lambda a: cv2.resize(a, (ws, hs), interpolation=cv2.INTER_AREA))
    lin = lambda z, mx, my: cv2.remap(z, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    lps = lambda z: cv2.GaussianBlur(z, (0, 0), sigmaX=SIG / sx, sigmaY=SIG / sy)

    def flow_small(a, b):
        f = shot._flow(a, b)
        if s == 1.0:
            return np.ascontiguousarray(f[..., 0]), np.ascontiguousarray(f[..., 1])
        f = cv2.resize(f, (ws, hs), interpolation=cv2.INTER_AREA)
        return f[..., 0] / sx, f[..., 1] / sy

    def conf(mx, my, bx0, by0):  # forward-backward confidence on the grid (mx, my) starts from; error in full-size pixels
        bx, by = lin(bx0, mx, my), lin(by0, mx, my)
        return np.exp(-(np.sqrt(((bx - xx) * sx) ** 2 + ((by - yy) * sy) ** 2) / conf_px) ** 2).astype(np.float32)

    edit = {k: small(shot.E(k)) for k in keys}
    edited = {k: np.abs(edit[k]).max(axis=2) > 0.03 for k in keys}
    pos = {k: i for i, k in enumerate(keys)}
    reach = {k: {} for k in keys}
    differ = {k: [] for k in keys}

    def pair(r, t, fx, fy, bx, by):
        """keyframes r < t; (fx, fy): where a pixel of r is in t; (bx, by): where a pixel of t is in r."""
        c_t = conf(bx, by, fx, fy)
        c_r = conf(fx, fy, bx, by)
        r_on_t = np.abs(lps(_warp(np.abs(edit[r]), bx, by))).max(axis=2) > 0.02
        t_on_r = np.abs(lps(_warp(np.abs(edit[t]), fx, fy))).max(axis=2) > 0.02
        reach[r][t] = float((c_t * r_on_t)[edited[t]].mean()) if edited[t].any() else 1.0
        reach[t][r] = float((c_r * t_on_r)[edited[r]].mean()) if edited[r].any() else 1.0
        if pos[t] == pos[r] + 1:  # neighbours: how much do their overall edits differ where the flow connects them
            for me, other, c, mx, my in ((t, r, c_t, bx, by), (r, t, c_r, fx, fy)):
                m = edited[me]
                weight = float(c[m].sum())
                if weight * sx * sy > 1.0:
                    d = np.sqrt(((lps(edit[me]) - lps(_warp(edit[other], mx, my))) ** 2).sum(axis=2))
                    differ[me].append(float((c * d)[m].sum() / weight))

    first, last = keys[0], keys[-1]
    total = sum(last - k for k in keys)
    done = 0
    key_set = set(keys)
    zero = np.zeros_like(xx)
    state = {first: [xx.copy(), yy.copy(), zero, zero]}  # r -> where r's pixels are in the current frame; current frame's pixels in r (as offsets)
    for a in range(first, last):
        ax, ay = flow_small(a, a + 1)
        gx, gy = flow_small(a + 1, a)
        px, py = xx + gx, yy + gy  # where the pixels of frame a+1 are in frame a
        for st in state.values():
            if tick:
                tick(done, total)
            done += 1
            fx, fy, dx, dy = st
            st[0], st[1] = fx + lin(ax, fx, fy), fy + lin(ay, fx, fy)
            st[2], st[3] = gx + lin(dx, px, py), gy + lin(dy, px, py)
        if a + 1 in key_set:
            for r, (fx, fy, dx, dy) in state.items():
                pair(r, a + 1, fx, fy, xx + dx, yy + dy)
            state[a + 1] = [xx.copy(), yy.copy(), zero, zero]
    if tick:
        tick(total, total)

    mean = {k: (float(np.mean(list(reach[k].values()))) if reach[k] else 1.0) for k in keys}
    inner = keys[1:-1] or keys
    best = max(mean[k] for k in inner)
    cand = [k for k in inner if mean[k] >= best - REF_TOL]
    tie = {}
    if len(cand) > 1:
        tie = {k: (float(np.mean(differ[k])) if differ[k] else 1e9) for k in cand}
        cand.sort(key=lambda k: tie[k])
    return int(cand[0]), {
        "mean_reach": {int(shot.plate_nums[k]): round(mean[k], 4) for k in keys},
        "tie_break": {int(shot.plate_nums[k]): round(v, 4) for k, v in tie.items()},
        "scale": round(s, 3),
        "keyframes_used": len(keys),
    }


def _reference_labels(shot, ref, conf_px, tick=None, limit=True):
    """One version for every keyframe: the whole edit comes from the reference keyframe wherever the flow can carry it
    (and the reference actually has an edit there); each keyframe keeps its own edit only where it cannot."""
    keys = shot.keys
    labels = shot._store.alloc((len(keys), shot.h, shot.w, 3), np.float32, "labels")
    e_ref = shot.E(ref).copy()
    abs_ref = np.abs(e_ref)
    reach = {}
    for i, t in enumerate(keys):
        if tick:
            tick(i, len(keys))
        if t == ref:
            ef = e_ref
        else:
            mx, my, c = shot.link(t, ref, conf_px)
            g = _carry(shot, t, ref, e_ref, mx, my, limit)
            vis = c[..., None] * (np.abs(_lp(_warp(abs_ref, mx, my))).max(axis=2, keepdims=True) > 0.02)
            e_t = shot.E(t)
            ef = vis * g + (1.0 - vis) * e_t
            edited = np.abs(e_t).max(axis=2) > 0.03
            reach[t] = float(vis[..., 0][edited].mean()) if edited.any() else 1.0
        lf = _lp(ef)
        hf = ef - lf
        labels[i] = shot.from_log(t, lf + hf)
    return labels, reach


def _detail_labels(shot, ref, conf_px, tick=None):
    """Only the DETAIL comes from the reference keyframe: the part of its edit finer than ``DETAIL_SIGMA`` (scars,
    cracks, texture) replaces the same band of every other keyframe wherever the flow can carry it (and the reference
    actually has an edit there).  Everything coarser - the overall look, the lighting and the strength the artist gave
    each keyframe - stays that keyframe's own, so BROAD paint made for the lighting of the reference frame is not
    carried onto differently lit frames.  Thin highlights and fine cracks of the reference ARE detail and appear on
    every frame (seen on the real shot: a thin wet highlight of keyframe 11 on all frames).  The reference keyframe
    itself is left untouched.

    A band of the edit is carried as it is (plain warp of the log ratio): the highlight limit of ``_carry`` is for a
    whole edit layer, not for a band of one."""
    keys = shot.keys
    labels = shot._store.alloc((len(keys), shot.h, shot.w, 3), np.float32, "labels")
    low = lambda z: cv2.GaussianBlur(z, (0, 0), DETAIL_SIGMA)
    e_ref = shot.E(ref).copy()
    abs_ref = np.abs(e_ref)
    hf_ref = e_ref - low(e_ref)
    reach = {}
    for i, t in enumerate(keys):
        if tick:
            tick(i, len(keys))
        if t == ref:
            labels[i] = shot.T(t)
            continue
        mx, my, c = shot.link(t, ref, conf_px)
        g = _warp(hf_ref, mx, my)
        vis = c[..., None] * (np.abs(_lp(_warp(abs_ref, mx, my))).max(axis=2, keepdims=True) > 0.02)
        e_t = shot.E(t)
        lf_t = low(e_t)
        ef = vis * g + (1.0 - vis) * (e_t - lf_t)
        labels[i] = shot.from_log(t, lf_t + ef)
        edited = np.abs(e_t).max(axis=2) > 0.03
        reach[t] = float(vis[..., 0][edited].mean()) if edited.any() else 1.0
    return labels, reach


def _from_neighbours(shot, t, keys, e_of, conf_px, limit):
    """Frame t made from its two nearest keyframes: their edits (``e_of(k)``) carried along the chained flow and
    blended by distance and forward-backward confidence.  Returns (label, per-pixel confidence of the blend)."""
    acc = wsum = csum = None
    for k in _neighbours(t, keys):
        mx, my, c = shot.link(t, k, conf_px)
        g = _carry(shot, t, k, e_of(k), mx, my, limit)
        w = c / abs(t - k) + 1e-4
        acc = g * w[..., None] if acc is None else acc + g * w[..., None]
        wsum = w if wsum is None else wsum + w
        csum = c * w if csum is None else csum + c * w
    return shot.from_log(t, acc / wsum[..., None]), (csum / wsum).astype(np.float32)


def propagate(shot, out_dir, mode="nearest", ref_frame=None, progress=None, limit_highlights=True):
    """Write out_dir/source and out_dir/target for every plate frame from the first to the last keyframe.

    mode "nearest":   every keyframe is kept; an in-between frame gets the two nearest keyframes' edits carried along
                      the chained flow, blended by distance and forward-backward confidence (scale 2 px).
    mode "reference": the whole edit comes from ONE keyframe wherever the flow can carry it (scale 6 px); the other
                      keyframes only fill what it cannot reach.  ``ref_frame`` = the frame number in that keyframe's
                      file name (what the user sees in the dataset; for an uploaded dataset this is not the plate
                      frame number); None = chosen automatically (``pick_reference``).
    mode "detail":    only the detail (finer than ``DETAIL_SIGMA``) comes from the reference keyframe; every keyframe
                      keeps its own overall look, lighting and strength (``_detail_labels``).  Same ``ref_frame`` rule,
                      same scale (6 px); in-between frames are made from the two nearest rewritten keyframes as in
                      the other modes.
    ``limit_highlights``: a brightening is never carried as more than its additive amount (see ``_carry``); False
                      reproduces the first version of this module (plain log ratio everywhere).
    ``progress``: a ``Progress`` whose plan contains "select" (automatic reference only), "reference" and "write".
    Returns a dict describing the result.
    """
    if mode not in CONF_PX:
        raise KfError("bad_mode", "unknown mode %r" % (mode,))
    cv2.setNumThreads(NUM_THREADS)
    t_start = time.time()
    conf_px = CONF_PX[mode]
    keys = shot.keys
    key_set = set(keys)
    f0, f1 = keys[0], keys[-1]
    tick = (lambda stage: (lambda d, n: progress.tick(stage, d, n))) if progress else (lambda stage: None)
    ref = None
    ref_auto = False
    reach = {}
    selection = {}
    labels = None
    if mode in ("reference", "detail"):
        if ref_frame is None:
            ref_auto = True
            ref, selection = pick_reference(shot, conf_px, tick("select"))
        else:
            cand = [k for k in keys if shot.key_num[k] == int(ref_frame)]
            if not cand:
                have = [shot.key_num[k] for k in keys if shot.key_num[k] is not None]
                raise KfError("ref_not_keyframe", "reference frame %s is not one of the keyframes %s" % (ref_frame, have),
                              frame=int(ref_frame), frames=", ".join(str(v) for v in have))
            ref = cand[0]
        if mode == "detail":
            labels, reach = _detail_labels(shot, ref, conf_px, tick("reference"))
        else:
            labels, reach = _reference_labels(shot, ref, conf_px, tick("reference"), limit=bool(limit_highlights))
        index_of = {k: i for i, k in enumerate(keys)}
        label_of = lambda k: np.asarray(labels[index_of[k]], dtype=np.float32)
    else:
        label_of = shot.T
    e_cache = _Lru(3)
    e_of = lambda k: e_cache.get(k, lambda i: shot.to_log(i, label_of(i)))

    width = max(4, len(str(max(shot.plate_nums[f0 : f1 + 1]))))
    src_dir, tgt_dir = os.path.join(out_dir, "source"), os.path.join(out_dir, "target")
    os.makedirs(src_dir, exist_ok=True)
    os.makedirs(tgt_dir, exist_ok=True)
    conf_mean, low_conf_frames = [], []
    tk = tick("write")
    for t in range(f0, f1 + 1):
        if tk:
            tk(t - f0, f1 - f0 + 1)
        if t in key_set:
            lab = label_of(t)
        else:
            lab, conf = _from_neighbours(shot, t, keys, e_of, conf_px, bool(limit_highlights))
            edited = np.abs(lab - shot.S(t)).max(axis=2) >= 5e-3
            cm = float(conf[edited].mean()) if edited.any() else 1.0
            conf_mean.append(cm)
            if cm < 0.5:
                low_conf_frames.append(int(shot.plate_nums[t]))
        num = shot.plate_nums[t]
        color.write_exr_file(os.path.join(src_dir, "source.%0*d.exr" % (width, num)), shot.S(t), bit_depth="half")
        color.write_exr_file(os.path.join(tgt_dir, "target.%0*d.exr" % (width, num)), np.maximum(lab, 0.0).astype(np.float32), bit_depth="half")
    if tk:
        tk(f1 - f0 + 1, f1 - f0 + 1)
    labels = None
    return {
        "mode": mode,
        **shot.flow_report(),  # "flow": "dis" | "raft" (+ "flow_fallback" when RAFT was asked for and could not be used)
        "conf_px": conf_px,
        "limit_highlights": bool(limit_highlights),
        "detail_sigma": DETAIL_SIGMA if mode == "detail" else None,
        "frame_count": f1 - f0 + 1,
        "first_frame": int(shot.plate_nums[f0]),
        "last_frame": int(shot.plate_nums[f1]),
        "keyframes": [int(shot.plate_nums[k]) for k in keys],
        "keyframe_files": [os.path.basename(shot.key_source[k]) for k in keys],
        "propagated_frames": f1 - f0 + 1 - len(keys),
        "ref_frame": int(shot.plate_nums[ref]) if ref is not None else None,  # plate frame number
        "ref_file": os.path.basename(shot.key_source[ref]) if ref is not None else None,
        "ref_auto": bool(ref_auto),
        "ref_reach": {str(shot.plate_nums[k]): round(v, 4) for k, v in reach.items()},  # the chosen reference -> every other keyframe
        # automatic choice only: what every keyframe could supply as the reference (mean over the other keyframes),
        # and the tie-break among the best ones (difference of the overall edit to the neighbouring keyframes)
        "ref_candidates": {str(k): v for k, v in (selection.get("mean_reach") or {}).items()},
        "ref_tie_break": {str(k): v for k, v in (selection.get("tie_break") or {}).items()},
        "ref_select_scale": selection.get("scale"),
        "mean_conf_edited": float(np.mean(conf_mean)) if conf_mean else 1.0,
        "min_conf_edited": float(np.min(conf_mean)) if conf_mean else 1.0,
        "low_conf_frames": low_conf_frames,
        "plate_frames": shot.n,
        "width": shot.w,
        "height": shot.h,
        "matched_by_number": shot.matched_by_number,
        "buffers_on_disk": bool(shot._store.mapped),
        "seconds": round(time.time() - t_start, 1),
    }


def write_card(out_dir, info, name, parent_name, internal=False):
    """CARD.md of a propagated dataset (Chinese): what was done, with what, and what not to trust."""
    keys = info["keyframes"]
    is_detail = info["mode"] == "detail"
    extra_limit = ""
    if info["mode"] in ("reference", "detail"):
        chosen = "自动选择：沿光流能覆盖其它关键帧最多的一张（处在快速运动、运动模糊里的关键帧会被避开）" if info["ref_auto"] else "用户指定"
        if is_detail:
            how = ("细节以一张关键帧为准（detail）：只有细节——比 %g 像素（高斯 σ）更细的结构，如疤痕、裂纹、纹理——取自参考关键帧，即底板第 **%d** 帧（关键帧文件 `%s`，%s），"
                   "沿光流能搬到的地方都用它的细节。比这更粗的部分（整体明暗、光照、强弱）每张关键帧都保留美术师自己画的，"
                   "所以参考关键帧上大块的明暗不会被搬到别的帧上；但它上面画的细高光、细裂纹属于细节，会出现在每一帧上。参考关键帧本身保持美术师原稿；其余关键帧位置上 target 的细节被改写过。置信尺度 %g 像素。"
                   % (info.get("detail_sigma") or DETAIL_SIGMA, info["ref_frame"], info.get("ref_file") or "", chosen, info["conf_px"]))
            extra_limit = "关键帧之间位置差得多的地方，参考关键帧的细节会叠在那张关键帧自己的整体形状上，可能出现重影。"
        else:
            how = ("以一张关键帧为准（reference）：整个修改取自参考关键帧——底板第 **%d** 帧（关键帧文件 `%s`，%s），沿光流能搬到的地方都用它；其它关键帧只补它够不到的地方。"
                   "因此除参考关键帧外，其余关键帧位置上的 target 也被改写过，不再是美术师原稿。置信尺度 %g 像素。"
                   % (info["ref_frame"], info.get("ref_file") or "", chosen, info["conf_px"]))
            extra_limit = "参考关键帧上按当时光照画的暗部和整体强弱也会被搬到其它帧，光照变化大的镜头尤其要检查。"
        pct = lambda d: "；".join("第 %s 帧 %.0f%%" % (k, 100 * v) for k, v in d.items())
        cands = info.get("ref_candidates") or {}
        if cands:
            tie = info.get("ref_tie_break") or {}
            how += ("\n- 各关键帧作参考时，沿光流能覆盖其它关键帧修改区域的平均比例：%s。自动选择在比例最高的几张（相差不超过 %d 个百分点）里%s。"
                    % (pct(cands), round(100 * REF_TOL),
                       ("取整体外观与相邻关键帧差别最小的一张（差别：%s）" % "；".join("第 %s 帧 %.3f" % (k, v) for k, v in tie.items())) if tie else "只有这一张"))
        how += ("\n- 参考关键帧的%s在其它关键帧修改区域内的覆盖率：%s。覆盖率低的关键帧基本保留自己的%s。"
                % ("细节" if is_detail else "修改", pct(info.get("ref_reach") or {}) or "—", "细节" if is_detail else "版本"))
        key_txt = ("其中 %d 帧在关键帧位置（底板帧号 %s），只有参考关键帧保持美术师原稿%s"
                   % (len(keys), keys, "，其余关键帧保留自己的整体明暗和强弱、细节换成参考关键帧的" if is_detail else ""))
    else:
        how = ("保留每张关键帧（nearest）：关键帧的 target 就是美术师原稿；中间帧取相邻两张关键帧的修改，沿光流搬过来后按距离和光流置信度混合。置信尺度 %g 像素。"
               % info["conf_px"])
        key_txt = "其中 %d 帧是美术师关键帧（底板帧号 %s）" % (len(keys), keys)
    low = info.get("low_conf_frames") or []
    lines = [
        "# %s — 关键帧传播数据集" % name,
        "",
    ]
    if internal:
        lines += ["> **仅内部使用，不得对外。**（由内部数据集 `%s` 生成）" % parent_name, ""]
    lines += [
        "- 来源：数据集 `%s`（关键帧 %d 张 + 整条底板 %d 帧）。" % (parent_name, len(keys), info["plate_frames"]),
        "- 做法：把手绘关键帧的**修改**（目标/输入的对数比，不是画面本身）沿底板自身的逐帧光流（%s）搬到第一张与最后一张关键帧之间的每一帧。" % ("RAFT" if info.get("flow") == "raft" else "OpenCV DIS")
        + ("变暗、变色按比例搬运（跟随底板的明暗）；**提亮**的部分不超过它在关键帧上的绝对增量，避免底板更亮的帧上出现过曝的高光。" if info.get("limit_highlights") else "")
        + "训练代码没有改动，导出的模型仍是普通模型。",
        "- 模式：%s" % how,
        "- 内容：`source/` 与 `target/` 各 %d 帧（底板第 %d–%d 帧，%d×%d，文件名里的帧号就是底板帧号）；%s；其余 %d 帧的 target 是传播得到的。`eval/` 是原数据集的整条底板。"
        % (info["frame_count"], info["first_frame"], info["last_frame"], info["width"], info["height"], key_txt, info["propagated_frames"]),
        "- 修改区域内的平均光流置信度：%.2f（逐帧最低 %.2f）。%s"
        % (info["mean_conf_edited"], info["min_conf_edited"], ("置信度低于 0.5 的帧：%s，请重点检查。" % low) if low else ""),
        "- 限制：中间帧的 target 是**传播得到的，不是美术师作品**；结果依赖光流，遮挡、张嘴、快速运动、运动模糊处可能出错；第一张关键帧之前和最后一张之后的帧不在数据集内。"
        + extra_limit + "训练前请在数据集页面拖动滑条逐帧检查。",
        "- 依据：在一条真实镜头上，用普通训练对比：中间帧效果不再消失，整体外观游移降到约 1/3。其它镜头尚未验证。"
        + ("细节模式在同一条镜头上的一次对比里：纹理的稳定程度与「以一张关键帧为准」相当，整体外观的游移介于另外两种模式之间。" if is_detail else ""),
        "- 生成：WildCat 关键帧传播（`wildcat/kfprop.py`，由接口 `POST /api/v1/assets/<id>/kfprop` 调用；界面上没有入口），%s，耗时 %.0f 秒。" % (time.strftime("%Y-%m-%d %H:%M"), info.get("seconds", 0.0)),
        "",
    ]
    with open(os.path.join(out_dir, "CARD.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# ---------------------------------------------------------------------- consistency check
def _verdict(band, value):
    if value is None:
        return None
    good, fair = BAND_THRESHOLDS[band]
    return "ok" if value >= good else ("mostly" if value >= fair else "bad")


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def mode_advice(verdict):
    """Which propagation mode the overall verdicts of the consistency check suggest (a pure function of the verdicts):

      "nearest"  all three bands "ok" or "mostly": the keyframes agree, keep every keyframe.  On keyframes that are
                 consistent the two one-reference modes only move the in-between frames away from the truth.
      "detail"   the broad look agrees ("ok" / "mostly") but texture or fine detail is "bad": one version of the
                 detail (mode "detail") is worth considering; "nearest" still keeps the effect on in-between frames.
      "inspect"  the broad look itself is "bad", or a band could not be measured: look at the heat maps first.

    It is advice for the reader of the check, never a setting: nothing selects a mode from it.  The page (Tools tab)
    does not name modes at all - propagation has no entry in the UI - and shows for each code only what the check
    found and where to look (locale keys ``priv.kfAdvice_<code>``)."""
    v = verdict or {}
    broad, medium, fine = v.get("broad"), v.get("medium"), v.get("fine")
    if None in (broad, medium, fine) or broad == "bad":
        return "inspect"
    if "bad" in (medium, fine):
        return "detail"
    return "nearest"


def consistency(shot, jpg_path=None, progress=None, max_rows=24, tile_width=560):
    """How much of each keyframe's edit do the neighbouring keyframes explain after being carried onto it along the
    flow?  Three bands (broad > 10 px, medium 4-10 px, fine 1.5-4 px); 1.0 = identical, 0 = nothing explained,
    negative = the neighbour's version is further away than no edit at all.  Optionally writes one JPG with a row per
    keyframe: the keyframe, where the overall look differs, where the texture differs.
    ``progress``: a ``Progress`` whose plan contains "compare"."""
    cv2.setNumThreads(NUM_THREADS)
    keys = shot.keys
    g = lambda z, s: cv2.GaussianBlur(z, (0, 0), s)
    bands = (("broad", lambda e: g(e, 10)), ("medium", lambda e: g(e, 4) - g(e, 10)), ("fine", lambda e: g(e, 1.5) - g(e, 4)))
    to8 = lambda z, sc: np.clip(127.5 + 127.5 * z / sc, 0, 255).astype(np.uint8)
    conf_px = CONF_PX["nearest"]
    scale = min(1.0, float(tile_width) / shot.w)
    rows, rep = [], []

    def tile(img, text):
        if scale < 1.0:
            img = cv2.resize(img, (max(1, int(round(shot.w * scale))), max(1, int(round(shot.h * scale)))), interpolation=cv2.INTER_AREA)
        img = np.ascontiguousarray(img)
        cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        return img

    def heat(v, vmax, mask):
        h = cv2.applyColorMap(np.clip(v / vmax * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        h[~mask] //= 5
        return h

    for i, t in enumerate(keys):
        if progress:
            progress.tick("compare", i, len(keys))
        nb = [k for k in (keys[i - 1] if i > 0 else None, keys[i + 1] if i + 1 < len(keys) else None) if k is not None]
        e_t = shot.E(t)
        ed = np.abs(e_t).max(axis=2) > 0.03
        acc = {b: [] for b, _ in bands}
        offs, gains, cov = [], [], []
        dmap = np.zeros((shot.h, shot.w), np.float32)
        tmap = np.zeros((shot.h, shot.w), np.float32)
        for k in nb:
            mx, my, c = shot.link(t, k, conf_px)
            ek = _warp(shot.E(k), mx, my)
            m = ed & (c > 0.6)
            cov.append(float(m.sum() / max(ed.sum(), 1)))
            if m.sum() < 500:
                continue
            for b, f in bands:
                a, bb = f(e_t), f(ek)
                acc[b].append(1.0 - float(((a - bb) ** 2)[m].sum() / max(float((a**2)[m].sum()), 1e-12)))
            # how far the neighbour's edit sits off after the plate motion is removed (only meaningful when the texture matches)
            a8 = g(e_t, 2.0).mean(axis=2)
            b8 = g(ek, 2.0).mean(axis=2)
            sc = 4 * float(a8.std()) + 1e-6
            dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            dis.setFinestScale(1)
            dis.setPatchSize(16)
            dis.setPatchStride(4)
            d = cv2.GaussianBlur(dis.calc(to8(a8, sc), to8(b8, sc), None), (0, 0), 8)
            offs.append(float(np.median(np.sqrt((d**2).sum(axis=2))[m])))
            la, lb = g(np.abs(e_t).mean(axis=2), 10), g(np.abs(ek).mean(axis=2), 10)
            r = np.log((lb + 1e-3) / (la + 1e-3))
            gains.append(float(np.median(np.abs(r[m]))))
            dmap = np.maximum(dmap, np.sqrt(((g(e_t, 10) - g(ek, 10)) ** 2).sum(axis=2)) * c)  # where the broad look differs
            ma, mb = g(e_t, 4) - g(e_t, 10), g(ek, 4) - g(ek, 10)
            num = g((ma * mb).sum(axis=2), 12)
            den = np.sqrt(g((ma**2).sum(axis=2), 12) * g((mb**2).sum(axis=2), 12)) + 1e-9
            tmap = np.maximum(tmap, (1.0 - num / den) * c)  # 0 = same texture, 1 = unrelated, 2 = opposite
        vals = {b: (float(np.mean(acc[b])) if acc[b] else None) for b, _ in bands}
        medium_ok = vals["medium"] is not None and vals["medium"] >= OFFSET_MIN_MEDIUM
        r = {
            "frame": int(shot.plate_nums[t]),  # plate frame number
            "name": os.path.basename(shot.key_source[t]),  # the keyframe's file in the dataset
            "neighbours": [int(shot.plate_nums[k]) for k in nb],
            "edited_area": float(ed.mean()),
            "comparable_share": float(np.mean(cov)) if cov else None,
            "broad": vals["broad"],
            "medium": vals["medium"],
            "fine": vals["fine"],
            "verdict": {b: _verdict(b, vals[b]) for b, _ in bands},
            "offset_px": (float(np.mean(offs)) if (offs and medium_ok) else None),
            "strength_diff": float(np.expm1(np.mean(gains))) if gains else None,
        }
        rep.append(r)
        if jpg_path:
            pct = lambda v: ("%.0f%%" % (100 * v)) if v is not None else "n/a"
            img = (_srgb(shot.T(t))[..., ::-1] * 255 + 0.5).astype(np.uint8)
            rows.append((r["broad"] if r["broad"] is not None else 2.0, i, np.concatenate([
                tile(img, "keyframe %d" % r["frame"]),
                tile(heat(dmap, 0.5, ed), "overall look differs (agreement %s)" % pct(r["broad"])),
                tile(heat(tmap, 1.5, ed), "texture differs (agreement %s)" % pct(r["medium"])),
            ], axis=1)))
    if progress:
        progress.tick("compare", len(keys), len(keys))

    summary = {b: _mean([r[b] for r in rep]) for b, _ in bands}
    medium_ok = summary["medium"] is not None and summary["medium"] >= OFFSET_MIN_MEDIUM
    verdict = {b: _verdict(b, summary[b]) for b, _ in bands}
    out = {
        "keyframes": [int(shot.plate_nums[k]) for k in keys],
        "plate_frames": shot.n,
        "width": shot.w,
        "height": shot.h,
        **shot.flow_report(),
        "broad": summary["broad"],
        "medium": summary["medium"],
        "fine": summary["fine"],
        "verdict": verdict,
        "advice": mode_advice(verdict),  # "nearest" | "detail" | "inspect": which mode the verdicts suggest (see mode_advice)
        "offset_px": _mean([r["offset_px"] for r in rep]) if medium_ok else None,
        "strength_diff": _mean([r["strength_diff"] for r in rep]),
        "thresholds": {b: list(v) for b, v in BAND_THRESHOLDS.items()},
        "per_keyframe": rep,
        "image_frames": [],
    }
    if jpg_path and rows:
        if len(rows) > max_rows:  # too many keyframes for one picture: show the least consistent ones
            rows = sorted(sorted(rows, key=lambda x: x[0])[:max_rows], key=lambda x: x[1])
        out["image_frames"] = [rep[i]["frame"] for _, i, _ in rows]
        os.makedirs(os.path.dirname(os.path.abspath(jpg_path)), exist_ok=True)
        if not color.imwrite_unicode(jpg_path, np.concatenate([x[2] for x in rows], axis=0), [cv2.IMWRITE_JPEG_QUALITY, 88]):
            raise IOError("cannot write " + jpg_path)
    return out


# ---------------------------------------------------------------------- keyframe coverage
def suggest_keyframes(cov, keys, threshold=COVERAGE_THRESHOLD):
    """One suggestion per gap between neighbouring keyframes whose lowest coverage is below the threshold: the frame
    with the lowest value (several frames within 0.01 of it: the middle one).  ``cov``: {plate index: coverage}."""
    out = []
    for a, b in zip(keys[:-1], keys[1:]):
        gap = list(range(a + 1, b))
        if not gap:
            continue
        lo = min(cov[t] for t in gap)
        if lo < threshold:
            cand = [t for t in gap if cov[t] <= lo + 0.01]
            out.append({"between": [a, b], "frame": cand[len(cand) // 2], "coverage": lo})
    return out


def coverage(shot, progress=None, threshold=COVERAGE_THRESHOLD):
    """Which frames of the shot are too far from every keyframe, and where would one more keyframe help most?

    A network trained on a few keyframes does well on frames that look like one of them.  Between two keyframes that
    are far apart in pose (a head turn, a change of expression, something moving in front) the effect fades or
    disappears.  For every plate frame between the first and the last keyframe this follows the edited area of the
    two nearest keyframes along the plate's own optical flow and measures how much of it can be followed reliably:

      coverage(t) = mean, over the edited area of frame t, of the forward-backward confidence of the two nearest
                    keyframes' edits carried onto t  -  the very number ``propagate`` (mode "nearest", plain log
                    ratio) reports per frame.  1.0 = fully connected to its keyframes; a keyframe itself is 1.0.

    Nothing is trained and nothing is written.  A gap whose lowest frame is below ``threshold`` gets one suggestion:
    that lowest frame (``suggest_keyframes``).  Frame numbers in the result are plate frame numbers.
    ``progress``: a ``Progress`` whose plan contains "cover"."""
    cv2.setNumThreads(NUM_THREADS)
    t_start = time.time()
    keys = shot.keys
    key_set = set(keys)
    conf_px = CONF_PX["nearest"]
    f0, f1 = keys[0], keys[-1]
    frames = list(range(f0, f1 + 1))
    todo = [t for t in frames if t not in key_set]
    cov = {k: 1.0 for k in keys}
    for i, t in enumerate(todo):
        if progress:
            progress.tick("cover", i, len(todo))
        lab, conf = _from_neighbours(shot, t, keys, shot.E, conf_px, False)
        edited = np.abs(lab - shot.S(t)).max(axis=2) >= 5e-3
        cov[t] = float(conf[edited].mean()) if edited.any() else 1.0
    if progress:
        progress.tick("cover", len(todo), len(todo))
    nums = shot.plate_nums
    worst = min(frames, key=lambda t: cov[t])
    return {
        "threshold": float(threshold),
        **shot.flow_report(),
        "conf_px": conf_px,
        "keyframes": [int(nums[k]) for k in keys],
        "keyframe_files": [os.path.basename(shot.key_source[k]) for k in keys],
        "frames": [int(nums[t]) for t in frames],
        "coverage": [round(cov[t], 4) for t in frames],
        "suggest": [{"frame": int(nums[s["frame"]]), "between": [int(nums[s["between"][0]]), int(nums[s["between"][1]])], "coverage": round(s["coverage"], 4)}
                    for s in suggest_keyframes(cov, keys, threshold)],
        "below_threshold": sum(1 for t in frames if cov[t] < threshold),
        "min_coverage": round(cov[worst], 4),
        "min_frame": int(nums[worst]),
        "plate_frames": shot.n,
        "width": shot.w,
        "height": shot.h,
        "seconds": round(time.time() - t_start, 1),
    }


# ---------------------------------------------------------------------- which keyframe is the odd one, and where
def odd_keyframes(shot, jpg_path=None, progress=None, max_rows=12, tile_width=560):
    """Which keyframe is the odd one, and where?  Nothing is trained and nothing is written to the dataset.

    Runs the "keyframe commonality" computation (wildcat/kfcommon.py - the optional advanced module ``kf_common``) with
    its default settings and rules, whatever they currently are, and reports what the module WOULD rewrite before
    training: for every keyframe the share of its edited area that would be rewritten (``replaced_area``, as the
    module defines it) and how much the keyframes around it have in common there, band by band.  The figures are the
    module's own - nothing is added to them here - so the tool always shows what a training run with the module
    switched on would do.  Unlike ``consistency``, which compares neighbouring keyframes pairwise and cannot say WHO is
    off, a place counts here only when the other keyframes agree with each other and this one does not; where every
    keyframe differs and none is in the majority nothing is marked.

    ``jpg_path``: one picture, a row per rewritten keyframe (as painted | what would be trained on | amount of
    change), at most ``max_rows`` rows - the keyframes rewritten most.  Needs ``ODD_MIN_KEYFRAMES`` keyframes.
    Frame numbers in the result are plate frame numbers.  ``progress``: a ``Progress`` whose plan contains "odd"."""
    from . import kfcommon  # (kfcommon imports this module)

    cv2.setNumThreads(NUM_THREADS)
    t_start = time.time()
    keys = shot.keys
    if len(keys) < ODD_MIN_KEYFRAMES:
        raise KfError("too_few_keyframes", "at least %d keyframes are needed to tell which one is the odd one (found %d)" % (ODD_MIN_KEYFRAMES, len(keys)),
                      count=len(keys), need=ODD_MIN_KEYFRAMES)
    view = kfcommon.ShotView(shot)
    targets, rep = kfcommon.common_targets(view, keep_maps=True, tick=(lambda d, n: progress.tick("odd", d, n)) if progress else None)
    nums = shot.plate_nums
    drawn = []
    if jpg_path and targets:
        os.makedirs(os.path.dirname(os.path.abspath(jpg_path)), exist_ok=True)
        if not kfcommon.write_sheet(view, targets, rep, jpg_path, frame_label=lambda k: str(nums[k]), max_rows=max_rows, tile_width=tile_width):
            raise IOError("cannot write " + jpg_path)
        # the rows write_sheet draws: the keyframes rewritten most, in temporal order
        drawn = sorted(r["key"] for r in sorted((r for r in rep["keyframes"] if r["key"] in targets), key=lambda r: -r["replaced_area"])[:max_rows])
    worst = max(rep["keyframes"], key=lambda r: r["replaced_area"])
    return {
        **shot.flow_report(),
        "keyframes": [int(nums[k]) for k in keys],
        "plate_frames": shot.n,
        "width": shot.w,
        "height": shot.h,
        "threshold": rep["tau2"],  # the module's settings (its defaults: what a training run with the module on would use)
        "span": rep["span"],
        "conf_px": rep["conf_px"],
        "per_keyframe": [{
            "frame": int(nums[r["key"]]),  # plate frame number
            "name": os.path.basename(shot.key_source[r["key"]]),  # the keyframe's file in the dataset
            "edited_area": float(r["edited_px"]) / float(shot.h * shot.w),
            "replaced_area": float(r["replaced_area"]),  # share of the edited area that would be rewritten
            "replaced_mean": float(r.get("replaced_mean", 0.0)),
            "change_mean": float(r.get("change_mean", 0.0)),  # mean change of the edit inside the edited area (log units)
            "commonality": {b: float(v) for b, v in r["commonality"].items()},  # how much the keyframes around it have in common there
            "rewritten": r["key"] in targets,
        } for r in rep["keyframes"]],
        "replaced_area": float(rep["replaced_area"]),  # mean over the keyframes
        "replaced_mean": float(rep.get("replaced_mean", 0.0)),
        "change_mean": float(rep.get("change_mean", 0.0)),
        "commonality": {b: float(v) for b, v in rep["commonality"].items()},
        "worst_frame": int(nums[worst["key"]]),
        "worst_area": float(worst["replaced_area"]),
        "nothing_stands_out": bool(rep["replaced_area"] < ODD_NOTHING),
        "rewritten_frames": [int(nums[k]) for k in sorted(targets)],
        "image_frames": [int(nums[k]) for k in drawn],
        "seconds": round(time.time() - t_start, 1),
    }
