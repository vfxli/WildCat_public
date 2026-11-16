# -*- coding: utf-8 -*-
"""Quality monitor: what the model does on the frames BETWEEN the keyframes, measured while it trains.

Nuke's CopyCat shows one number during training: the loss on random crops of the keyframes.  It says how well the
keyframes themselves are reproduced; the shot is delivered on the frames in between, where no ground truth exists.
At checkpoint time this module measures, without ground truth, what an artist would look at there:

  keyframe fit      PSNR of the output against the keyframe targets the trainer trains on (after kf_common, if on),
                    over the whole frame and inside the edited area
  strength          is the edit there at all on an in-between frame: energy of the overall look of the model's edit,
                    relative to the keyframes around that frame (1.0 = as strong as painted)
  detail, texture   energy of all the fine detail / of the 3-10 px texture of the model's edit inside the edited area,
                    relative to the painted keyframes (1.0 = as much as painted, lower = blurred)
  drift             how much the overall look / the texture changes when it is followed along the plate's motion for
                    1 and 5 frames (lower = it sticks to the surface); the plate's own texture followed the same way
                    is the measurement floor
  background        mean change of the picture outside the edited area

Setting: an RGB -> RGB dataset with a plate (the evaluation sequence) that contains the keyframes; the frames from
the first to the last keyframe are evaluated (at most ``MAX_FRAMES``, evenly spaced beyond that).  With the plate S,
the model output P and ``E = log((max(P, 0) + EPS) / (max(S, 0) + EPS))``: overall look ``L`` = Gaussian sigma 3 of
E, all detail ``H = E - L``, texture ``M`` = Gaussian 1.5 minus Gaussian 5 of E, edited area ``ROI`` = the pixels
where a channel of P differs from S by ``EDIT_MIN`` or more (the model's own edit at this evaluation).
Any other dataset makes the monitor unavailable with a reason code; the training is never affected.

A MEASUREMENT, NOT AN ENHANCEMENT.  The monitor only reads the frames the trainer has loaded and runs inference on
weight snapshots in the validation worker (eval mode, no_grad, the weights validation uses).  It never touches the
training RNG, the sampler, the optimiser state or the label arrays: with ``io.quality_monitor`` on or off the trained
weights are identical (tools/test_quality.py trains the same configuration both ways and compares them bit for bit).

Cost: the plate is read and its flow computed once, in a thread beside the training.  An evaluation is one inference
per evaluated frame plus some arithmetic in a few threads (71 frames of 772x855, Small model: about 1.7 s); the
evaluations may take ``BUDGET`` of the elapsed training time (``TrainingMonitor.due``).  When validation has a card of
its own they run beside the training; on the training card the training thread waits for an evaluation instead
(``Trainer._preview``) - sharing the card with running training steps would stretch it to 5-11 s.

Output: ``run/<task>/quality.jsonl`` - a header line, then one line per evaluation (see ``TrainingMonitor``).
"""
import atexit
import bisect
import concurrent.futures
import functools
import json
import math
import os
import shutil
import threading
import time

import cv2
import numpy as np

try:
    from . import kfprop
except ImportError:  # an edition without the keyframe tools has no monitor (see for_trainer)
    kfprop = None

EPS = kfprop.EPS if kfprop else 0.02
SIG = kfprop.SIG if kfprop else 3.0  # "overall look": Gaussian sigma (px) of the edit layer
TEX_SIGMAS = (1.5, 5.0)  # "texture": Gaussian 1.5 minus Gaussian 5 of the edit layer (structures of about 3-10 px)
EDIT_MIN = 5e-3  # a pixel counts as edited when one channel differs from the plate by at least this
ROI_SHARE_MIN = 0.002  # a frame whose edited area covers less of the picture than this is skipped
FIT_ROI_SHARE = (0.0005, 0.90)  # keyframe fit inside the edited area: the rule of the recommender's roi_psnr
MAX_FRAMES = 96  # frames evaluated between the first and the last keyframe
BG_DILATE_PX = 8  # "outside the edited area": this far from it
BUDGET = 0.03  # the evaluations may take this share of the elapsed training time
FIRST_AFTER = 0.02  # the first evaluation is allowed after this share of the total steps
FINAL_WAIT_SEC = 60.0  # the evaluation at the end of a run waits this long for a plate that is still being prepared
MAX_FAILURES = 3  # evaluations that failed in a row before the monitor gives up for the run
RAM_LIMIT_BYTES = 1024**3  # flow / plate buffers beyond this go to a memory-mapped scratch file in the run folder
WORKERS = 4  # threads for the arithmetic of an evaluation (inference stays in the calling thread)
WORK_RAM_BYTES = 1.5e9  # ... fewer when that many frames under way would need more memory than this
FILE_NAME = "quality.jsonl"
SCRATCH = "_quality_tmp"


# ---------------------------------------------------------------------- small helpers
def select_frames(first, last, max_frames=MAX_FRAMES):
    """plate indices to evaluate from the first to the last keyframe: all of them, or every s-th with the smallest s
    that gives at most ``max_frames``.  Returns (indices, s)."""
    s = 1
    while (last - first) // s + 1 > max_frames:
        s += 1
    return list(range(first, last + 1, s)), s


def pair_gaps(stride):
    """distances (plate frames) of the two drift measures: the evaluated frames closest to 1 and to 5 frames apart."""
    return stride, max(1, int(math.floor(5.0 / stride + 0.5))) * stride


def _workers(h, w):
    per_frame = h * w * 3 * 4 * 8  # about eight float32 pictures are alive for a frame under way
    return int(max(1, min(WORKERS, (os.cpu_count() or 2) // 2, WORK_RAM_BYTES // per_frame)))


def _f32(a):
    a = np.asarray(a)
    if a.ndim != 3 or a.shape[2] < 3:
        raise ValueError("an RGB picture is needed, got an array of shape %s" % (a.shape,))
    return np.ascontiguousarray(a[..., :3], dtype=np.float32)


def _floor0(a):
    """max(a, 0) + EPS"""
    out = np.maximum(a, 0)
    out += EPS
    return out


def _edit(p, s):
    """edit layer: log ratio of a picture over the plate frame."""
    return cv2.log(cv2.divide(_floor0(p), _floor0(s)))


def _look(e):
    return cv2.GaussianBlur(e, (0, 0), SIG)


def _texture(e):
    return cv2.subtract(cv2.GaussianBlur(e, (0, 0), TEX_SIGMAS[0]), cv2.GaussianBlur(e, (0, 0), TEX_SIGMAS[1]))


def _edited(a, b):
    """(uint8 mask 0/1 of the pixels where a channel of a differs from b by EDIT_MIN or more, |a - b|)"""
    d = cv2.absdiff(a, b)
    c = cv2.split(d)
    return (cv2.max(cv2.max(c[0], c[1]), c[2]) >= EDIT_MIN).view(np.uint8), d


def _share(mask):
    return cv2.countNonZero(mask) / float(mask.size)


def _mean_sq(band, mask=None):
    """mean square of a band over the whole frame or inside a mask (None when the mask is empty)."""
    if mask is None:
        return float(cv2.norm(band, cv2.NORM_L2SQR)) / band.size
    n = cv2.countNonZero(mask)
    return float(cv2.norm(band, cv2.NORM_L2SQR, mask)) / (n * band.shape[2]) if n else None


def _psnr(mse):
    return 10.0 * math.log10(1.0 / max(float(mse), 1e-12))


def _mean(vals):
    vals = [float(v) for v in vals if v is not None and math.isfinite(v)]
    return sum(vals) / len(vals) if vals else None


def _num(v, digits=4):
    """a number for the JSON line: None for anything that is not a finite number."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return round(x, digits) if math.isfinite(x) else None


def _check(cancelled):
    if cancelled is not None and cancelled():
        raise kfprop.KfCancelled()


# ---------------------------------------------------------------------- one pass over the frames
class _Pass:
    """One pass over frames in ascending order.  The caller hands in a frame (after inference) and goes on to the
    next; the arithmetic of the frame and its drift against the frames 1 and 5 evaluated frames back run in a few
    threads.  The figures come back in frame order, so the result does not depend on the threads.

    Drift = real_eval.py ``drift``: for a frame pair (t - d, t) the band of frame t is brought onto frame t - d with
    ``Shot.chain_map`` and compared inside the edited area of t - d - the sum of the squared difference divided by
    the band's own energy there."""

    def __init__(self, shot, measures, cancelled=None):
        self.shot = shot
        self.measures = dict(measures)  # name -> (band, how many evaluated frames back)
        self.backs = sorted({back for _, back in self.measures.values()})
        self.cancelled = cancelled
        n = _workers(shot.h, shot.w)
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=n, thread_name_prefix="wildcat-quality")
        self.slots = threading.Semaphore(n + 1)  # frames under way: bounds the memory
        self.lock = threading.Lock()
        self.frames = []
        self.futures = []
        self.sides = []

    def add(self, t, work):
        """``work()`` -> (figures of the frame, uint8 mask of its edited area, the mask's share of the picture, bands
        by name); it runs in one of the threads."""
        _check(self.cancelled)
        self.slots.acquire()
        me = {"t": t, "ready": threading.Event(), "kept": None, "uses": len(self.backs)}
        self.frames.append(me)
        try:
            self.futures.append(self.pool.submit(self._frame, len(self.frames) - 1, work))
        except BaseException:
            me["ready"].set()
            self.slots.release()
            raise

    def side(self, work):
        """a piece of work that is not a frame of the sequence (a keyframe the frame spacing skips)."""
        _check(self.cancelled)
        self.slots.acquire()
        try:
            self.sides.append(self.pool.submit(self._side, work))
        except BaseException:
            self.slots.release()
            raise

    def _side(self, work):
        try:
            return work()
        finally:
            self.slots.release()

    def _frame(self, i, work):
        me = self.frames[i]
        try:
            try:
                _check(self.cancelled)
                figures, mask, share, bands = work()
                if share >= ROI_SHARE_MIN:  # a smaller area is nothing a later frame can be compared with
                    me["kept"] = self._keep(me["t"], mask, bands)
            finally:
                me["ready"].set()  # a later frame may be waiting for this one, whatever happened
            drift, maps = {}, {}
            for back in self.backs:
                if i < back:
                    continue
                old = self.frames[i - back]
                old["ready"].wait()
                kept = old["kept"]
                for name, (band, b) in self.measures.items():
                    if b != back or kept is None or band not in kept["bands"] or bands.get(band) is None:
                        continue
                    if back not in maps:
                        maps[back] = self.shot.chain_map(kept["t"], me["t"], box=kept["box"])
                    moved = cv2.remap(bands[band], maps[back][0], maps[back][1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                    ref = kept["bands"][band]
                    den = float(cv2.norm(ref, cv2.NORM_L2SQR, kept["mask"]))
                    if den > 1e-9:
                        drift[name] = float(cv2.norm(ref, moved, cv2.NORM_L2SQR, kept["mask"])) / den
                with self.lock:
                    old["uses"] -= 1
                    if old["uses"] <= 0:
                        old["kept"] = None  # no later frame needs it any more
            return figures, drift
        finally:
            self.slots.release()

    @staticmethod
    def _keep(t, mask, bands):
        """what a later frame needs of this one: the window around the edited area (the whole picture when the area
        is scattered all over it - then the frame's own arrays are kept and nothing is copied)."""
        bands = {k: v for k, v in bands.items() if v is not None}
        x, y, w, h = cv2.boundingRect(mask)
        if 2 * w * h > mask.size:
            return {"t": t, "box": None, "mask": mask, "bands": bands}
        cut = lambda a: np.ascontiguousarray(a[y : y + h, x : x + w])
        return {"t": t, "box": (y, y + h, x, x + w), "mask": cut(mask), "bands": {k: cut(v) for k, v in bands.items()}}

    def results(self):
        """wait for everything: ([figures per frame, in order], {measure: [drift values, in frame order]}, [side results])"""
        try:
            figures, drift = [], {name: [] for name in self.measures}
            for fut in self.futures:
                fig, dr = fut.result()
                figures.append(fig)
                for name, v in dr.items():
                    drift[name].append(v)
            return figures, drift, [fut.result() for fut in self.sides]
        finally:
            self.close()

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)


# ---------------------------------------------------------------------- what is measured once per run
class Prepared:
    """Everything that does not depend on the weights: the shot, the evaluated frames, the keyframes' painted
    statistics and the measurement floor."""

    def __init__(self):
        self.shot = None
        self.keys = []  # plate indices of the keyframes
        self.key_src = {}  # the arrays the trainer trains on (not copied)
        self.key_tgt = {}
        self.frames = []  # plate indices that are evaluated
        self.stride = 1
        self.gap1 = 1
        self.gap5 = 5
        self.key_look = {}  # keyframe -> RMS of the painted overall look over the whole frame
        self.painted_detail = None  # RMS of the painted detail / texture inside the keyframes' edited area (mean over keyframes)
        self.painted_texture = None
        self.key_area = {}  # keyframe -> packed bits of its edited area
        self.floor = {"tex_drift1": None, "tex_drift5": None}
        self.seconds = 0.0
        self.shared = False  # the Shot was built by the keyframe-commonality step of the same run
        self._store = None
        self._plate = None
        self._row = {}

    @property
    def flow(self):
        return self.shot.flow_engine

    def numbers(self, idx):
        return [int(self.shot.plate_nums[t]) for t in idx]

    def source(self, t):
        """what the model is given for plate frame t: the trainer's own array on a keyframe, else the plate frame."""
        if t in self.key_src:
            return _f32(self.key_src[t])
        return np.asarray(self._plate[self._row[t]], dtype=np.float32)

    def area(self, k):
        return np.unpackbits(self.key_area[k], count=self.shot.h * self.shot.w).reshape(self.shot.h, self.shot.w)

    def expected_look(self, t):
        """the keyframes' painted look around frame t, interpolated linearly in time."""
        i = bisect.bisect_left(self.keys, t)
        hi = self.keys[min(i, len(self.keys) - 1)]
        lo = hi if hi == t or i == 0 else self.keys[i - 1]
        if lo == hi:
            return self.key_look[lo]
        return self.key_look[lo] + (self.key_look[hi] - self.key_look[lo]) * (t - lo) / float(hi - lo)

    def measures(self, look=True):
        n1, n5 = self.gap1 // self.stride, self.gap5 // self.stride
        out = {"tex_drift1": ("tex", n1), "tex_drift5": ("tex", n5)}
        if look:
            out["look_drift5"] = ("look", n5)
        return out

    def close(self):
        self._plate = None
        if self._store is not None:
            self._store.close()
            self._store = None


def _carried_areas(shot, prep, cancelled=None):
    """For every evaluated frame: the edited area of the keyframes around it, carried along the plate's motion (packed
    bits).  It does not depend on a model, so the measurement floor can be taken before the first evaluation."""
    wanted = set(prep.frames)
    out = {}
    lin = lambda z, mx, my: cv2.remap(z, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    for a, b in zip(prep.keys[:-1], prep.keys[1:]):
        for start, stop, step in ((a, b, 1), (b, a, -1)):
            area = prep.area(start).astype(np.float32)
            cx, cy = shot.xx, shot.yy  # for every pixel of the current frame: where it is in the keyframe
            for t in range(start + step, stop, step):
                _check(cancelled)
                px, py = shot.chain_map(t, t - step)  # one frame back towards the keyframe
                cx, cy = lin(cx, px, py), lin(cy, px, py)
                if t in wanted:
                    got = lin(area, cx, cy) > 0.5
                    out[t] = got if t not in out else (out[t] | got)
    packed = {t: np.packbits(m) for t, m in out.items()}
    for k in prep.keys:
        if k in wanted:
            packed[k] = prep.key_area[k]
    return packed


def _plate_texture(s, packed, h, w):
    """the plate's own texture band and the area it is followed in (a frame of the measurement-floor pass)."""
    mask = np.unpackbits(packed, count=h * w).reshape(h, w) if packed is not None else np.zeros((h, w), np.uint8)
    return {}, mask, _share(mask), {"tex": _texture(cv2.log(_floor0(s)))}


def prepare(shot, key_src, key_tgt, scratch_dir=None, cancelled=None, max_frames=MAX_FRAMES, ram_limit=RAM_LIMIT_BYTES):
    """Measure what does not depend on the weights.  ``shot``: a kfprop.Shot with matched keyframes and its flow;
    ``key_src`` / ``key_tgt``: plate index of a keyframe -> the source / target array the trainer trains on."""
    t0 = time.time()
    if not shot.has_flow:
        raise kfprop.KfError("no_flow", "the plate's optical flow is missing")
    prep = Prepared()
    prep.shot = shot
    prep.keys = list(shot.keys)
    prep.key_src = {k: key_src[k] for k in prep.keys}
    prep.key_tgt = {k: key_tgt[k] for k in prep.keys}
    prep.frames, prep.stride = select_frames(prep.keys[0], prep.keys[-1], max_frames)
    prep.gap1, prep.gap5 = pair_gaps(prep.stride)
    h, w = shot.h, shot.w

    # the keyframes as painted (as trained on): overall look, detail and texture energy, edited area
    detail, texture = [], []
    for k in prep.keys:
        _check(cancelled)
        s, g = _f32(prep.key_src[k]), _f32(prep.key_tgt[k])
        if s.shape[:2] != (h, w) or g.shape[:2] != (h, w):
            raise kfprop.KfError("size", "keyframe %s is not %dx%d" % (shot.plate_nums[k], w, h), name=str(shot.plate_nums[k]), w=int(s.shape[1]), h=int(s.shape[0]), w0=w, h0=h)
        area, _ = _edited(g, s)
        e = _edit(g, s)
        look = _look(e)
        prep.key_look[k] = math.sqrt(_mean_sq(look))
        if cv2.countNonZero(area):
            detail.append(math.sqrt(_mean_sq(cv2.subtract(e, look), area)))
            texture.append(math.sqrt(_mean_sq(_texture(e), area)))
        prep.key_area[k] = np.packbits(area)
    prep.painted_detail, prep.painted_texture = _mean(detail), _mean(texture)

    # the plate frames that are evaluated, kept as half floats (Shot.S: exactly what the plate files hold)
    rows = [t for t in prep.frames if t not in prep.key_src]
    prep._row = {t: i for i, t in enumerate(rows)}
    prep._store = kfprop._Store(scratch_dir, ram_limit)
    prep._plate = prep._store.alloc((max(1, len(rows)), h, w, 3), np.float16, "plate")

    # measurement floor: the plate's own texture band followed the same way, inside the keyframes' carried edited area
    areas = _carried_areas(shot, prep, cancelled)
    run = _Pass(shot, prep.measures(look=False), cancelled)
    try:
        for t in prep.frames:
            if t in prep._row:
                s = shot.S(t)
                prep._plate[prep._row[t]] = s
            else:
                s = prep.source(t)
            run.add(t, functools.partial(_plate_texture, s, areas.get(t), h, w))
        _, drift, _ = run.results()
    finally:
        run.close()
    prep.floor = {name: _mean(drift[name]) for name in ("tex_drift1", "tex_drift5")}
    shot.forget_frames()
    prep.seconds = time.time() - t0
    return prep


# ---------------------------------------------------------------------- one evaluation
def _fit(prep, t, p):
    """keyframe t: PSNR of the output against the target the trainer trains on - whole frame, edited area."""
    g = _f32(prep.key_tgt[t])
    whole = _psnr(float(cv2.norm(p, g, cv2.NORM_L2SQR)) / p.size)
    area = prep.area(t)
    n = cv2.countNonZero(area)
    if FIT_ROI_SHARE[0] <= n / float(area.size) <= FIT_ROI_SHARE[1]:
        return whole, _psnr(float(cv2.norm(p, g, cv2.NORM_L2SQR, area)) / (n * 3))
    return whole, whole  # no usable edited area on this keyframe: the recommender's fallback


def _figures(prep, t, s, p, kernel):
    """everything one evaluated frame contributes, except the drift (a frame of the evaluation pass)."""
    is_key = t in prep.key_src
    fig = {"t": t, "fit": _fit(prep, t, p) if is_key else None, "strength": None, "detail": None, "texture": None, "background": None}
    roi, diff = _edited(p, s)
    fig["share"] = share = _share(roi)
    e = _edit(p, s)
    look, tex = _look(e), _texture(e)
    if not is_key:
        expected = prep.expected_look(t)
        if expected > 1e-9:
            fig["strength"] = math.sqrt(_mean_sq(look)) / expected
        if share > ROI_SHARE_MIN:
            if prep.painted_detail:
                fig["detail"] = math.sqrt(_mean_sq(cv2.subtract(e, look), roi)) / prep.painted_detail
            if prep.painted_texture:
                fig["texture"] = math.sqrt(_mean_sq(tex, roi)) / prep.painted_texture
    outside = (cv2.dilate(roi, kernel) == 0).view(np.uint8)
    if cv2.countNonZero(outside):
        fig["background"] = sum(cv2.mean(diff, mask=outside)[:3]) / 3.0
    return fig, roi, share, {"look": look, "tex": tex}


def evaluate(prep, infer, cancelled=None):
    """One evaluation of a set of weights.  ``infer(source)``: float32 HxWx3 plate frame -> the model's output for it
    (HxWx3); it is called from this thread, frame after frame.  Returns the figures of a "point" line (without type /
    step / seconds)."""
    shot = prep.shot
    key_set, frame_set = set(prep.keys), set(prep.frames)
    kernel = np.ones((2 * BG_DILATE_PX + 1, 2 * BG_DILATE_PX + 1), np.uint8)
    run = _Pass(shot, prep.measures(), cancelled)
    try:
        for t in sorted(key_set | frame_set):
            _check(cancelled)
            s = prep.source(t)
            p = _f32(infer(s))
            if p.shape != s.shape:
                raise ValueError("the model output is %s, the frame is %s" % (p.shape, s.shape))
            if t in frame_set:
                run.add(t, functools.partial(_figures, prep, t, s, p, kernel))
            else:  # a keyframe between two evaluated frames: only its fit
                run.side(functools.partial(_fit, prep, t, p))
        figures, drift, sides = run.results()
    finally:
        run.close()

    fits = [f["fit"] for f in figures if f["fit"] is not None] + list(sides)
    between = [f for f in figures if f["t"] not in key_set]
    valid = {f["t"]: f["strength"] for f in between if f["strength"] is not None and math.isfinite(f["strength"])}
    weakest = min(valid, key=valid.get) if valid else None
    return {
        "kf_fit_db": _num(_mean(a for a, _ in fits), 3),
        "kf_fit_roi_db": _num(_mean(b for _, b in fits), 3),
        "strength_mean": _num(_mean(valid.values())),
        "strength_min": _num(valid[weakest]) if weakest is not None else None,
        "strength_min_frame": int(shot.plate_nums[weakest]) if weakest is not None else None,
        "detail": _num(_mean(f["detail"] for f in between)),
        "texture": _num(_mean(f["texture"] for f in between)),
        "look_drift5": _num(_mean(drift["look_drift5"])),
        "tex_drift1": _num(_mean(drift["tex_drift1"])),
        "tex_drift5": _num(_mean(drift["tex_drift5"])),
        "bg_change": _num(_mean(f["background"] for f in figures), 6),
        "roi_share": _num(_mean(f["share"] for f in figures)),  # share of the picture the model edits (where detail / texture / drift are taken)
        "per_frame": {"frame": prep.numbers([f["t"] for f in between]), "strength": [_num(f["strength"], 3) for f in between]},
    }


def header_of(prep, prepare_seconds=None):
    return {
        "type": "header",
        "available": True,
        "keyframes": prep.numbers(prep.keys),
        "frames": prep.numbers(prep.frames),
        "stride": int(prep.stride),
        "gap1": int(prep.gap1),
        "gap5": int(prep.gap5),
        "floor": {k: _num(v) for k, v in prep.floor.items()},
        "flow": prep.flow,
        "prepare_seconds": _num(prep.seconds if prepare_seconds is None else prepare_seconds, 1),
    }


def log_line(step, point, floor):
    f = lambda v, fmt="%.2f": (fmt % v) if v is not None else "n/a"
    return ("WildCat | quality @step %d: keyframe fit %s dB (edited area %s dB) | in-between strength mean %s, weakest %s at frame %s | detail %s | "
            "look drift %s | texture drift %s (floor %s) | %s s"
            % (step, f(point.get("kf_fit_db")), f(point.get("kf_fit_roi_db")), f(point.get("strength_mean")), f(point.get("strength_min")),
               point.get("strength_min_frame") if point.get("strength_min_frame") is not None else "n/a", f(point.get("detail")),
               f(point.get("look_drift5")), f(point.get("tex_drift1")), f((floor or {}).get("tex_drift1")), f(point.get("seconds"), "%.1f")))


# ---------------------------------------------------------------------- the monitor of one training run
class TrainingMonitor:
    """Prepares in a background thread, decides at checkpoint time whether an evaluation fits the time budget, and
    writes ``run/<task>/quality.jsonl`` line by line:

      {"type": "header", "available": true, "keyframes": [...], "frames": [...], "stride": 1, "gap1": 1, "gap5": 5,
       "floor": {"tex_drift1": .., "tex_drift5": ..}, "flow": "dis", "prepare_seconds": ..}
      or {"type": "header", "available": false, "reason": code, "detail": "..."}
         (codes: those of kfprop.KfError, "rgb_only", "disabled", "failed")
      {"type": "point", "step": .., "seconds": .., "kf_fit_db": .., "kf_fit_roi_db": .., "strength_mean": ..,
       "strength_min": .., "strength_min_frame": .., "detail": .., "texture": .., "look_drift5": .., "tex_drift1": ..,
       "tex_drift5": .., "bg_change": .., "roi_share": .., "per_frame": {"frame": [...], "strength": [...]}}

    Frame numbers are those of the plate's file names; a figure that could not be measured is null.  A resumed run
    appends (its own header line first); a fresh run starts the file again.  No method raises."""

    def __init__(self, run_dir, log=print, resume=False):
        self.run_dir = run_dir
        self.log = log
        self.path = os.path.join(run_dir, FILE_NAME)
        self.state = "new"  # new | planned | preparing | ready | unavailable | failed | closed
        self.prep = None
        self.header = None
        self.spent = 0.0  # seconds charged to the budget so far (evaluations of this process; see due / waited)
        self.last_seconds = None  # what the previous evaluation was charged: the estimate for the next one
        self.points = 0
        self._build = None
        self._thread = None
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._owned = []  # [(shot or None, scratch folder or None)] released at the end
        self._busy = 0  # evaluations booked and not finished
        self._steps = set()
        self._charged = {}  # step -> seconds its evaluation has been charged to the budget
        self._failures = 0
        if not resume:
            try:
                os.remove(self.path)
            except OSError:
                pass
        atexit.register(self.close)

    # ------------------------------------------------------------------ file
    def _write(self, obj):
        try:
            line = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            with self._lock:
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
                    fh.flush()
            return True
        except (OSError, ValueError) as exc:
            self.log("WildCat | quality monitor: %s not written (%s)" % (FILE_NAME, exc))
            return False

    def unavailable(self, reason, detail):
        self.state = "unavailable"
        self.header = {"type": "header", "available": False, "reason": str(reason), "detail": str(detail)[:400]}
        self._write(self.header)
        if reason == "disabled":
            self.log("WildCat | quality monitor: switched off (io.quality_monitor)")
        else:
            self.log("WildCat | quality monitor: not available for this run - %s (training is not affected)" % detail)
        self._release()

    # ------------------------------------------------------------------ preparation (background thread)
    def own(self, shot=None, folder=None):
        """a Shot / scratch folder the monitor closes / removes when the run ends."""
        self._owned.append((shot, folder))

    def plan(self, build):
        """``build(cancelled)`` returns a ``Prepared`` (or raises kfprop.KfError); it runs in a thread of its own once
        ``start()`` is called."""
        if self.state == "new":
            self._build = build
            self.state = "planned"

    def start(self):
        if self.state != "planned":
            return
        self.state = "preparing"
        self._thread = threading.Thread(target=self._prepare, name="wildcat-quality-prepare", daemon=True)
        self._thread.start()

    def _prepare(self):
        t0 = time.time()
        try:
            prep = self._build(self._cancel.is_set)
            self.prep = prep
            header = header_of(prep, time.time() - t0)
            _check(self._cancel.is_set)
            self.header = header
            self._write(header)
            self.state = "ready"
            show = lambda v: "%.2f" % v if v is not None else "n/a"
            self.log("WildCat | quality monitor: ready - %d frames from keyframe %s to keyframe %s (%d keyframes%s, %s flow%s), measurement floor of the "
                     "texture drift %s / %s over %d / %d frames, prepared in %.0f s"
                     % (len(prep.frames), header["keyframes"][0], header["keyframes"][-1], len(prep.keys), "" if prep.stride == 1 else ", every %d. frame" % prep.stride,
                        prep.flow.upper(), " taken over from kf_common" if prep.shared else "", show(prep.floor["tex_drift1"]), show(prep.floor["tex_drift5"]),
                        prep.gap1, prep.gap5, time.time() - t0))
        except kfprop.KfCancelled:
            self.state = "closed"
        except kfprop.KfError as exc:
            self.unavailable(exc.code, str(exc))
        except Exception as exc:  # whatever happens here must not reach the training
            self.unavailable("failed", "%s: %s" % (type(exc).__name__, exc))

    # ------------------------------------------------------------------ schedule (training thread)
    def due(self, step, total_steps, elapsed, final=False):
        """Called at checkpoint time: may this checkpoint be evaluated?  Yes when the evaluations so far plus this one
        (estimated from the previous one) stay within ``BUDGET`` of the elapsed training time; the first one is
        allowed after ``FIRST_AFTER`` of the total steps; the end of the run (``final``) is always evaluated, so the
        checkpoints of the last ``FIRST_AFTER`` of the steps before it are not.  The preparation (reading the plate,
        its flow) runs beside the training in a thread of its own and is not part of the budget; it is recorded in
        the header.  What the budget is charged with: the duration of an evaluation, or - when validation shares the
        training card and the training thread waits for it (``waited``) - the length of that wait.
        A True answer books the evaluation: ``run(step, ...)`` or ``skipped(step)`` has to follow."""
        try:
            with self._lock:
                if self.state not in ("preparing", "ready") or int(step) in self._steps or self._cancel.is_set():
                    return False
                if final:
                    ok = True
                elif self.state != "ready" or self._busy:
                    ok = False
                elif step > total_steps * (1.0 - FIRST_AFTER):
                    ok = False  # the last step is evaluated anyway: not a second time just before it
                elif self.last_seconds is None:
                    ok = step >= FIRST_AFTER * total_steps
                else:
                    ok = self.spent + self.last_seconds <= BUDGET * elapsed
                if ok:
                    self._busy += 1
                    self._steps.add(int(step))
                return ok
        except Exception:
            return False

    # ------------------------------------------------------------------ evaluation (validation worker)
    def run(self, step, infer):
        """Evaluate the weights behind ``infer`` (see ``evaluate``), write the point and the log line."""
        try:
            th = self._thread
            if th is not None and th.is_alive() and th is not threading.current_thread():
                th.join(FINAL_WAIT_SEC)  # only the evaluation at the end of a run can arrive this early
                if th.is_alive():
                    self.log("WildCat | quality monitor: the plate was not prepared in time, step %d is not measured" % step)
                    return None
            if self.state != "ready" or self._cancel.is_set():
                return None
            t0 = time.time()
            figures = evaluate(self.prep, infer, self._cancel.is_set)
            seconds = time.time() - t0
            point = {"type": "point", "step": int(step), "seconds": _num(seconds, 2)}
            point.update(figures)
            self._write(point)
            with self._lock:
                self.spent += seconds
                self.last_seconds = seconds
                self._charged[int(step)] = seconds
                self.points += 1
                self._failures = 0
            self.log(log_line(int(step), point, self.prep.floor))
            return point
        except kfprop.KfCancelled:
            return None
        except Exception as exc:
            self._failures += 1
            self.log("WildCat | quality monitor: the measurement at step %d failed (%s: %s) - training is not affected" % (step, type(exc).__name__, exc))
            if self._failures >= MAX_FAILURES and self.state == "ready":
                self.state = "failed"
                self.log("WildCat | quality monitor: switched off for this run after %d failed measurements" % self._failures)
            return None
        finally:
            with self._lock:
                self._busy = max(0, self._busy - 1)

    def skipped(self, step):
        """the evaluation booked for ``step`` will not take place (its validation job failed)."""
        with self._lock:
            self._busy = max(0, self._busy - 1)

    def waited(self, step, seconds):
        """The training thread waited ``seconds`` for the evaluation of ``step`` (validation shares the training card,
        see Trainer._preview).  That wait - a little longer than the evaluation itself: the weights are loaded and
        validated first - is what the run was held up by, so the budget is charged with it."""
        with self._lock:
            extra = float(seconds) - self._charged.pop(int(step), 0.0)
            if extra > 0:
                self.spent += extra
                if self.last_seconds is not None:
                    self.last_seconds = max(self.last_seconds, float(seconds))

    # ------------------------------------------------------------------ end of the run
    def cancel(self):
        """a stop was asked for: a preparation or an evaluation under way gives up at its next frame."""
        self._cancel.set()

    def _release(self):
        prep, self.prep = self.prep, None
        if prep is not None:
            prep.close()
        owned, self._owned = self._owned, []
        for shot, folder in owned:
            try:
                if shot is not None:
                    shot.close()
            except Exception:
                pass
            if folder:
                shutil.rmtree(folder, ignore_errors=True)

    def close(self):
        """release the plate, the flow and the scratch files (call it after the validation worker has finished)."""
        try:
            self._cancel.set()
            th = self._thread
            if th is not None and th.is_alive() and th is not threading.current_thread():
                th.join(120.0)
            if th is not None and th.is_alive():
                return  # still inside a long read: leave its buffers alone, the process is about to end
            if self.state in ("planned", "preparing", "ready"):
                self.state = "closed"
            self._release()
        except Exception:
            pass


class _NoMonitor:
    """stands in when the monitor could not even be set up."""

    state = "unavailable"

    def start(self):
        pass

    def due(self, *a, **k):
        return False

    def run(self, *a, **k):
        return None

    def skipped(self, *a, **k):
        pass

    def waited(self, *a, **k):
        pass

    def cancel(self):
        pass

    def close(self):
        pass


def wanted(io_cfg, c_in, c_out):
    """(is the monitor switched on - ``io.quality_monitor``, default true; is the model RGB -> RGB)"""
    v = (io_cfg or {}).get("quality_monitor", True)
    on = v.strip().lower() not in ("0", "false", "no", "off", "") if isinstance(v, str) else bool(v)
    return on, int(c_in) == 3 and int(c_out) == 3


def for_trainer(trainer, cfg, src_files, tgt_files, kept=None):
    """The monitor of a training run (wildcat/trainer.py); never raises.  ``kept``: [(shot, scratch folder)] when the
    keyframe-commonality step has already built the plate's Shot in this run - its flow is then used, not computed
    again.  The trainer calls ``start()`` once training runs, ``due()`` / ``run()`` at checkpoints, ``close()`` at
    the end."""
    from .log import log

    if kfprop is None:  # an edition without the keyframe tools: no monitor, no file, nothing on the task page
        return _NoMonitor()
    mon = None
    try:
        mon = TrainingMonitor(trainer.run_dir, log=log, resume=bool(trainer.tr.get("resume", False)))
        scratch = os.path.join(trainer.run_dir, SCRATCH)
        shutil.rmtree(scratch, ignore_errors=True)  # left behind by a run that was killed
        kept = list(kept or [])
        for shot, folder in kept:
            mon.own(shot, folder)
        on, rgb = wanted(trainer.io, trainer.c_in, trainer.c_out)
        if not on:
            mon.unavailable("disabled", "switched off in the task settings (io.quality_monitor)")
            return mon
        if not rgb:
            mon.unavailable("rgb_only", "the quality monitor needs an RGB to RGB model (this one is %d to %d channels)" % (trainer.c_in, trainer.c_out))
            return mon
        from . import kfcommon, recommender

        plate_files = recommender.resolve_eval_sequence_files(cfg)
        if len(plate_files) < 2:
            mon.unavailable("no_plate", "the dataset has no evaluation sequence (the plate)")
            return mon
        cv2.setNumThreads(kfprop.NUM_THREADS)  # here, not in the background thread: no OpenCV call is under way yet
        mon.own(None, scratch)
        colour = (trainer.source_colorspace, trainer.working_colorspace, trainer.ocio_config)
        src_full, tgt_full = trainer.src_full, trainer.tgt_full
        src_files, tgt_files = list(src_files), list(tgt_files)

        def build(cancelled):
            def tick(done, total):
                _check(cancelled)

            shot = kept[0][0] if kept else None
            if shot is None:
                shot = kfcommon._PlateShot(src_files, tgt_files, plate_files, colour[0], colour[1], colour[2], scratch, RAM_LIMIT_BYTES)
                mon.own(shot, None)
                shot.match_keyframes(tick)
                shot.compute_flows(tick, engine="dis")
            pair_of = {k: src_files.index(shot.key_source[k]) for k in shot.keys}
            prep = prepare(shot, {k: src_full[i] for k, i in pair_of.items()}, {k: tgt_full[i] for k, i in pair_of.items()}, scratch, cancelled)
            prep.shared = bool(kept)
            return prep

        mon.plan(build)
        return mon
    except Exception as exc:
        try:
            log("WildCat | quality monitor: not started (%s: %s) - training is not affected" % (type(exc).__name__, exc))
        except Exception:
            pass
        return mon if mon is not None else _NoMonitor()
