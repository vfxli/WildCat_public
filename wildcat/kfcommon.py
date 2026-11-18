# -*- coding: utf-8 -*-
"""Keyframe commonality (optional advanced module ``kf_common``, off by default).

Hand-made keyframes of one shot never agree completely: on some of them a region of the edit is painted weaker or
stronger, sits a few pixels off, or has its texture redrawn.  A network that is asked to reproduce every keyframe
exactly has to learn those accidents as well; between the keyframes its result then swims from one version to the next.

This module looks for what the keyframes have IN COMMON and rewrites a keyframe only where the others agree with each
other and this one does not:

  * the edit of a keyframe is the log ratio  artist's version / plate frame;
  * for keyframe t the edits of the keyframes around it (``span`` on each side) are brought onto t along the plate's own
    motion (``kfprop.Shot.link``: chained frame-to-frame flow with a forward-backward confidence);
  * band by band (broad > ~10 px, medium 4-10 px, fine < 4 px) and pixel by pixel
        rho  how much the others have in common there (1 = their carried edits coincide, 0 = unrelated; an unbiased
             estimate, so unrelated textures give 0; also 0 where fewer than two of them can be followed),
        u    how typical this keyframe is: its distance to the others' consensus in units of their own spread,
        new  = own + rho * (1 - u) * (consensus - own).

So a local outlier is pulled to what the other keyframes agree on (it is not learned), a keyframe that agrees is left
exactly as painted, and where the keyframes have nothing in common nothing is changed either - there is nothing to pull
towards, and averaging unrelated versions would only blur.  One exception, the overall look (setting ``overall``, on by
default): the broad band holds no detail, so where the keyframes' overall strength / colour disagree it is pulled to
their consensus (see BROAD_CONSENSUS below).  Nothing is invented on frames that have no keyframe: the
training set stays the artist's keyframes.

Numpy + OpenCV only.  With the module off none of this code runs.
"""
import numpy as np
import cv2

from . import kfprop

EPS = kfprop.EPS
BAND_NAMES = ("broad", "medium", "fine")
BAND_EDGES = (10.0, 4.0)  # Gaussian sigmas (px) that separate broad | medium | fine
POOL_SIGMA = (16.0, 8.0, 4.0)  # window (sigma, px) the statistics of each band are pooled over
CONF_PX = 2.0  # forward-backward flow error (px) at which a carried edit is trusted ~37%
EDIT_MIN = 0.03  # |log ratio| above which a pixel counts as edited (about 3 %)
DEFAULT_TAU2 = 2.0  # squared distance (in units of the others' spread) at which a keyframe is "half an outlier"
DEFAULT_SPAN = 2  # keyframes looked at on each side
CHANGE_MIN = 0.05  # a pixel counts as "rewritten" in the report when its edit was changed by at least this much (log units, about 5 %)
# Safeguards against errors of the optical flow (fast motion).  Keyframes on the same side of the shot are reached
# through the same stretch of flow, so a flow error moves their edits TOGETHER: they agree with each other and the
# keyframe under test looks like the odd one.
ONE_SIDED_DELTA = 0.75  # px; where only one side can be followed, a difference that a misplacement of this size explains is no outlier
SIDE_BALANCED = True  # each side of the shot counts as one voice (and no majority re-weighting, which would side with the larger group)
# Overall look (the broad band only).  Artists' keyframes also differ in overall strength / colour, everywhere and not
# as local outliers: then the other keyframes do not agree with each other either and the rule above changes nothing,
# and the trained result drifts slowly from one keyframe's look to the next.  A band this broad carries no detail, so
# it can be averaged without blurring anything:
#   "adapt"  pull the overall look to the consensus of the keyframes around (itself included) as far as they do NOT
#            agree - and not where the disagreement comes from ONE voice: this keyframe and the others are taken
#            together and each one is left out in turn; if the rest then agree (commonality above BROAD_HI), a single
#            keyframe is off there (this one: the outlier rule deals with it; another one: it must not be averaged
#            into this keyframe).  Full strength where even the best such subset agrees less than BROAD_LO.
#   "mean"   always pull it there (costs 0.1-0.2 dB when the overall look was consistent already, and smears a
#            keyframe whose whole look is off over its neighbours; kept for comparison only)
#   "off"    outliers only
BROAD_CONSENSUS = "adapt"
BROAD_HI, BROAD_LO = 0.9, 0.7
SHEET_FULL = 0.25  # report picture: a change of the edit by this much (log units, about 25 %) is drawn at full brightness


def _g(z, sigma):
    return cv2.GaussianBlur(z, (0, 0), sigma)


def _split(e):
    lo = _g(e, BAND_EDGES[0])
    mid = _g(e, BAND_EDGES[1])
    return lo, mid - lo, e - mid


class ShotView:
    """What the algorithm needs from a shot.  This default view reads everything through ``kfprop.Shot``; the trainer
    uses a view that takes the keyframes from the arrays it has already loaded (``TrainView``)."""

    def __init__(self, shot):
        self.shot = shot
        self.keys = list(shot.keys)
        self.h, self.w = shot.h, shot.w

    def source(self, k):
        return self.shot.S(k)

    def target(self, k):
        return self.shot.T(k)

    def edit(self, k):
        return to_log(self.source(k), self.target(k))

    def link(self, t, k, conf_px):
        return self.shot.link(t, k, conf_px)


class TrainView(ShotView):
    """Keyframes exactly as the trainer holds them (same colour handling as training); the Shot only supplies the motion."""

    def __init__(self, shot, source_by_key, target_by_key):
        ShotView.__init__(self, shot)
        self._src = source_by_key
        self._tgt = target_by_key

    def source(self, k):
        return np.asarray(self._src[k], dtype=np.float32)[..., :3]

    def target(self, k):
        return np.asarray(self._tgt[k], dtype=np.float32)[..., :3]


def to_log(src, lab):
    return np.log((np.maximum(lab, 0) + EPS) / (np.maximum(src, 0) + EPS)).astype(np.float32)


def from_log(src, e):
    return ((np.maximum(src, 0) + EPS) * np.exp(e) - EPS + np.minimum(src, 0)).astype(np.float32)


def _commonality(voices, weights, idx, pool):
    """how much the voices ``idx`` have in common, per pixel (1 = they coincide, 0 = unrelated or fewer than two of them)"""
    w = np.stack([weights[j] for j in idx])[..., None]
    bb = np.stack([voices[j] for j in idx])
    s = w.sum(0) + 1e-6
    mm = (w * bb).sum(0) / s
    keff = (s[..., 0] ** 2) / ((w[..., 0] ** 2).sum(0) + 1e-6)
    r2 = _g((w * ((bb - mm[None]) ** 2).sum(axis=3, keepdims=True)).sum(0)[..., 0] / s[..., 0], pool)
    p2 = _g((w * (bb**2).sum(axis=3, keepdims=True)).sum(0)[..., 0] / s[..., 0], pool)
    return np.clip(1.0 - (r2 * keff / np.maximum(keff - 1.0, 0.25)) / (p2 + 1e-7), 0.0, 1.0) * np.clip(keff - 1.0, 0.0, 1.0)


def _one_keyframe(view, i, tau2, span, conf_px, edits, delta=None, balanced=None, broad=None):
    """-> (own edit as the sum of its bands, rewritten edit, per-pixel share replaced (max over bands), commonality per band)"""
    delta = ONE_SIDED_DELTA if delta is None else float(delta)
    balanced = SIDE_BALANCED if balanced is None else bool(balanced)
    broad = BROAD_CONSENSUS if broad is None else (broad if isinstance(broad, str) else ("mean" if broad else "off"))
    keys = view.keys
    t = keys[i]
    e_t = edits(t)
    own = _split(e_t)
    ops, ws = [], []
    others = [k for k in keys[max(0, i - span) : i + span + 1] if k != t]
    for k in others:
        mx, my, conf = view.link(t, k, conf_px)
        ops.append(_split(cv2.remap(edits(k), mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)))
        ws.append(conf)
    edited = np.abs(e_t).max(axis=2) > EDIT_MIN
    changed = np.zeros(e_t.shape[:2], np.float32)
    own_sum = own[0] + own[1] + own[2]
    if len(ops) < 2:  # one other keyframe cannot say which of the two is the odd one
        return own_sum, own_sum, changed, [0.0, 0.0, 0.0], edited
    new, common = [], []
    zero = np.zeros(e_t.shape[:2], np.float32)
    w_left = sum([c for k, c in zip(others, ws) if k < t], zero)
    w_right = sum([c for k, c in zip(others, ws) if k > t], zero)
    one_sided = 1.0 - np.clip(np.minimum(w_left, w_right), 0.0, 1.0)
    if balanced:
        W = np.stack([c / np.maximum(w_left if k < t else w_right, 1.0) for k, c in zip(others, ws)])[..., None]
    else:
        W = np.stack(ws)[..., None]
    sw = W.sum(0) + 1e-6
    keff = (sw[..., 0] ** 2) / ((W[..., 0] ** 2).sum(0) + 1e-6)  # effective number of opinions
    for b in range(3):
        B = np.stack([o[b] for o in ops])
        m = (W * B).sum(0) / sw
        for _ in range(0 if balanced else 2):  # robust consensus: an opinion far from it counts less
            r = ((B - m[None]) ** 2).sum(axis=3, keepdims=True)
            sc = np.median(r, axis=0, keepdims=True) + 1e-8
            Wr = W / (1.0 + r / (6.0 * sc))
            m = (Wr * B).sum(0) / (Wr.sum(0) + 1e-6)
        r2 = _g((W * ((B - m[None]) ** 2).sum(axis=3, keepdims=True)).sum(0)[..., 0] / sw[..., 0], POOL_SIGMA[b])  # spread of the others
        p2 = _g((W * (B**2).sum(axis=3, keepdims=True)).sum(0)[..., 0] / sw[..., 0], POOL_SIGMA[b])  # their own energy
        v = r2 * keff / np.maximum(keff - 1.0, 0.25)  # variance of one opinion (unbiased)
        rho = np.clip(1.0 - v / (p2 + 1e-7), 0.0, 1.0) * np.clip(keff - 1.0, 0.0, 1.0)  # commonality; none when < 2 opinions reach here
        d2 = _g(((own[b] - m) ** 2).sum(axis=2), POOL_SIGMA[b])
        floor = 0.02 * (p2 + _g((own[b] ** 2).sum(axis=2), POOL_SIGMA[b])) + 1e-7
        if delta > 0:  # what a misplacement of `delta` px would change, counted only where the support is one-sided
            gy, gx = np.gradient(own[b], axis=(0, 1))
            hy, hx = np.gradient(m, axis=(0, 1))
            floor = floor + one_sided * (delta**2) * 0.5 * _g(((gy**2) + (gx**2) + (hy**2) + (hx**2)).sum(axis=2), POOL_SIGMA[b])
        z2 = d2 / (v * (1.0 + 1.0 / np.maximum(keff, 1.0)) + floor)
        u = 1.0 / (1.0 + (z2 / tau2) ** 2)  # 1 = this keyframe is as typical as the others are among themselves
        a = (rho * (1.0 - u)).astype(np.float32)
        if b == 0 and broad != "off":  # overall look: the consensus INCLUDING this keyframe (weight 1 for itself, the others by confidence)
            pull = sw[..., 0] / (1.0 + sw[..., 0])
            if broad == "adapt":  # ... only as far as the keyframes' overall look differs in general (see BROAD_CONSENSUS)
                voices = [own[b]] + [o[b] for o in ops]
                weights = [np.ones(e_t.shape[:2], np.float32)] + [W[j, ..., 0] for j in range(len(ops))]
                best = _commonality(voices, weights, list(range(len(voices))), POOL_SIGMA[b])
                if len(voices) >= 3:
                    for leave in range(len(voices)):
                        best = np.maximum(best, _commonality(voices, weights, [j for j in range(len(voices)) if j != leave], POOL_SIGMA[b]))
                pull = pull * np.clip((BROAD_HI - best) / (BROAD_HI - BROAD_LO), 0.0, 1.0)
            a = np.maximum(a, pull.astype(np.float32))
        new.append(own[b] + a[..., None] * (m - own[b]))
        changed = np.maximum(changed, a)
        common.append(float(rho[edited].mean()) if edited.any() else 0.0)
    return own_sum, new[0] + new[1] + new[2], changed, common, edited


def common_targets(view, tau2=DEFAULT_TAU2, span=DEFAULT_SPAN, conf_px=CONF_PX, tick=None, keep_maps=False, delta=None, balanced=None, broad=None):
    """Rewrite the keyframes of ``view`` towards their commonality.

    Returns ``(targets, report)``: ``targets`` maps the plate index of every keyframe that was touched to its new
    target picture (float32, working space; pixels that were not touched are bit-identical to the artist's version);
    ``report`` holds per-keyframe figures and, with ``keep_maps``, the per-pixel share that was replaced."""
    tick = tick or (lambda done, total: None)
    keys = view.keys
    cache = {}

    def edits(k):  # a keyframe's edit is needed by itself and by its 2 * span neighbours: keep a short window
        if k not in cache:
            for old in [c for c in cache if abs(keys.index(c) - keys.index(k)) > 2 * span + 1]:
                del cache[old]
            cache[k] = view.edit(k)
        return cache[k]

    targets, rows, maps, deltas = {}, [], {}, {}
    for i, t in enumerate(keys):
        tick(i, len(keys))
        own_sum, new, changed, common, edited = _one_keyframe(view, i, float(tau2), int(span), float(conf_px), edits, delta, balanced, broad)
        n_ed = int(edited.sum())
        dmag = np.abs(new - own_sum).max(axis=2)  # how much the edit itself was changed (log units: 0.1 = about 10 %)
        row = {
            "key": int(t),
            "edited_px": n_ed,
            "replaced_mean": float(changed[edited].mean()) if n_ed else 0.0,  # mean share of the edit replaced inside the edited area
            "replaced_area": float((dmag[edited] >= CHANGE_MIN).mean()) if n_ed else 0.0,  # share of the edited area whose edit really changed
            "change_mean": float(dmag[edited].mean()) if n_ed else 0.0,  # mean change of the edit inside the edited area (log units)
            "commonality": {BAND_NAMES[b]: round(float(common[b]), 4) for b in range(3)},
        }
        if float(changed.max()) > 0.0:
            src = view.source(t)
            tgt = view.target(t)
            out = tgt + (from_log(src, new) - from_log(src, own_sum))  # exact zero difference wherever nothing was replaced
            row["max_change"] = float(np.abs(out - tgt).max())
            if row["max_change"] > 0.0:
                targets[t] = out.astype(np.float32)
        else:
            row["max_change"] = 0.0
        rows.append(row)
        if keep_maps:
            maps[t] = changed
            deltas[t] = dmag
    tick(len(keys), len(keys))
    n = max(len(rows), 1)
    report = {
        "tau2": float(tau2),
        "span": int(span),
        "conf_px": float(conf_px),
        "keyframes": rows,
        "touched": sorted(int(t) for t in targets),
        "replaced_mean": float(sum(r["replaced_mean"] for r in rows) / n),
        "replaced_area": float(sum(r["replaced_area"] for r in rows) / n),
        "change_mean": float(sum(r["change_mean"] for r in rows) / n),
        "commonality": {name: float(sum(r["commonality"][name] for r in rows) / n) for name in BAND_NAMES},
    }
    if keep_maps:
        report["maps"] = maps
        report["deltas"] = deltas
    return targets, report


# ---------------------------------------------------------------------- report picture
def write_sheet(view, targets, report, jpg_path, frame_label=None, max_rows=12, tile_width=560):
    """One JPG, a row per keyframe that was rewritten most: as painted | what is trained | how much the edit was changed
    (bright = the edit differs by ``SHEET_FULL`` or more; places where a share of a near-zero edit was replaced stay dark)."""
    deltas = report.get("deltas") or {}
    rows = sorted((r for r in report["keyframes"] if r["key"] in targets), key=lambda r: -r["replaced_area"])[:max_rows]
    if not rows:
        return False
    frame_label = frame_label or (lambda k: str(k))
    scale = min(1.0, float(tile_width) / view.w)

    def tile(img, text):
        if scale < 1.0:
            img = cv2.resize(img, (max(1, int(round(view.w * scale))), max(1, int(round(view.h * scale)))), interpolation=cv2.INTER_AREA)
        img = np.ascontiguousarray(img)
        cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        return img

    to8 = lambda a: (kfprop._srgb(a)[..., ::-1] * 255 + 0.5).astype(np.uint8)
    out = []
    for r in sorted(rows, key=lambda r: r["key"]):
        k = r["key"]
        heat = cv2.applyColorMap(np.clip(deltas.get(k, np.zeros((view.h, view.w), np.float32)) / SHEET_FULL * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        out.append(np.concatenate([
            tile(to8(view.target(k)), "keyframe %s as painted" % frame_label(k)),
            tile(to8(targets[k]), "trained on (commonality)"),
            tile(heat, "changed (rewritten: %.0f%% of the edited area)" % (100 * r["replaced_area"])),
        ], axis=1))
    return bool(cv2.imwrite(jpg_path, np.concatenate(out, axis=0), [cv2.IMWRITE_JPEG_QUALITY, 88]))


# ---------------------------------------------------------------------- trainer entry
class _PlateShot(kfprop.Shot):
    """The trainer only needs the plate's motion and which plate frame each keyframe is; an alpha channel in the
    files is irrelevant here (the model is RGB -> RGB), so the propagation feature's alpha check is not applied."""

    def _read_rgb(self, path, check_alpha=False):
        return kfprop.Shot._read_rgb(self, path, check_alpha=False)


def apply_to_training(kc, src_files, tgt_files, src_full, tgt_full, plate_files, run_dir, colour, log=print, keep_shot=None):
    """advanced.kf_common: rewrite the loaded Ground Truth keyframes (``tgt_full``, in place) towards their commonality.

    Never raises for a dataset the module cannot handle: it logs why, leaves the keyframes untouched and returns a
    report with ``applied = False`` (training then runs exactly as with the module off).

    ``keep_shot``: a list.  When the plate's Shot and its flow exist at the end, ``(shot, scratch folder)`` is appended
    instead of being closed / removed here, and the caller owns both (the quality monitor of the same run follows the
    plate with the same flow - it is not computed twice)."""
    import json
    import os
    import shutil
    import time

    t0 = time.time()
    want_flow = str(kc.get("flow") or "dis")
    rep = {"applied": False, "threshold": float(kc["threshold"]), "span": int(kc["span"]), "flow_requested": want_flow, "overall": bool(kc.get("overall", True))}
    tmp = os.path.join(run_dir, "_kf_common_tmp")
    shot = None
    usable = False  # the Shot may be handed to the caller: the step ended normally or with a reason of its own
    try:
        cv2.setNumThreads(kfprop.NUM_THREADS)
        work_cs = colour.get("working_colorspace", "ACEScg")
        raw_in_cs = colour.get("input_colorspace")
        in_cs = work_cs if raw_in_cs in (None, "", "auto") else raw_in_cs
        shot = _PlateShot(src_files, tgt_files, plate_files, in_cs, work_cs,
                          colour.get("ocio_config_path", ""), tmp)
        shot.match_keyframes()
        if len(shot.keys) < 3:
            raise kfprop.KfError("too_few_keyframes", "at least 3 keyframes are needed to tell which one is the odd one (found %d)" % len(shot.keys),
                                 count=len(shot.keys))
        shot.compute_flows(engine=want_flow)
        rep["flow"] = shot.flow_engine
        if shot.flow_fallback:  # RAFT was asked for and could not run: DIS was used
            rep["flow_fallback"] = str(shot.flow_fallback.get("code") or "failed")
            log("WildCat | kf_common: RAFT optical flow not available (%s), using DIS" % rep["flow_fallback"])
        pair_of = {k: src_files.index(shot.key_source[k]) for k in shot.keys}
        view = TrainView(shot, {k: src_full[i] for k, i in pair_of.items()}, {k: tgt_full[i] for k, i in pair_of.items()})
        targets, r = common_targets(view, tau2=kc["threshold"], span=kc["span"], keep_maps=True, broad=(None if kc.get("overall", True) else "off"))
        label = lambda k: str(shot.plate_nums[k])
        try:
            write_sheet(view, targets, r, os.path.join(run_dir, "kf_common.jpg"), frame_label=label)
        except Exception as exc:  # the picture is a convenience; training must not depend on it
            log("WildCat | kf_common: report picture not written (%s)" % exc)
        for k, new in targets.items():
            old = tgt_full[pair_of[k]]
            arr = np.array(old, dtype=np.float32, copy=True)
            arr[..., :3] = new
            tgt_full[pair_of[k]] = arr.astype(old.dtype) if old.dtype != np.float32 else arr
        r.pop("maps", None)
        r.pop("deltas", None)
        for row in r["keyframes"]:
            row["frame"] = int(shot.plate_nums[row["key"]])
            row["name"] = os.path.basename(shot.key_target[row["key"]])
        rep.update(r)
        rep.update(applied=True, keyframe_count=len(shot.keys), plate_frames=int(shot.n), seconds=round(time.time() - t0, 1),
                   touched=[int(shot.plate_nums[k]) for k in sorted(targets)])
        worst = max(r["keyframes"], key=lambda row: row["replaced_area"])
        log("WildCat | kf_common: %d keyframes compared along %d plate frames (%s flow) in %.0f s | common share broad %.0f%% medium %.0f%% fine %.0f%% | "
            "rewritten: %.1f%% of the edited area on average, most on frame %d (%.0f%%) | kf_common.jpg shows where"
            % (len(shot.keys), shot.n, shot.flow_engine.upper(), time.time() - t0, 100 * r["commonality"]["broad"], 100 * r["commonality"]["medium"],
               100 * r["commonality"]["fine"], 100 * r["replaced_area"], worst["frame"], 100 * worst["replaced_area"]))
        usable = True
    except kfprop.KfError as exc:
        rep.update(reason=exc.code, detail=str(exc))
        log("WildCat | kf_common: NOT applied, training uses the keyframes as they are - %s" % exc)
        usable = True
    finally:
        if usable and keep_shot is not None and shot is not None and shot.has_flow:
            keep_shot.append((shot, tmp))
        else:
            if shot is not None:
                shot.close()
            shutil.rmtree(tmp, ignore_errors=True)
    try:
        with open(os.path.join(run_dir, "kf_common.json"), "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=1)
    except OSError:
        pass
    return rep
