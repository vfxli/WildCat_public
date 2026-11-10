# -*- coding: utf-8 -*-
"""Log Telemetry Parser, Preview Cache, 3x3 Contact Sheet Indexer & Nuke Model Inventory."""
import datetime
import glob
import hashlib
import json
import os
import re
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

import cv2
import numpy as np

from wildcat import color

from . import db, storage

CACHE_DIR = os.path.join(db.STORAGE_ROOT, "cache")
JPG_CACHE_DIR = os.path.join(CACHE_DIR, "jpg")
os.makedirs(JPG_CACHE_DIR, exist_ok=True)

_LOCK = threading.Lock()
_POOL = ThreadPoolExecutor(max_workers=max(2, (os.cpu_count() or 4) // 2))
_LOGGED_IMG_ERRS = set()

_STEP_LINE = re.compile(r"^\[(?P<t>\d{2}:\d{2}:\d{2})\]\s+step\s+(?P<step>\d+)\s+(?P<rest>.*)$")
_KV = re.compile(r"([A-Za-z_]+)\s+(-?\d+\.?\d*(?:[eE][-+]?\d+)?)")
_VAL_LINE = re.compile(r"^\[(?P<t>[\d:]+)\]\s+val(?P<ema>-ema)?\([^)]*\):\s*(?P<rest>.*)$")
_VAL_F = re.compile(r"f(\d+)\s+(-?[\d.]+)(?:/(-?[\d.]+))?")
_STEP_PNG = re.compile(r"step_(\d+)\.png$")
_CS_SUP_PNG = re.compile(r"step_(\d+)(?:_s(\d+))?\.png$")
_NUKE_CS_PNG = re.compile(r"^(Training_\d{6}_\d{6})\.(\d+)\.png$")
_NUKE_MODEL_RE = re.compile(r"^(Training_\d{6}_\d{6})\.(\d+)\.(cat|pt)$")
_STEP_PT = re.compile(r"step_(\d+)\.pt$")


def parse_train_log(path: str, max_bytes: int = 400_000) -> dict:
    steps = []
    vals = []
    tail_lines = []
    last_step = 0
    if not path or not os.path.isfile(path):
        return {"steps": [], "vals": [], "tail": [], "latest": {}}

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            if max_bytes:
                f.seek(0, 2)
                sz = f.tell()
                if sz > max_bytes:
                    f.seek(sz - max_bytes)
                    f.readline()
                else:
                    f.seek(0)
            all_lines = f.readlines()
            tail_lines = [ln.rstrip("\n") for ln in all_lines[-120:]]
            for line in all_lines:
                s_line = line.rstrip("\n")
                m = _STEP_LINE.match(s_line)
                if m:
                    d = {"step": int(m.group("step")), "time": m.group("t")}
                    for k, v in _KV.findall(m.group("rest")):
                        try:
                            d[k] = float(v)
                        except ValueError:
                            pass
                    if "loss" in d and "total" not in d:
                        d["total"] = d["loss"]
                    last_step = d["step"]
                    steps.append(d)
                    continue
                mv = _VAL_LINE.match(s_line)
                if mv:
                    fvals = []
                    for _, l1, _ in _VAL_F.findall(mv.group("rest")):
                        try:
                            fv = float(l1)
                            if fv > -0.999:
                                fvals.append(fv)
                        except ValueError:
                            pass
                    if fvals:
                        vals.append(
                            {
                                "step": last_step,
                                "time": mv.group("t"),
                                "mse": sum(fvals) / len(fvals),
                            }
                        )
    except Exception:
        sys.stderr.write(f"[WildCat ERROR] parse_train_log({path}) failed:\n{traceback.format_exc()}\n")
        sys.stderr.flush()

    latest = steps[-1] if steps else {}
    return {
        "steps": steps,
        "vals": vals,
        "tail": tail_lines,
        "latest": latest,
    }


def _cache_key(src: str) -> str:
    try:
        st = os.stat(src)
        sig = f"{src}|{st.st_size}|{int(st.st_mtime)}"
    except OSError:
        sig = src
    return hashlib.sha1(sig.encode("utf-8")).hexdigest()


_CACHE_WRITES = [0]
_CACHE_MAX_BYTES = 2 * 1024**3


def _prune_jpg_cache(max_bytes: int = _CACHE_MAX_BYTES, keep_ratio: float = 0.75):
    """Preview thumbnails are keyed by a hash of their source path, so deleting a task leaves
    them orphaned. Keep the cache bounded: drop least-recently-used files beyond the limit."""
    try:
        ents = []
        total = 0
        for e in os.scandir(JPG_CACHE_DIR):
            if e.is_file():
                st = e.stat()
                ents.append((max(st.st_atime, st.st_mtime), st.st_size, e.path))
                total += st.st_size
        if total <= max_bytes:
            return
        ents.sort()
        target = int(max_bytes * keep_ratio)
        for _, size, path in ents:
            if total <= target:
                break
            try:
                os.remove(path)
                total -= size
            except OSError:
                pass
    except OSError:
        pass


def _log_img_err_once(src_abs: str):
    if len(_LOGGED_IMG_ERRS) > 200:
        _LOGGED_IMG_ERRS.clear()
    if src_abs not in _LOGGED_IMG_ERRS:
        _LOGGED_IMG_ERRS.add(src_abs)
        sys.stderr.write(f"[WildCat ERROR] Failed to render preview thumbnail for {src_abs}:\n{traceback.format_exc()}\n")
        sys.stderr.flush()


def to_jpg_url(src_abs: str, quality: int = 88) -> str | None:
    if not src_abs or not os.path.isfile(src_abs):
        return None
    fname = f"{_cache_key(src_abs)}_{quality}.jpg"
    dst = os.path.join(JPG_CACHE_DIR, fname)
    if not os.path.isfile(dst):
        _CACHE_WRITES[0] += 1
        if _CACHE_WRITES[0] % 200 == 0:
            _prune_jpg_cache()
        if src_abs.lower().endswith(".exr"):
            try:
                exr_rgb = color.read_exr_file(src_abs)
                disp_srgb = color.acescg_to_display_srgb(exr_rgb, src_space="ACEScg")
                img = (disp_srgb[..., ::-1] * 255.0).clip(0, 255).astype(np.uint8)
            except Exception:
                _log_img_err_once(src_abs)
                return None
        else:
            try:
                img = color.imread_unicode(src_abs, cv2.IMREAD_UNCHANGED)
            except Exception:
                _log_img_err_once(src_abs)
                return None
            if img is None:
                return None
            if img.dtype == np.uint16:
                img = (img.astype(np.float32) / 257.0).clip(0, 255).astype(np.uint8)
            elif img.dtype != np.uint8:
                img = (img.astype(np.float32) * 255.0).clip(0, 255).astype(np.uint8)
            if img.ndim == 3 and img.shape[2] == 4:
                img = img[:, :, :3]
        tmp = f"{dst[:-4]}.tmp{os.getpid()}_{threading.get_ident()}.jpg"
        if color.imwrite_unicode(tmp, img, [cv2.IMWRITE_JPEG_QUALITY, quality]):
            with _LOCK:
                try:
                    os.replace(tmp, dst)
                except OSError:
                    if os.path.exists(tmp):
                        try:
                            os.remove(tmp)
                        except OSError:
                            pass
        else:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            return None
    return "/f/cache/jpg/" + quote(fname)


def batch_to_jpg_urls(files_abs: list, quality: int = 88) -> list:
    if not files_abs:
        return []
    return list(_POOL.map(lambda p: to_jpg_url(p, quality), files_abs))


_CKPT_META_CACHE = {}


def inspect_ckpt_cached(abs_path: str) -> dict:
    from wildcat import models

    try:
        st = os.stat(abs_path)
        key = (abs_path, int(st.st_mtime), st.st_size)
    except OSError:
        key = (abs_path, 0, 0)
    if key not in _CKPT_META_CACHE:
        _CKPT_META_CACHE[key] = models.inspect_weight_source(abs_path)
    return _CKPT_META_CACHE[key]


def resolve_nuke_prefix(rdir_abs: str) -> str:
    """Return the authoritative Nuke CopyCat run prefix 'Training_YYMMDD_HHMMSS' for a run directory."""
    info_p = os.path.join(rdir_abs, "run_info.json")
    if os.path.isfile(info_p):
        try:
            with open(info_p, "r", encoding="utf-8") as fh:
                info = json.load(fh)
            pfx = str(info.get("nuke_prefix") or "").strip()
            if pfx.startswith("Training_"):
                return pfx
        except Exception:
            pass
    summaries = sorted(glob.glob(os.path.join(rdir_abs, "Training_*.summary")))
    if summaries:
        return os.path.basename(summaries[-1])[: -len(".summary")]
    if os.path.isdir(rdir_abs):
        for fn in sorted(os.listdir(rdir_abs)):
            if fn.endswith(".tmp") or ".tmp." in fn:
                continue
            m = _NUKE_MODEL_RE.match(fn) or _NUKE_CS_PNG.match(fn)
            if m:
                return m.group(1)
    ref_p = os.path.join(rdir_abs, "ckpt", "latest.pt")
    if not os.path.isfile(ref_p):
        ref_p = rdir_abs
    try:
        ts = os.path.getmtime(ref_p)
    except OSError:
        ts = datetime.datetime.now().timestamp()
    return "Training_" + datetime.datetime.fromtimestamp(ts).strftime("%y%m%d_%H%M%S")


def list_task_models(run_dir_rel: str, cfg: dict = None, parsed_log: dict = None) -> dict:
    """Scan all saved model checkpoints of a task and return Nuke-aligned metadata sorted newest-first."""
    rdir_abs = storage.abs_from_root(run_dir_rel)
    nuke_prefix = resolve_nuke_prefix(rdir_abs)
    if not os.path.isdir(rdir_abs):
        return {"nuke_prefix": nuke_prefix, "count": 0, "items": []}

    total_steps = int(((cfg or {}).get("train") or {}).get("total_steps") or 0)
    epochs = int(((cfg or {}).get("train") or {}).get("epochs") or 0)
    info_p = os.path.join(rdir_abs, "run_info.json")
    if os.path.isfile(info_p):
        try:
            with open(info_p, "r", encoding="utf-8") as fh:
                rinfo = json.load(fh)
            total_steps = int(rinfo.get("total_steps") or total_steps)
            epochs = int(rinfo.get("epochs") or epochs)
        except Exception:
            pass

    loss_by_step = {}
    val_by_step = {}
    if parsed_log:
        for s in parsed_log.get("steps") or []:
            if "step" in s and s.get("total") is not None:
                loss_by_step[int(s["step"])] = float(s["total"])
        for v in parsed_log.get("vals") or []:
            if "step" in v and v.get("mse") is not None:
                val_by_step[int(v["step"])] = float(v["mse"])

    step_records = {}  # step -> dict of discovered files

    # 1. Scan Nuke-named files in run root (Training_YYMMDD_HHMMSS.<step>.cat / .pt)
    for fn in os.listdir(rdir_abs):
        if fn.endswith(".tmp") or ".tmp." in fn:
            continue
        m = _NUKE_MODEL_RE.match(fn)
        if not m:
            continue
        pfx, st_str, ext = m.group(1), m.group(2), m.group(3)
        if ext != "pt":
            continue
        st = int(st_str)
        if st <= 0:
            continue
        fp = os.path.join(rdir_abs, fn)
        rec = step_records.setdefault(st, {"prefix": pfx})
        rec[f"{ext}_file"] = fn
        rec[f"{ext}_abs"] = fp

    # 2. Scan ckpt/step_XXXXXX.pt (and ckpt/latest.pt if paused between checkpoint intervals)
    ckpt_dir = os.path.join(rdir_abs, "ckpt")
    if os.path.isdir(ckpt_dir):
        for fn in os.listdir(ckpt_dir):
            if fn.endswith(".tmp"):
                continue
            m = _STEP_PT.search(fn)
            if m:
                st = int(m.group(1))
                if st > 0:
                    rec = step_records.setdefault(st, {"prefix": nuke_prefix})
                    rec["ckpt_abs"] = os.path.join(ckpt_dir, fn)
        lat_abs = os.path.join(ckpt_dir, "latest.pt")
        if os.path.isfile(lat_abs):
            try:
                sp = inspect_ckpt_cached(lat_abs)
                st_lat = int(sp.get("step") or 0)
                if st_lat > 0 and st_lat not in step_records:
                    step_records[st_lat] = {"prefix": nuke_prefix, "ckpt_abs": lat_abs}
            except Exception:
                pass

    if not step_records:
        return {"nuke_prefix": nuke_prefix, "count": 0, "items": []}

    max_step = max(step_records.keys())
    best_val_step = None
    best_val = None
    for st in step_records:
        vm = val_by_step.get(st)
        if vm is not None and (best_val is None or vm < best_val):
            best_val = vm
            best_val_step = st

    items = []
    for st in sorted(step_records.keys(), reverse=True):
        rec = step_records[st]
        pfx = rec.get("prefix") or nuke_prefix
        primary_ext = "pt"
        stat_target = rec.get("pt_abs") or rec.get("ckpt_abs")
        size_mb = 0.0
        mtime_str = ""
        if stat_target and os.path.isfile(stat_target):
            try:
                st_info = os.stat(stat_target)
                size_mb = round(st_info.st_size / (1024.0 * 1024.0), 2)
                mtime_str = datetime.datetime.fromtimestamp(st_info.st_mtime).strftime("%m-%d %H:%M:%S")
            except OSError:
                pass
        ep_est = int(round((st * epochs) / total_steps)) if (epochs > 0 and total_steps > 0) else 0
        primary_name = rec.get(f"{primary_ext}_file") or f"{pfx}.{st}.{primary_ext}"
        item = {
            "step": st,
            "epoch": ep_est,
            "filename": primary_name,
            "pt_filename": rec.get("pt_file") or f"{pfx}.{st}.pt",
            "has_pt": bool(rec.get("pt_abs")),
            "size_mb": size_mb,
            "created_time": mtime_str,
            "loss": loss_by_step.get(st),
            "val_mse": val_by_step.get(st),
            "is_latest": st == max_step,
            "is_best": (best_val_step is not None and st == best_val_step),
        }
        items.append(item)

    return {
        "nuke_prefix": nuke_prefix,
        "zip_filename": f"{nuke_prefix}.zip",
        "count": len(items),
        "items": items,
    }


def list_all_checkpoints(user_id: int = None, all_users: bool = True) -> list:
    """Return list of checkpoints across runs with their locked architecture metadata."""
    out = []
    if not os.path.isdir(storage.RUN_ROOT):
        return out
    allowed_nodes = None
    if user_id is not None and not all_users:
        rows = db.query_all("SELECT node_name FROM tasks WHERE user_id = ?", (user_id,))
        allowed_nodes = {r["node_name"] for r in rows}
    for d in sorted(os.listdir(storage.RUN_ROOT)):
        if d.startswith(".") or d == "verification_report":
            continue
        if allowed_nodes is not None and d not in allowed_nodes:
            continue
        rdir = os.path.join(storage.RUN_ROOT, d)
        if not os.path.isdir(rdir):
            continue
        lat = os.path.join(rdir, "ckpt", "latest.pt")
        if os.path.isfile(lat):
            sp = inspect_ckpt_cached(lat)
            out.append(
                {
                    "path": f"run/{d}/ckpt/latest.pt",
                    "label": f"{d} / latest.pt [{sp['model_size']} · {sp['in_channels']}ch→{sp['out_channels']}ch · Step {sp['step']}]",
                    "run_name": d,
                    **sp,
                }
            )
        ckpts = []
        for p in glob.glob(os.path.join(rdir, "ckpt", "step_*.pt")):
            m = _STEP_PT.search(p)
            if m:
                ckpts.append((int(m.group(1)), p))
        for st, p in sorted(ckpts):
            sp = inspect_ckpt_cached(p)
            out.append(
                {
                    "path": f"run/{d}/ckpt/step_{st:06d}.pt",
                    "label": f"{d} / step_{st:06d}.pt [{sp['model_size']} · {sp['in_channels']}ch→{sp['out_channels']}ch]",
                    "run_name": d,
                    **sp,
                }
            )
    return out


def get_contact_sheet_frames(run_dir_rel: str, cfg: dict = None) -> dict:
    """Return all 3x3 Nuke-style local crop contact sheets (OCIO display-managed first, raw fallback)."""
    rdir_abs = storage.abs_from_root(run_dir_rel)
    nuke_prefix = resolve_nuke_prefix(rdir_abs)
    crop_size = int(((cfg or {}).get("train") or {}).get("crop_size") or 256)
    total_steps = int(((cfg or {}).get("train") or {}).get("total_steps") or 0)
    multi_res = bool(((cfg or {}).get("train") or {}).get("multi_resolution", True))
    info_p = os.path.join(rdir_abs, "run_info.json")
    if os.path.isfile(info_p):
        try:
            with open(info_p, "r", encoding="utf-8") as fh:
                rinfo = json.load(fh)
            crop_size = int(rinfo.get("crop_size") or crop_size)
            total_steps = int(rinfo.get("total_steps") or total_steps)
        except Exception:
            pass

    def _infer_scale(st_val: int) -> int:
        if not multi_res or total_steps <= 0:
            return 1
        s1 = (total_steps + 2) // 3
        if st_val <= s1:
            return 4
        if st_val <= 2 * s1:
            return 2
        return 1

    by_step = {}  # step -> (abs_path, scale, nuke_png_name)
    if os.path.isdir(rdir_abs):
        for fn in os.listdir(rdir_abs):
            if fn.endswith(".tmp.png") or ".tmp" in fn:
                continue
            m = _NUKE_CS_PNG.match(fn)
            if m:
                st = int(m.group(2))
                by_step[st] = (os.path.join(rdir_abs, fn), _infer_scale(st), fn)

    cs_dir = os.path.join(rdir_abs, "contact_sup")
    if os.path.isdir(cs_dir):
        for fn in os.listdir(cs_dir):
            if fn.endswith(".tmp.png") or ".tmp" in fn:
                continue
            m = _CS_SUP_PNG.search(fn)
            if m:
                st = int(m.group(1))
                sc = int(m.group(2)) if m.group(2) else _infer_scale(st)
                nuke_fn = by_step[st][2] if st in by_step else f"{nuke_prefix}.{st}.png"
                by_step[st] = (os.path.join(cs_dir, fn), sc, nuke_fn)

    ordered = sorted(by_step.items(), key=lambda kv: kv[0])
    urls = batch_to_jpg_urls([v[0] for _, v in ordered], quality=92)
    frames = []
    for (st, (_, sc, nuke_fn)), u in zip(ordered, urls):
        if u:
            frames.append(
                {
                    "step": st,
                    "scale": sc,
                    "url": u,
                    "label": f"Step {st} · 阶段 1/{sc}",
                    "nuke_file": nuke_fn,
                }
            )
    return {
        "nuke_prefix": nuke_prefix,
        "crop_size": crop_size,
        "frames": frames,
    }


def get_viewer_frames(run_dir_rel: str, cfg: dict, mode: str = "preview", pair: str = None) -> dict:
    rdir_abs = storage.abs_from_root(run_dir_rel)
    pairs = []
    selected_pair = pair
    frames = []
    if mode not in ("preview", "source", "target"):
        mode = "preview"

    if mode == "preview":
        sup_dir = os.path.join(rdir_abs, "preview_sup")
        if os.path.isdir(sup_dir):
            entries = sorted(
                e for e in os.listdir(sup_dir) if os.path.isdir(os.path.join(sup_dir, e))
            )
            if entries:
                pairs = entries
                if not selected_pair or selected_pair not in pairs:
                    selected_pair = pairs[0]
                target_d = os.path.join(sup_dir, selected_pair)
            else:
                target_d = sup_dir
            step_items = []
            for p in glob.glob(os.path.join(target_d, "step_*.png")):
                if p.endswith(".tmp.png"):
                    continue
                m = _STEP_PNG.search(p)
                if m:
                    step_items.append((int(m.group(1)), p))
            step_items.sort(key=lambda x: x[0])
            urls = batch_to_jpg_urls([p for _, p in step_items])
            for (st, _), u in zip(step_items, urls):
                if u:
                    frames.append({"url": u, "label": f"Step {st}", "step": st})
    elif mode == "source":
        src_p = (cfg or {}).get("paths", {}).get("source", "")
        files = storage.expand_sequence_files(src_p)
        urls = batch_to_jpg_urls(files)
        for fp, u in zip(files, urls):
            if u:
                frames.append({"url": u, "label": os.path.basename(fp)})
    elif mode == "target":
        tgt_p = (cfg or {}).get("paths", {}).get("target", "")
        files = storage.expand_sequence_files(tgt_p)
        urls = batch_to_jpg_urls(files)
        for fp, u in zip(files, urls):
            if u:
                frames.append({"url": u, "label": os.path.basename(fp)})

    return {
        "mode": mode,
        "pairs": pairs,
        "selected_pair": selected_pair,
        "frames": frames,
    }
