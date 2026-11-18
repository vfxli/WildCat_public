# -*- coding: utf-8 -*-
"""Log Telemetry Parser, Preview Cache, 3x3 Contact Sheet Indexer & Nuke Model Inventory."""
import datetime
import glob
import hashlib
import json
import math
import os
import re
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote, unquote

import cv2
import numpy as np

from wildcat import color

from . import db, storage

CACHE_DIR = storage.get_cache_dir()
JPG_CACHE_DIR = storage.get_jpg_cache_dir()

_LOCK = threading.Lock()
_POOL = ThreadPoolExecutor(max_workers=max(2, (os.cpu_count() or 4) // 2))
_LOGGED_IMG_ERRS = set()

_STEP_LINE = re.compile(r"^\[(?P<t>\d{2}:\d{2}:\d{2})\]\s+step\s+(?P<step>\d+)\s+(?P<rest>.*)$")
_KV = re.compile(r"([A-Za-z_]+)\s+(-?\d+\.?\d*(?:[eE][-+]?\d+)?)")
_VAL_LINE = re.compile(
    r"^\[(?P<t>[\d:]+)\]\s+(?:@step\s+(?P<vstep>\d+)\s+)?val(?P<ema>-ema)?\([^)]*\)\s*:\s*(?P<rest>.*)$"
)
_VAL_F = re.compile(r"f(\d+)\s+(-?[\d.]+)(?:/(-?[\d.]+))?")
_STEP_PNG = re.compile(r"step_(\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)
_CS_SUP_PNG = re.compile(r"step_(\d+)(?:_s(\d+))?\.(?:jpg|jpeg|png)$", re.IGNORECASE)
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
                        vs_raw = mv.group("vstep")
                        vstep = int(vs_raw) if vs_raw else last_step
                        vals.append(
                            {
                                "step": vstep,
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


_FULL_LOG_CACHE = {}


def _downsample_chart_steps(steps: list, keep_steps: set, max_points: int = 3000) -> list:
    n = len(steps)
    if n <= max_points or max_points < 64:
        return steps
    must = {0, n - 1}
    for i, s in enumerate(steps):
        st = s.get("step")
        if st in keep_steps:
            must.add(i)
        if i > 0 and s.get("scale") != steps[i - 1].get("scale"):
            must.add(i - 1)
            must.add(i)
    budget = max(32, (max_points - len(must)) // 2)
    stride = max(1, n / float(budget))
    for b in range(budget):
        lo = int(b * stride)
        hi = min(n, max(lo + 1, int((b + 1) * stride)))
        min_i, max_i = lo, lo
        min_v, max_v = None, None
        for idx in range(lo, hi):
            v = steps[idx].get("total")
            if v is None:
                continue
            if min_v is None or v < min_v:
                min_v, min_i = v, idx
            if max_v is None or v > max_v:
                max_v, max_i = v, idx
        must.add(min_i)
        must.add(max_i)
    return [steps[i] for i in sorted(must)]


def parse_train_log_full_cached(path: str, max_chart_points: int = 3000) -> dict:
    """Parse 100% of train.log with an mtime/size cache, preserving exact per-step loss & val_mse
    across 200,000+ step runs while envelope-downsampling chart points for smooth UI rendering."""
    if not path or not os.path.isfile(path):
        return {"steps": [], "vals": [], "tail": [], "latest": {}, "loss_by_step": {}, "val_by_step": {}}
    try:
        st = os.stat(path)
        sig = (st.st_mtime_ns, st.st_size)
    except OSError:
        sig = None

    with _LOCK:
        hit = _FULL_LOG_CACHE.get(path)
        if hit is not None and sig is not None and hit[0] == sig:
            c = hit[1]
            return {**c, "tail": list(c["tail"])}

    raw = parse_train_log(path, max_bytes=0)
    loss_by_step = {}
    for s in raw["steps"]:
        if "step" in s and s.get("total") is not None:
            loss_by_step[int(s["step"])] = float(s["total"])
    val_by_step = {}
    for v in raw["vals"]:
        if "step" in v and v.get("mse") is not None:
            val_by_step[int(v["step"])] = float(v["mse"])

    chart_steps = _downsample_chart_steps(raw["steps"], set(val_by_step.keys()), max_points=max_chart_points)
    entry = {
        "steps": chart_steps,
        "vals": raw["vals"],
        "tail": raw["tail"],
        "latest": raw["latest"],
        "loss_by_step": loss_by_step,
        "val_by_step": val_by_step,
    }
    if sig is not None:
        with _LOCK:
            if len(_FULL_LOG_CACHE) >= 32 and path not in _FULL_LOG_CACHE:
                _FULL_LOG_CACHE.pop(next(iter(_FULL_LOG_CACHE)), None)
            _FULL_LOG_CACHE[path] = (sig, entry)
    return {**entry, "tail": list(entry["tail"])}


def _cache_key(src: str) -> str:
    pfx = "aces1_v2|" if str(src).lower().endswith(".exr") else ""
    try:
        st = os.stat(src)
        sig = f"{pfx}{src}|{st.st_size}|{st.st_mtime_ns}"
    except OSError:
        sig = f"{pfx}{src}"
    return hashlib.sha1(sig.encode("utf-8")).hexdigest()


_CACHE_WRITES = [0]
_CACHE_MAX_BYTES = 2 * 1024**3


def _prune_jpg_cache(max_bytes: int = _CACHE_MAX_BYTES, keep_ratio: float = 0.75):
    """Preview thumbnails are keyed by a hash of their source path, so deleting a task leaves
    them orphaned. Keep the cache bounded: drop least-recently-used files beyond the limit."""
    try:
        jpg_dir = storage.get_jpg_cache_dir()
        ents = []
        total = 0
        for e in os.scandir(jpg_dir):
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


def ensure_jpg_file(src_abs: str, quality: int = 88) -> str | None:
    if not src_abs or not os.path.isfile(src_abs):
        return None
    fname = f"{_cache_key(src_abs)}_{quality}.jpg"
    jpg_dir = storage.get_jpg_cache_dir()
    dst = os.path.join(jpg_dir, fname)
    if not os.path.isfile(dst):
        fallback_dst = os.path.join(storage.DEFAULT_CACHE_DIR, "jpg", fname)
        if os.path.isfile(fallback_dst):
            return fallback_dst
        _CACHE_WRITES[0] += 1
        if _CACHE_WRITES[0] % 200 == 0:
            _prune_jpg_cache()
        if src_abs.lower().endswith(".exr"):
            try:
                exr_rgb = color.read_exr_file(src_abs)
                disp_srgb = color.acescg_to_display_srgb(exr_rgb, src_space="ACEScg")
                if exr_rgb.ndim == 3 and exr_rgb.shape[2] == 4:
                    alpha = np.clip(exr_rgb[..., 3:4], 0.0, 1.0)
                    if float(np.min(alpha)) < 0.995:
                        h, w = disp_srgb.shape[:2]
                        yy = (np.arange(h, dtype=np.int32) // 16)[:, None]
                        xx = (np.arange(w, dtype=np.int32) // 16)[None, :]
                        chk = np.where(((yy + xx) & 1) == 0, 0.11, 0.165).astype(np.float32)[..., None]
                        disp_srgb = np.clip(disp_srgb + (1.0 - alpha) * chk, 0.0, 1.0)
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
                alpha = img[:, :, 3:4].astype(np.float32) / 255.0
                if float(np.min(alpha)) < 0.995:
                    h, w = img.shape[:2]
                    yy = (np.arange(h, dtype=np.int32) // 16)[:, None]
                    xx = (np.arange(w, dtype=np.int32) // 16)[None, :]
                    chk = np.where(((yy + xx) & 1) == 0, 28.0, 42.0).astype(np.float32)[..., None]
                    img = np.clip(img[:, :, :3].astype(np.float32) * alpha + (1.0 - alpha) * chk, 0, 255).astype(np.uint8)
                else:
                    img = img[:, :, :3]
        os.makedirs(jpg_dir, exist_ok=True)
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
    return dst


def to_jpg_url(src_abs: str, quality: int = 88) -> str | None:
    dst = ensure_jpg_file(src_abs, quality=quality)
    if not dst:
        return None
    return "/f/cache/jpg/" + quote(os.path.basename(dst))


def _rel_under_run_roots(src_abs: str) -> str | None:
    ap = os.path.abspath(src_abs)
    for rroot in (storage.get_run_root(), storage.DEFAULT_RUN_ROOT):
        if not rroot:
            continue
        ra = os.path.abspath(rroot)
        if os.path.normcase(ap).startswith(os.path.normcase(ra + os.sep)):
            return ap[len(ra) + 1 :].replace("\\", "/")
    return None


def run_preview_url(src_abs: str, quality: int = 90) -> str | None:
    """Return an O(1) preview URL for files inside run/<task>/ without blocking /telemetry on
    hundreds of legacy 16-bit PNG conversions."""
    if not src_abs or not os.path.isfile(src_abs):
        return None
    low = src_abs.lower()
    rel_run = _rel_under_run_roots(src_abs)
    if low.endswith((".jpg", ".jpeg")) and rel_run:
        return "/f/run/" + quote(rel_run)
    fname = f"{_cache_key(src_abs)}_{quality}.jpg"
    jpg_dir = storage.get_jpg_cache_dir()
    if os.path.isfile(os.path.join(jpg_dir, fname)) or os.path.isfile(os.path.join(storage.DEFAULT_CACHE_DIR, "jpg", fname)):
        return "/f/cache/jpg/" + quote(fname)
    if rel_run and low.endswith(".png"):
        return "/f/run_lazy_jpg/" + quote(rel_run)
    return to_jpg_url(src_abs, quality=quality)


def batch_to_jpg_urls(files_abs: list, quality: int = 88) -> list:
    if not files_abs:
        return []
    return list(_POOL.map(lambda p: to_jpg_url(p, quality), files_abs))


_CKPT_META_CACHE = {}


def inspect_ckpt_cached(abs_path: str) -> dict:
    from wildcat import models

    try:
        st = os.stat(abs_path)
        key = (abs_path, st.st_mtime_ns, st.st_size)
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


def _resolve_cfg_eval_seq(cfg: dict) -> str:
    paths_cfg = (cfg or {}).get("paths") or {}
    seq_p = str(paths_cfg.get("eval_sequence") or "").strip()
    if seq_p and len(storage.expand_sequence_files(seq_p)) >= 2:
        return seq_p
    exp_ds_id = ((cfg or {}).get("experiment") or {}).get("dataset_id")
    if exp_ds_id:
        try:
            row = db.query_one("SELECT seq_path FROM assets WHERE id = ?", (int(exp_ds_id),))
            if row and row.get("seq_path") and len(storage.expand_sequence_files(row["seq_path"])) >= 2:
                return str(row["seq_path"]).strip()
        except Exception:
            pass
    src_p = str(paths_cfg.get("source") or "").strip()
    if src_p:
        try:
            row = db.query_one("SELECT seq_path FROM assets WHERE source_path = ? AND seq_path != '' LIMIT 1", (src_p,))
            if row and row.get("seq_path") and len(storage.expand_sequence_files(row["seq_path"])) >= 2:
                return str(row["seq_path"]).strip()
        except Exception:
            pass
    return ""


def list_task_models(run_dir_rel: str, cfg: dict = None, parsed_log: dict = None) -> dict:
    """Scan all saved model checkpoints of a task and return Nuke-aligned metadata sorted newest-first."""
    rdir_abs = storage.abs_from_root(run_dir_rel)
    nuke_prefix = resolve_nuke_prefix(rdir_abs)
    has_eval_seq = bool(_resolve_cfg_eval_seq(cfg))
    if not os.path.isdir(rdir_abs):
        return {
            "nuke_prefix": nuke_prefix,
            "count": 0,
            "items": [],
            "has_eval_sequence": has_eval_seq,
        }

    total_steps = int(((cfg or {}).get("train") or {}).get("total_steps") or 0)
    epochs = int(((cfg or {}).get("train") or {}).get("epochs") or 0)
    rinfo = {}
    info_p = os.path.join(rdir_abs, "run_info.json")
    if os.path.isfile(info_p):
        try:
            with open(info_p, "r", encoding="utf-8") as fh:
                rinfo = json.load(fh) or {}
            total_steps = int(rinfo.get("total_steps") or total_steps)
            epochs = int(rinfo.get("epochs") or epochs)
        except Exception:
            rinfo = {}

    if parsed_log is None:
        parsed_log = parse_train_log_full_cached(os.path.join(rdir_abs, "train.log"))
    loss_by_step = dict(parsed_log.get("loss_by_step") or {})
    val_by_step = dict(parsed_log.get("val_by_step") or {})
    if not loss_by_step:
        for s in parsed_log.get("steps") or []:
            if "step" in s and s.get("total") is not None:
                loss_by_step[int(s["step"])] = float(s["total"])
    if not val_by_step:
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
        return {
            "nuke_prefix": nuke_prefix,
            "count": 0,
            "items": [],
            "has_eval_sequence": has_eval_seq,
        }

    max_step = max(step_records.keys())

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
        }
        items.append(item)

    return {
        "nuke_prefix": nuke_prefix,
        "zip_filename": f"{nuke_prefix}.zip",
        "count": len(items),
        "items": items,
        "has_eval_sequence": has_eval_seq,
    }


def list_all_checkpoints(user_id: int = None, all_users: bool = True) -> list:
    """Return list of checkpoints across runs with their locked architecture metadata."""
    out = []
    run_roots = []
    for cand_r in (storage.get_run_root(), storage.DEFAULT_RUN_ROOT):
        if os.path.isdir(cand_r) and os.path.abspath(cand_r) not in [os.path.abspath(x) for x in run_roots]:
            run_roots.append(cand_r)
    if not run_roots:
        return out
    allowed_nodes = None
    if user_id is not None and not all_users:
        rows = db.query_all("SELECT node_name FROM tasks WHERE user_id = ?", (user_id,))
        allowed_nodes = {r["node_name"] for r in rows}
    seen_nodes = set()
    for rroot in run_roots:
        for d in sorted(os.listdir(rroot)):
            if d in seen_nodes or d.startswith(".") or d == "verification_report":
                continue
            if allowed_nodes is not None and d not in allowed_nodes:
                continue
            rdir = os.path.join(rroot, d)
            if not os.path.isdir(rdir):
                continue
            seen_nodes.add(d)
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
                fn = os.path.basename(p)
                out.append(
                    {
                        "path": f"run/{d}/ckpt/{fn}",
                        "label": f"{d} / {fn} [{sp['model_size']} · {sp['in_channels']}ch→{sp['out_channels']}ch]",
                        "run_name": d,
                        **sp,
                    }
                )
    return out


# Frame lists of the task page (viewer, contact sheets): listing the folders and mapping every picture to its cached JPG
# took ~0.1 s each, on every poll of every open page, also for finished runs whose folders never change. A list is reused
# while the folders it was read from keep their modification time (adding or renaming a file changes it) and every JPG
# it points to is still in the cache (the cache is pruned by size).
_FRAMES_CACHE: dict = {}
_FRAMES_CACHE_MAX = 512
_FRAMES_LOCK = threading.Lock()


def _mtimes(paths) -> tuple:
    out = []
    for p in paths:
        try:
            out.append(os.stat(p).st_mtime_ns)
        except OSError:
            out.append(None)
    return tuple(out)


def _dir_content_sig(d: str) -> tuple:
    try:
        d_mt = os.stat(d).st_mtime_ns
        cnt = 0
        latest_mt = 0
        for e in os.scandir(d):
            if e.name.endswith(".tmp.png") or ".tmp" in e.name:
                continue
            if e.is_file():
                cnt += 1
                mt = e.stat().st_mtime_ns
                if mt > latest_mt:
                    latest_mt = mt
        return (d_mt, cnt, latest_mt)
    except OSError:
        return (None, 0, 0)


def _jpg_urls_present(frames: list) -> bool:
    jpg_dir = storage.get_jpg_cache_dir()
    fallback = os.path.join(storage.DEFAULT_CACHE_DIR, "jpg")
    for fr in frames:
        u = str(fr.get("url") or "")
        if not u.startswith("/f/cache/jpg/"):
            continue
        name = unquote(u[len("/f/cache/jpg/"):])
        if not (os.path.isfile(os.path.join(jpg_dir, name)) or os.path.isfile(os.path.join(fallback, name))):
            return False
    return True


def _frames_cached(key, sig, build):
    with _FRAMES_LOCK:
        hit = _FRAMES_CACHE.get(key)
    if hit is not None and hit[0] == sig and _jpg_urls_present(hit[1].get("frames") or []):
        return {**hit[1], "frames": list(hit[1].get("frames") or [])}
    res = build()
    with _FRAMES_LOCK:
        while len(_FRAMES_CACHE) >= _FRAMES_CACHE_MAX:
            _FRAMES_CACHE.pop(next(iter(_FRAMES_CACHE)))
        _FRAMES_CACHE[key] = (sig, res)
    return {**res, "frames": list(res.get("frames") or [])}


def get_contact_sheet_frames(run_dir_rel: str, cfg: dict = None) -> dict:
    rdir_abs = storage.abs_from_root(run_dir_rel)
    tr = (cfg or {}).get("train") or {}
    ss = ((cfg or {}).get("advanced") or {}).get("stage_schedule")
    key = ("cs", rdir_abs, json.dumps([tr.get("crop_size"), tr.get("total_steps"), tr.get("multi_resolution", True), ss], sort_keys=True, default=str))
    sig = (_mtimes([rdir_abs, os.path.join(rdir_abs, "run_info.json")]), _dir_content_sig(os.path.join(rdir_abs, "contact_sup")))
    return _frames_cached(key, sig, lambda: _build_contact_sheet_frames(run_dir_rel, cfg))


def _build_contact_sheet_frames(run_dir_rel: str, cfg: dict = None) -> dict:
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

    from wildcat import advanced as _adv

    adv_cfg = _adv.parse_advanced_config(cfg or {})
    stage_bounds = (
        _adv.compute_stage_list(total_steps, multi_res, adv_cfg["stage_schedule"])
        if total_steps > 0
        else [(1, 0)]
    )

    def _infer_scale(st_val: int) -> int:
        if not multi_res or total_steps <= 0:
            return 1
        for sc, end_st in stage_bounds:
            if st_val <= end_st:
                return int(sc)
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
        for fn in sorted(os.listdir(cs_dir)):
            if fn.endswith(".tmp.png") or fn.endswith(".tmp.jpg") or ".tmp" in fn:
                continue
            m = _CS_SUP_PNG.search(fn)
            if m:
                st = int(m.group(1))
                sc = int(m.group(2)) if m.group(2) else _infer_scale(st)
                nuke_fn = by_step[st][2] if st in by_step else f"{nuke_prefix}.{st}.png"
                if st in by_step and by_step[st][0].lower().endswith((".jpg", ".jpeg")) and fn.lower().endswith(".png"):
                    continue
                by_step[st] = (os.path.join(cs_dir, fn), sc, nuke_fn)

    ordered = sorted(by_step.items(), key=lambda kv: kv[0])
    urls = [run_preview_url(v[0], quality=92) for _, v in ordered]
    frames = []
    for (st, (_, sc, nuke_fn)), u in zip(ordered, urls):
        if u:
            frames.append(
                {
                    "step": st,
                    "scale": sc,
                    "url": u,
                    "label": f"Step {st} · 1/{sc}",
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
    if mode == "source" or mode == "target":
        pat = str(((cfg or {}).get("paths") or {}).get(mode) or "")
        sig = (pat, _dir_content_sig(os.path.dirname(storage.abs_from_root(pat))) if pat else None)
    else:
        sup_dir = os.path.join(rdir_abs, "preview_sup")
        try:
            subs = sorted(e for e in os.listdir(sup_dir) if os.path.isdir(os.path.join(sup_dir, e)))
        except OSError:
            subs = []
        sig = (_dir_content_sig(sup_dir), tuple(subs), tuple(_dir_content_sig(os.path.join(sup_dir, e)) for e in subs))
    return _frames_cached(("vw", rdir_abs, mode, pair or ""), sig, lambda: _build_viewer_frames(run_dir_rel, cfg, mode, pair))


def _build_viewer_frames(run_dir_rel: str, cfg: dict, mode: str = "preview", pair: str = None) -> dict:
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
            step_map = {}
            if os.path.isdir(target_d):
                for fn in sorted(os.listdir(target_d)):
                    if ".tmp" in fn:
                        continue
                    m = _STEP_PNG.search(fn)
                    if m:
                        st = int(m.group(1))
                        if st in step_map and step_map[st].lower().endswith((".jpg", ".jpeg")) and fn.lower().endswith(".png"):
                            continue
                        step_map[st] = os.path.join(target_d, fn)
            step_items = sorted(step_map.items(), key=lambda x: x[0])
            urls = [run_preview_url(p, quality=90) for _, p in step_items]
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


# Files the optional keyframe-commonality step (advanced.kf_common, wildcat/kfcommon.py) leaves in the run folder.
# Both are served by /f/run/<task>/... (backend/server.py), i.e. to the task's owner and to admin-role accounts only.
KF_COMMON_JSON = "kf_common.json"
KF_COMMON_JPG = "kf_common.jpg"


def _json_num(v, digits: int = 4):
    """A number the API can return: None for anything that is not a finite number (a report may hold NaN / inf)."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return round(x, digits) if math.isfinite(x) else None


def _json_int(v):
    x = _json_num(v, 0)
    return int(x) if x is not None else None


# What the quality monitor of a run measured (wildcat/quality.py): a header line, then one line per evaluation.
QUALITY_JSONL = "quality.jsonl"
_QUALITY_FIGURES = (  # (field of a point, decimals)
    ("seconds", 2), ("kf_fit_db", 3), ("kf_fit_roi_db", 3), ("strength_mean", 4), ("strength_min", 4), ("detail", 4), ("texture", 4),
    ("look_drift5", 4), ("tex_drift1", 4), ("tex_drift5", 4), ("bg_change", 6), ("roi_share", 4),
)
_QUALITY_MAX_POINTS = 2000
_QUALITY_CACHE = {}


def _quality_header(h: dict) -> dict:
    ints = lambda v: [x for x in (_json_int(i) for i in (v if isinstance(v, list) else [])) if x is not None]
    if h.get("available") is not True:
        return {"type": "header", "available": False, "reason": str(h.get("reason") or "")[:40], "detail": str(h.get("detail") or "")[:400]}
    floor = h.get("floor") if isinstance(h.get("floor"), dict) else {}
    return {
        "type": "header",
        "available": True,
        "keyframes": ints(h.get("keyframes")),
        "frames": ints(h.get("frames")),
        "stride": _json_int(h.get("stride")),
        "gap1": _json_int(h.get("gap1")),
        "gap5": _json_int(h.get("gap5")),
        "floor": {"tex_drift1": _json_num(floor.get("tex_drift1")), "tex_drift5": _json_num(floor.get("tex_drift5"))},
        "flow": str(h.get("flow") or "")[:16],
        "prepare_seconds": _json_num(h.get("prepare_seconds"), 1),
    }


def _quality_point(p: dict, per_frame: bool) -> dict:
    out = {"type": "point", "step": _json_int(p.get("step"))}
    for key, digits in _QUALITY_FIGURES:
        out[key] = _json_num(p.get(key), digits)
        if key == "strength_min":
            out["strength_min_frame"] = _json_int(p.get("strength_min_frame"))
    if per_frame:
        pf = p.get("per_frame") if isinstance(p.get("per_frame"), dict) else {}
        frames = pf.get("frame") if isinstance(pf.get("frame"), list) else []
        vals = pf.get("strength") if isinstance(pf.get("strength"), list) else []
        rows = [(f, _json_num(v, 3)) for f, v in ((_json_int(f), v) for f, v in zip(frames, vals)) if f is not None]
        out["per_frame"] = {"frame": [f for f, _ in rows], "strength": [v for _, v in rows]}
    return out


def get_quality_report(run_dir_rel: str) -> dict | None:
    """The quality monitor of a run: None when run/<task>/quality.jsonl does not exist (or holds nothing yet), else

        {"available": bool, "reason": "", "detail": "", "header": {...}, "points": [every evaluation, without the
         per-frame list], "latest": the last evaluation with its per-frame list (None before the first one)}

    ``available`` / ``reason`` / ``detail`` are those of the header (the newest one: a resumed run writes its own).
    Reason codes: those of wildcat.kfprop.KfError, "rgb_only", "disabled", "failed".  Points are ordered by step; a
    figure that could not be measured is null (the file may hold NaN: it never reaches the page as such)."""
    path = os.path.join(storage.abs_from_root(run_dir_rel), QUALITY_JSONL)
    try:
        st = os.stat(path)
    except OSError:
        return None
    stamp = (st.st_mtime_ns, st.st_size)
    hit = _QUALITY_CACHE.get(path)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    header, points = None, {}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue  # an empty line, or the line that is being written right now
                if not isinstance(obj, dict):
                    continue
                if obj.get("type") == "header":
                    header = obj
                elif obj.get("type") == "point" and _json_int(obj.get("step")) is not None:
                    points[_json_int(obj.get("step"))] = obj  # a step measured again (after a resume) replaces the older line
    except OSError:
        return None
    if header is None and not points:
        return None
    head = _quality_header(header or {})
    steps = sorted(points)[-_QUALITY_MAX_POINTS:]
    out = {
        "available": head["available"],
        "reason": head.get("reason", ""),
        "detail": head.get("detail", ""),
        "header": head,
        "points": [_quality_point(points[s], per_frame=False) for s in steps],
        "latest": _quality_point(points[steps[-1]], per_frame=True) if steps else None,
    }
    if len(_QUALITY_CACHE) > 256:  # (the page polls: the file is parsed again only when it has grown)
        _QUALITY_CACHE.clear()
    _QUALITY_CACHE[path] = (stamp, out)
    return out


def get_kf_common_report(run_dir_rel: str) -> dict | None:
    """What the keyframe-commonality step did for a run: the summary figures of run/<task>/kf_common.json and the
    addresses of the comparison picture and of the full report. None when the run folder has no such report.

    The step never fails a training: when it could not run, the report says so (``applied`` false, ``reason`` = the
    code of wildcat.kfprop.KfError, ``detail`` = the line the trainer logged)."""
    rdir_abs = storage.abs_from_root(run_dir_rel)
    json_abs = os.path.join(rdir_abs, KF_COMMON_JSON)
    if not os.path.isfile(json_abs):
        return None
    try:
        with open(json_abs, "r", encoding="utf-8") as fh:
            rep = json.load(fh)
        json_mtime = int(os.path.getmtime(json_abs))
    except (OSError, ValueError):
        return None
    if not isinstance(rep, dict):
        return None

    applied = bool(rep.get("applied"))
    com = rep.get("commonality") if isinstance(rep.get("commonality"), dict) else {}
    rows = [r for r in (rep.get("keyframes") or []) if isinstance(r, dict)] if isinstance(rep.get("keyframes"), list) else []
    touched = rep.get("touched") if isinstance(rep.get("touched"), list) else []
    # "replaced_area" of a keyframe = the part of its edited area whose edit really changed.  "touched" lists every
    # keyframe whose pixels changed at all, however slightly (with consistent keyframes: all of them, by amounts far
    # below anything visible), so the page counts the keyframes that were NOTICEABLY rewritten instead.
    area_of = lambda r: _json_num(r.get("replaced_area")) or 0.0
    rewritten = [r for r in rows if area_of(r) > 0.0] if applied and touched else []
    # The keyframe that was rewritten most (the same one the trainer names in its log line); none when no keyframe
    # was noticeably rewritten.
    worst = None
    if rewritten:
        top = max(rewritten, key=area_of)
        worst = {
            "frame": _json_int(top.get("frame")),
            "name": str(top.get("name") or ""),
            "replaced_area": _json_num(top.get("replaced_area")),
        }

    base_url = "/f/run/" + quote(os.path.basename(os.path.normpath(rdir_abs)))
    image_url = None
    jpg_abs = os.path.join(rdir_abs, KF_COMMON_JPG)
    # the picture has a row per rewritten keyframe: none is written (and an older one is not shown) when nothing was rewritten
    if applied and touched and os.path.isfile(jpg_abs):
        try:
            image_url = "%s/%s?v=%d" % (base_url, KF_COMMON_JPG, int(os.path.getmtime(jpg_abs)))
        except OSError:
            image_url = None

    # optical flow the keyframes were compared with: "dis" | "raft". A report from before the flow choice existed has no
    # such field and was made with DIS; a report that stopped before the flow was computed has none either.
    flow_of = lambda v: v if v in ("dis", "raft") else ""
    flow = flow_of(str(rep.get("flow") or "").lower()) or ("dis" if applied else "")
    return {
        "applied": applied,
        "reason": "" if applied else str(rep.get("reason") or ""),
        "detail": "" if applied else str(rep.get("detail") or "")[:400],
        "flow": flow,
        "flow_requested": flow_of(str(rep.get("flow_requested") or "").lower()) or flow,
        # RAFT was asked for and could not run, DIS was used: the reason code (no_weights | no_torch | no_gpu | no_memory | failed)
        "flow_fallback": str(rep.get("flow_fallback") or "")[:40],
        "threshold": _json_num(rep.get("threshold")),
        "span": _json_int(rep.get("span")),
        "keyframe_count": _json_int(rep.get("keyframe_count")),
        "plate_frames": _json_int(rep.get("plate_frames")),
        "seconds": _json_num(rep.get("seconds"), 1),
        "replaced_area": _json_num(rep.get("replaced_area")),
        "replaced_mean": _json_num(rep.get("replaced_mean")),
        "commonality": {band: _json_num(com.get(band)) for band in ("broad", "medium", "fine")},
        "touched_count": len(touched),
        "rewritten_count": len(rewritten),
        "worst": worst,
        "image_url": image_url,
        "json_url": "%s/%s?v=%d" % (base_url, KF_COMMON_JSON, json_mtime),
    }
