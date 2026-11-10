# -*- coding: utf-8 -*-
"""Log Telemetry Parser & Fast 16-bit PNG -> 8-bit JPG Sequence Preview Cache."""
import glob
import hashlib
import os
import re
import threading
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

_STEP_LINE = re.compile(r"^\[(?P<t>\d{2}:\d{2}:\d{2})\]\s+step\s+(?P<step>\d+)\s+(?P<rest>.*)$")
_KV = re.compile(r"([A-Za-z_]+)\s+(-?\d+\.?\d*(?:[eE][-+]?\d+)?)")
_VAL_LINE = re.compile(r"^\[(?P<t>[\d:]+)\]\s+val(?P<ema>-ema)?\([^)]*\):\s*(?P<rest>.*)$")
_VAL_F = re.compile(r"f(\d+)\s+(-?[\d.]+)(?:/(-?[\d.]+))?")
_STEP_PNG = re.compile(r"step_(\d+)\.png$")
_STEP_PT = re.compile(r"step_(\d+)\.pt$")


def parse_train_log(path: str, max_bytes: int = 400_000) -> dict:
    steps = []
    vals = []
    tail_lines = []
    last_step = 0
    if not path or not os.path.isfile(path):
        return {"steps": [], "vals": [], "tail": [], "latest": {}}

    try:
        with open(path, "r", errors="replace") as f:
            if max_bytes:
                f.seek(0, 2)
                sz = f.tell()
                if sz > max_bytes:
                    f.seek(sz - max_bytes)
                    f.readline()
                else:
                    f.seek(0)
            all_lines = f.readlines()
            tail_lines = [ln.rstrip("\n") for ln in all_lines[-80:]]
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
        pass

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
                return None
        else:
            img = cv2.imread(src_abs, cv2.IMREAD_UNCHANGED)
            if img is None:
                return None
            if img.dtype == np.uint16:
                img = (img.astype(np.float32) / 257.0).clip(0, 255).astype(np.uint8)
            elif img.dtype != np.uint8:
                img = (img.astype(np.float32) * 255.0).clip(0, 255).astype(np.uint8)
            if img.ndim == 3 and img.shape[2] == 4:
                img = img[:, :, :3]
        tmp = f"{dst[:-4]}.tmp{os.getpid()}.jpg"
        if cv2.imwrite(tmp, img, [cv2.IMWRITE_JPEG_QUALITY, quality]):
            with _LOCK:
                os.replace(tmp, dst)
        else:
            if os.path.exists(tmp):
                os.remove(tmp)
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


def list_all_checkpoints() -> list:
    """Return list of checkpoints across all runs with their locked architecture metadata."""
    out = []
    if not os.path.isdir(storage.RUN_ROOT):
        return out
    for d in sorted(os.listdir(storage.RUN_ROOT)):
        if d.startswith(".") or d == "verification_report":
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


def get_viewer_frames(run_dir_rel: str, cfg: dict, mode: str = "preview", pair: str = None) -> dict:
    rdir_abs = storage.abs_from_root(run_dir_rel)
    pairs = []
    selected_pair = pair
    frames = []

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
                m = _STEP_PNG.search(p)
                if m:
                    step_items.append((int(m.group(1)), p))
            step_items.sort(key=lambda x: x[0])
            urls = batch_to_jpg_urls([p for _, p in step_items])
            for (st, _), u in zip(step_items, urls):
                if u:
                    frames.append({"url": u, "label": f"Step {st}"})
    elif mode == "infer":
        inf_dir = os.path.join(rdir_abs, "infer")
        files = sorted(glob.glob(os.path.join(inf_dir, "*.png"))) if os.path.isdir(inf_dir) else []
        urls = batch_to_jpg_urls(files)
        for fp, u in zip(files, urls):
            if u:
                frames.append({"url": u, "label": os.path.basename(fp)})
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
