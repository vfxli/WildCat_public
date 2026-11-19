# -*- coding: utf-8 -*-
"""User-Isolated Storage Manager for Assets (Sequences) and Task Run Directories."""
import contextlib
import glob
import io
import os
from pathlib import Path, PureWindowsPath
import re
import shutil
import threading
import time
import zipfile

import cv2
import numpy as np

from wildcat import color

from . import db, i18n

ROOT = db.ROOT
STORAGE_ROOT = db.STORAGE_ROOT
DEFAULT_USERS_STORAGE = os.path.join(STORAGE_ROOT, "users")
DEFAULT_RUN_ROOT = os.path.join(ROOT, "run")
DEFAULT_CONFIG_ROOT = os.path.join(ROOT, "config")
DEFAULT_CACHE_DIR = os.path.join(STORAGE_ROOT, "cache")
DEFAULT_INFER_DIR = os.path.join(STORAGE_ROOT, "infer")
DATASET_ROOT = os.path.join(ROOT, "dataset")

USERS_STORAGE = DEFAULT_USERS_STORAGE
RUN_ROOT = DEFAULT_RUN_ROOT
CONFIG_ROOT = DEFAULT_CONFIG_ROOT
CACHE_DIR = DEFAULT_CACHE_DIR
INFER_DIR = DEFAULT_INFER_DIR

_PATH_CACHE = {"loaded": False, "custom_root": ""}


def _normalize_custom_root(raw_path: str) -> str:
    s = str(raw_path or "").strip()
    if not s:
        return ""
    expanded = os.path.expanduser(s)
    return os.path.abspath(expanded)


def refresh_storage_paths(force_db: bool = True) -> str:
    """Reload `data_cache_root` from SQLite settings and update active storage root globals."""
    global USERS_STORAGE, RUN_ROOT, CONFIG_ROOT, CACHE_DIR, INFER_DIR
    if force_db or not _PATH_CACHE["loaded"]:
        try:
            raw = db.get_setting("data_cache_root", os.environ.get("WC_CACHE_ROOT", "")) or ""
        except Exception:
            raw = os.environ.get("WC_CACHE_ROOT", "") or ""
        _PATH_CACHE["custom_root"] = _normalize_custom_root(raw)
        _PATH_CACHE["loaded"] = True

    croot = _PATH_CACHE["custom_root"]
    if croot:
        USERS_STORAGE = os.path.join(croot, "users")
        RUN_ROOT = os.path.join(croot, "run")
        CONFIG_ROOT = os.path.join(croot, "config")
        CACHE_DIR = os.path.join(croot, "cache")
        INFER_DIR = os.path.join(croot, "infer")
    else:
        USERS_STORAGE = DEFAULT_USERS_STORAGE
        RUN_ROOT = DEFAULT_RUN_ROOT
        CONFIG_ROOT = DEFAULT_CONFIG_ROOT
        CACHE_DIR = DEFAULT_CACHE_DIR
        INFER_DIR = DEFAULT_INFER_DIR
    return croot


def get_data_cache_root() -> str:
    if not _PATH_CACHE["loaded"]:
        refresh_storage_paths(force_db=True)
    return _PATH_CACHE["custom_root"]


def get_users_storage() -> str:
    get_data_cache_root()
    return USERS_STORAGE


def get_run_root() -> str:
    get_data_cache_root()
    return RUN_ROOT


def get_config_root() -> str:
    get_data_cache_root()
    return CONFIG_ROOT


def get_cache_dir() -> str:
    get_data_cache_root()
    return CACHE_DIR


def get_infer_dir() -> str:
    get_data_cache_root()
    os.makedirs(INFER_DIR, exist_ok=True)
    return INFER_DIR


def get_jpg_cache_dir() -> str:
    d = os.path.join(get_cache_dir(), "jpg")
    os.makedirs(d, exist_ok=True)
    return d


def apply_subprocess_storage_env(env: dict) -> dict:
    """Populate WC_* storage root env vars for child training / evaluation / export processes."""
    env["WC_RUN_ROOT"] = get_run_root()
    env["WC_USERS_STORAGE"] = get_users_storage()
    env["WC_CONFIG_ROOT"] = get_config_root()
    env["WC_INFER_DIR"] = get_infer_dir()
    croot = get_data_cache_root()
    if croot:
        env["WC_CACHE_ROOT"] = croot
    return env


def sanitize_name(name: str, default="item") -> str:
    s = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", (name or "").strip())
    return s.strip("._") or default


def user_root(username: str) -> str:
    p = os.path.join(get_users_storage(), sanitize_name(username, "user"))
    os.makedirs(os.path.join(p, "datasets"), exist_ok=True)
    return p


def _is_under_dir(abs_target: str, base_dir: str) -> str | None:
    t_norm = os.path.normcase(os.path.abspath(abs_target))
    b_abs = os.path.abspath(base_dir)
    b_norm = os.path.normcase(b_abs)
    if t_norm == b_norm:
        return ""
    if t_norm.startswith(b_norm + os.sep):
        return Path(os.path.relpath(os.path.abspath(abs_target), b_abs)).as_posix()
    return None


def rel_to_root(abs_path: str) -> str:
    ap = os.path.abspath(abs_path)
    for base, prefix in (
        (get_run_root(), "run"),
        (DEFAULT_RUN_ROOT, "run"),
        (get_config_root(), "config"),
        (DEFAULT_CONFIG_ROOT, "config"),
        (get_users_storage(), "storage/users"),
        (DEFAULT_USERS_STORAGE, "storage/users"),
        (get_infer_dir(), "storage/infer"),
        (DEFAULT_INFER_DIR, "storage/infer"),
    ):
        sub = _is_under_dir(ap, base)
        if sub is not None:
            return f"{prefix}/{sub}" if sub else prefix
    try:
        rel = os.path.relpath(ap, ROOT)
        if not rel.startswith(".."):
            return Path(rel).as_posix()
    except ValueError:
        pass
    return Path(ap).as_posix()


def abs_from_root(path_str: str, for_write: bool = False) -> str:
    if not path_str:
        return ROOT
    norm = PureWindowsPath(str(path_str)).as_posix()
    p = Path(norm)
    if p.is_absolute():
        return str(p)

    if norm == "run" or norm.startswith("run/"):
        sub = norm[4:] if norm.startswith("run/") else ""
        primary = os.path.join(get_run_root(), sub) if sub else get_run_root()
        if for_write or os.path.exists(primary):
            return primary
        fallback = os.path.join(DEFAULT_RUN_ROOT, sub) if sub else DEFAULT_RUN_ROOT
        if os.path.exists(fallback):
            return fallback
        return primary

    if norm == "config" or norm.startswith("config/"):
        sub = norm[7:] if norm.startswith("config/") else ""
        primary = os.path.join(get_config_root(), sub) if sub else get_config_root()
        if for_write or os.path.exists(primary):
            return primary
        fallback = os.path.join(DEFAULT_CONFIG_ROOT, sub) if sub else DEFAULT_CONFIG_ROOT
        if os.path.exists(fallback):
            return fallback
        return primary

    if norm == "storage/users" or norm.startswith("storage/users/"):
        sub = norm[len("storage/users/"):] if norm.startswith("storage/users/") else ""
        primary = os.path.join(get_users_storage(), sub) if sub else get_users_storage()
        if for_write or os.path.exists(primary):
            return primary
        fallback = os.path.join(DEFAULT_USERS_STORAGE, sub) if sub else DEFAULT_USERS_STORAGE
        if os.path.exists(fallback):
            return fallback
        return primary

    if norm == "users" or norm.startswith("users/"):
        sub = norm[len("users/"):] if norm.startswith("users/") else ""
        primary = os.path.join(get_users_storage(), sub) if sub else get_users_storage()
        if for_write or os.path.exists(primary):
            return primary
        fallback = os.path.join(DEFAULT_USERS_STORAGE, sub) if sub else DEFAULT_USERS_STORAGE
        if os.path.exists(fallback):
            return fallback
        return primary

    if norm == "storage/infer" or norm.startswith("storage/infer/"):
        sub = norm[len("storage/infer/"):] if norm.startswith("storage/infer/") else ""
        primary = os.path.join(get_infer_dir(), sub) if sub else get_infer_dir()
        if for_write or os.path.exists(primary):
            return primary
        fallback = os.path.join(DEFAULT_INFER_DIR, sub) if sub else DEFAULT_INFER_DIR
        if os.path.exists(fallback):
            return fallback
        return primary

    if norm == "infer" or norm.startswith("infer/"):
        sub = norm[len("infer/"):] if norm.startswith("infer/") else ""
        primary = os.path.join(get_infer_dir(), sub) if sub else get_infer_dir()
        if for_write or os.path.exists(primary):
            return primary
        fallback = os.path.join(DEFAULT_INFER_DIR, sub) if sub else DEFAULT_INFER_DIR
        if os.path.exists(fallback):
            return fallback
        return primary

    return str(Path(ROOT) / p)


_NAT_SPLIT_RE = re.compile(r"(\d+)")
_UPLOAD_PROG_LOCK = threading.Lock()
_UPLOAD_PROGRESS: dict[str, dict] = {}


def _natural_sort_key(path_or_name: str):
    bn = PureWindowsPath(str(path_or_name or "")).name
    return tuple((0, int(p), p) if p.isdigit() else (1, 0, p.lower()) for p in _NAT_SPLIT_RE.split(bn))


def set_upload_progress(upload_id: str, stage: str, done: int = 0, total: int = 0) -> None:
    uid = str(upload_id or "").strip()
    if not uid or len(uid) > 80:
        return
    now = time.time()
    with _UPLOAD_PROG_LOCK:
        if len(_UPLOAD_PROGRESS) > 256:
            stale = [k for k, v in _UPLOAD_PROGRESS.items() if now - v.get("updated_at", 0.0) > 600.0]
            for k in stale:
                _UPLOAD_PROGRESS.pop(k, None)
        _UPLOAD_PROGRESS[uid] = {
            "stage": stage,
            "done": int(max(0, done)),
            "total": int(max(0, total)),
            "updated_at": now,
        }


def get_upload_progress(upload_id: str) -> dict | None:
    uid = str(upload_id or "").strip()
    if not uid:
        return None
    with _UPLOAD_PROG_LOCK:
        item = _UPLOAD_PROGRESS.get(uid)
        return dict(item) if item else None


def clear_upload_progress(upload_id: str) -> None:
    uid = str(upload_id or "").strip()
    if not uid:
        return
    with _UPLOAD_PROG_LOCK:
        _UPLOAD_PROGRESS.pop(uid, None)


def _sanitize_frame_stem(raw_bn: str, idx: int) -> str:
    raw_stem = PureWindowsPath(raw_bn or "").stem.strip()
    if not raw_stem:
        return f"frame_{idx:04d}"
    s = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", raw_stem)
    s = re.sub(r"_+", "_", s)
    # If non-ASCII prefix turned into leading '_' before a frame separator (e.g. '镜头.1001' -> '_.1001' or '镜头_1001' -> '_1001')
    if re.match(r"^_[\._\-]?\d+$", s) and not raw_stem[0].isascii():
        s = "seq" + (s[1:] if s.startswith(("_.", "__", "_-")) else s)
    s = s.strip("._")
    return s or f"frame_{idx:04d}"


def _unique_exr_filename(raw_bn: str, idx: int, used_lower: set[str]) -> str:
    stem = _sanitize_frame_stem(raw_bn, idx)
    cand = f"{stem}.exr"
    if cand.lower() in used_lower:
        k = 2
        while f"{stem}_{k}.exr".lower() in used_lower:
            k += 1
        cand = f"{stem}_{k}.exr"
    used_lower.add(cand.lower())
    return cand


def _dedup_sorted(paths: list) -> list:
    seen = set()
    out = []
    for p in sorted(paths, key=_natural_sort_key):
        key = os.path.normcase(os.path.abspath(p))
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def expand_sequence_files(pattern_or_dir: str) -> list:
    if not pattern_or_dir:
        return []
    p = abs_from_root(pattern_or_dir)
    if os.path.isdir(p):
        exr_files = _dedup_sorted(glob.glob(os.path.join(p, "*.exr")) + glob.glob(os.path.join(p, "*.EXR")))
        if exr_files:
            return exr_files
        files = _dedup_sorted(glob.glob(os.path.join(p, "*.png")))
        if not files:
            files = _dedup_sorted(
                glob.glob(os.path.join(p, "*.tif"))
                + glob.glob(os.path.join(p, "*.tiff"))
                + glob.glob(os.path.join(p, "*.jpg"))
                + glob.glob(os.path.join(p, "*.jpeg"))
            )
        return files
    return _dedup_sorted(glob.glob(p.replace("####", "*")))


def dir_stats(pattern_or_dir: str) -> tuple:
    """Return (frame_count, total_bytes)."""
    files = expand_sequence_files(pattern_or_dir)
    total_bytes = 0
    for f in files:
        try:
            total_bytes += os.path.getsize(f)
        except OSError:
            pass
    return len(files), total_bytes


def dir_total_bytes(dir_path: str) -> int:
    p = abs_from_root(dir_path)
    if not os.path.exists(p):
        return 0
    if os.path.isfile(p):
        return os.path.getsize(p)
    tot = 0
    for r, _, files in os.walk(p):
        for fn in files:
            try:
                tot += os.path.getsize(os.path.join(r, fn))
            except OSError:
                pass
    return tot


def user_dataset_dir(username: str, dataset_name: str) -> str:
    p = os.path.join(user_root(username), "datasets", sanitize_name(dataset_name, "dataset"))
    return p


def _norm_img_float(img: np.ndarray) -> np.ndarray:
    """Normalize image array into [0, 1] perceptual range for MAE & cross-frame alignment checks."""
    if img.dtype == np.uint16:
        arr = img.astype(np.float32) / 65535.0
    elif img.dtype == np.uint8:
        arr = img.astype(np.float32) / 255.0
    else:
        arr = img.astype(np.float32)
        pos = np.maximum(0.0, arr)
        if float(pos.max()) > 1.05:
            # Reinhard tone-map HDR ACEScg float values into [0, 1] for stable percentage MAE comparison
            arr = pos / (1.0 + pos)
        else:
            arr = np.clip(arr, 0.0, 1.0)
    if arr.ndim == 2:
        arr = arr[..., None]
    return arr


def _payload_byte_len(content) -> int:
    if isinstance(content, (bytes, bytearray, memoryview)):
        return len(content)
    if isinstance(content, (str, os.PathLike)):
        return int(os.path.getsize(content))
    if hasattr(content, "seek") and hasattr(content, "tell"):
        pos = content.tell()
        content.seek(0, os.SEEK_END)
        sz = int(content.tell())
        content.seek(pos)
        return sz
    return 0


def _read_zip_member_checked(zf: zipfile.ZipFile, member_name: str, max_single_bytes: int | None, role_label: str) -> bytes:
    data = zf.read(member_name)
    if max_single_bytes is not None and len(data) > max_single_bytes:
        raise ValueError(i18n.t("server.uploadTooLarge", role=role_label))
    return data


def _read_filelike_all(fh) -> bytes:
    fh.seek(0)
    return fh.read()


def _collect_upload_readers(
    file_tuples: list,
    role_label: str,
    stack: contextlib.ExitStack,
    restricted: bool = False,
) -> list[tuple[str, object]]:
    """Build a sorted list of `(basename, lazy_bytes_reader)` without loading all uploaded files into RAM at once.
    Temporary-user caps (300 files, 128 MB/file, 1 GB decompressed) apply only when `restricted=True`."""
    max_role_files = 300 if restricted else None
    max_single_bytes = 128 * 1024 * 1024 if restricted else None
    max_total_bytes = 1024 * 1024 * 1024 if restricted else None

    entries = []
    total_uncompressed = 0
    valid_exts = (".exr", ".png", ".tif", ".tiff", ".jpg", ".jpeg")
    for fname, content in file_tuples:
        lower = (fname or "").lower()
        if lower.endswith(".zip"):
            zlen = _payload_byte_len(content)
            if max_total_bytes is not None and zlen > max_total_bytes:
                raise ValueError(i18n.t("server.uploadTooLarge", role=role_label))
            if isinstance(content, (bytes, bytearray, memoryview)):
                zsrc = io.BytesIO(content)
            elif isinstance(content, (str, os.PathLike)):
                zsrc = os.fspath(content)
            else:
                content.seek(0)
                zsrc = content
            zf = stack.enter_context(zipfile.ZipFile(zsrc, "r"))
            infos = [zi for zi in sorted(zf.infolist(), key=lambda x: x.filename) if not zi.is_dir()]
            if max_role_files is not None and len(infos) > max_role_files * 2:
                raise ValueError(i18n.t("server.uploadTooManyFiles", role=role_label, limit=max_role_files))
            for zi in infos:
                bn = PureWindowsPath(zi.filename).name
                if bn.startswith(".") or "__MACOSX" in zi.filename:
                    continue
                ext = PureWindowsPath(bn).suffix.lower()
                if ext in valid_exts:
                    if max_single_bytes is not None and zi.file_size > max_single_bytes:
                        raise ValueError(i18n.t("server.uploadTooLarge", role=role_label))
                    total_uncompressed += int(zi.file_size)
                    if max_total_bytes is not None and total_uncompressed > max_total_bytes:
                        raise ValueError(i18n.t("server.uploadTooLarge", role=role_label))
                    if max_role_files is not None and len(entries) >= max_role_files:
                        raise ValueError(i18n.t("server.uploadTooManyFiles", role=role_label, limit=max_role_files))
                    entries.append(
                        (
                            bn,
                            lambda z=zf, zname=zi.filename, lim=max_single_bytes, rlbl=role_label: _read_zip_member_checked(
                                z, zname, lim, rlbl
                            ),
                        )
                    )
        else:
            bn = PureWindowsPath(fname or "").name
            ext = PureWindowsPath(bn).suffix.lower()
            if ext in valid_exts:
                flen = _payload_byte_len(content)
                if max_single_bytes is not None and flen > max_single_bytes:
                    raise ValueError(i18n.t("server.uploadTooLarge", role=role_label))
                total_uncompressed += flen
                if max_total_bytes is not None and total_uncompressed > max_total_bytes:
                    raise ValueError(i18n.t("server.uploadTooLarge", role=role_label))
                if max_role_files is not None and len(entries) >= max_role_files:
                    raise ValueError(i18n.t("server.uploadTooManyFiles", role=role_label, limit=max_role_files))
                if isinstance(content, (bytes, bytearray, memoryview)):
                    b_ref = bytes(content) if not isinstance(content, bytes) else content
                    reader = lambda b=b_ref: b
                elif isinstance(content, (str, os.PathLike)):
                    p_str = os.fspath(content)
                    reader = lambda p=p_str: Path(p).read_bytes()
                else:
                    reader = lambda fh=content: _read_filelike_all(fh)
                entries.append((bn, reader))

    if not entries:
        raise ValueError(i18n.t("server.uploadNoValidImages", role=role_label))

    entries.sort(key=lambda x: _natural_sort_key(x[0]))
    return entries


def _decode_checked_image(data_bytes: bytes, bn: str, role_label: str, idx: int, restricted: bool = False) -> tuple[str, str, np.ndarray]:
    max_img_dim = 8192 if restricted else 16384
    try:
        img = color.decode_image_bytes(data_bytes, bn)
    except Exception as e:
        raise ValueError(i18n.t("server.uploadDecodeFailed", role=role_label, filename=bn, err=e))
    ih, iw = int(img.shape[0]), int(img.shape[1])
    if ih <= 0 or iw <= 0 or ih > max_img_dim or iw > max_img_dim:
        raise ValueError(i18n.t("server.uploadResolutionExceeded", filename=bn, w=iw, h=ih, max_dim=max_img_dim))
    stem = sanitize_name(PureWindowsPath(bn).stem, f"frame_{idx:04d}")
    return stem, bn, img


def _extract_upload_images(file_tuples: list, role_label: str, restricted: bool = False) -> list:
    """Decode uploaded files (.exr/.png/.tif/.jpg/.zip) into sorted list of `(stem_name, filename, float32_ndarray)`."""
    with contextlib.ExitStack() as stack:
        entries = _collect_upload_readers(file_tuples, role_label, stack, restricted=restricted)
        decoded = []
        for idx, (bn, reader) in enumerate(entries, start=1):
            decoded.append(_decode_checked_image(reader(), bn, role_label, idx, restricted=restricted))
        return decoded


_FRAME_NUM_RE = re.compile(r"(\d+)(?=\.[^.]+$)")


def _extract_frame_num(filename: str) -> int | None:
    bn = PureWindowsPath(filename or "").name
    m = _FRAME_NUM_RE.search(bn)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def _require_explicit_colorspace(input_colorspace: str, working_colorspace: str = "ACEScg") -> tuple[str, str]:
    raw_in = str(input_colorspace or "").strip()
    if not raw_in or raw_in.lower() == "auto":
        raise ValueError(i18n.t("server.inputColorspaceRequired"))
    try:
        src_cs = color.normalize_explicit_colorspace(raw_in)
    except ValueError:
        raise ValueError(i18n.t("server.inputColorspaceInvalid", cs=raw_in[:60]))
    eff_work = "Raw" if src_cs == "Raw" else (str(working_colorspace or "ACEScg").strip() or "ACEScg")
    return src_cs, eff_work


def _is_dummy_constant_alpha(a_min: float, a_max: float) -> bool:
    """True when a 4th channel is uniformly 0.0 or uniformly 1.0 (e.g. Nuke empty rgba.alpha or solid PNG alpha)."""
    if a_min > a_max:
        return True
    if (a_max - a_min) > 1e-3:
        return False
    return abs(a_min - 1.0) <= 1e-3 or abs(a_min - 0.0) <= 1e-3


def _has_varying_alpha(imgs: list) -> bool:
    a_min, a_max = float("inf"), float("-inf")
    for im in imgs:
        if im is not None and im.ndim == 3 and im.shape[2] >= 4:
            a = im[..., 3]
            a_min = min(a_min, float(np.min(a)))
            a_max = max(a_max, float(np.max(a)))
            if not _is_dummy_constant_alpha(a_min, a_max):
                return True
    return False


def _save_eval_frames_to_dir(
    eval_abs: str,
    seq_tuples: list,
    expected_w: int = 0,
    expected_h: int = 0,
    expected_in_ch: int = 0,
    input_colorspace: str = "",
    working_colorspace: str = "ACEScg",
    restricted: bool = False,
    upload_id: str = "",
) -> int:
    src_cs, eff_work_cs = _require_explicit_colorspace(input_colorspace, working_colorspace)
    role_lbl = i18n.t("server.roleEvalSeq")
    with contextlib.ExitStack() as stack:
        entries = _collect_upload_readers(seq_tuples, role_lbl, stack, restricted=restricted)
        if len(entries) < 2:
            raise ValueError(i18n.t("server.evalSeqMinFrames", count=len(entries)))

        tmp_eval_abs = f"{eval_abs}.tmp_{os.getpid()}"
        if os.path.exists(tmp_eval_abs):
            shutil.rmtree(tmp_eval_abs, ignore_errors=True)
        os.makedirs(tmp_eval_abs, exist_ok=True)
        try:
            h0 = int(expected_h or 0)
            w0 = int(expected_w or 0)
            total_seq = len(entries)
            used_seq_names: set[str] = set()
            set_upload_progress(upload_id, "saving_seq", 0, total_seq)
            for idx, (bn, reader) in enumerate(entries, start=1):
                _, s_bn, s_im = _decode_checked_image(reader(), bn, role_lbl, idx, restricted=restricted)
                sh, sw = int(s_im.shape[0]), int(s_im.shape[1])
                if idx == 1 and (w0 == 0 or h0 == 0):
                    h0 = h0 or sh
                    w0 = w0 or sw
                if (sw, sh) != (w0, h0):
                    raise ValueError(
                        i18n.t(
                            "server.evalSeqDatasetSizeMismatch",
                            index=idx,
                            filename=s_bn,
                            sw=sw,
                            sh=sh,
                            w0=w0,
                            h0=h0,
                        )
                    )
                s_out = s_im
                if expected_in_ch in (1, 3, 4) and s_out.ndim == 3 and s_out.shape[2] > expected_in_ch:
                    s_out = s_out[..., :expected_in_ch]
                if src_cs not in ("Raw", eff_work_cs) and eff_work_cs != "Raw" and s_out.shape[2] >= 3:
                    s_out = color.transform_colorspace(s_out, src_cs, eff_work_cs)
                out_fn = _unique_exr_filename(s_bn, idx, used_seq_names)
                color.write_exr_file(os.path.join(tmp_eval_abs, out_fn), s_out, bit_depth="half", colorspace=eff_work_cs)
                set_upload_progress(upload_id, "saving_seq", idx, total_seq)
                del s_im, s_out
        except Exception:
            shutil.rmtree(tmp_eval_abs, ignore_errors=True)
            raise

        if os.path.exists(eval_abs):
            shutil.rmtree(eval_abs, ignore_errors=True)
        os.replace(tmp_eval_abs, eval_abs)
        return len(entries)


def validate_and_save_paired_dataset(
    username: str,
    dataset_name: str,
    source_tuples: list,
    target_tuples: list,
    max_error_pct: float = 35.0,
    strategy: str = "standard",
    enable_check: bool = True,
    input_colorspace: str = "",
    working_colorspace: str = "ACEScg",
    seq_tuples: list | None = None,
    restricted: bool = False,
    upload_id: str = "",
) -> dict:
    """
    Validate a paired (Source Input, Ground Truth Target) sequence upload before saving:
      - Always checks basic physical requirement: frame count > 0, equal frame count, equal (width, height).
      - When `enable_check=True` (default ON):
          1. Missing frame & Frame number alignment check (extracts frame numbers from filenames to report exact missing/shifted frame numbers).
          2. Cross-frame content shift check (detects if frame i in Source actually matches frame j != i in Target).
          3. 100% duplicate identical sequence check (MAE == 0).
          4. Generous pixel error tolerance check (default 35% MAE threshold).
    """
    src_cs, eff_work_cs = _require_explicit_colorspace(input_colorspace, working_colorspace)
    src_role = i18n.t("server.roleSource")
    tgt_role = i18n.t("server.roleTarget")
    eval_role = i18n.t("server.roleEvalSeq")
    with contextlib.ExitStack() as stack:
        src_entries = _collect_upload_readers(source_tuples, src_role, stack, restricted=restricted)
        tgt_entries = _collect_upload_readers(target_tuples, tgt_role, stack, restricted=restricted)
        seq_entries = _collect_upload_readers(seq_tuples or [], eval_role, stack, restricted=restricted)
        if len(seq_entries) < 2:
            raise ValueError(i18n.t("server.evalSeqMinFrames", count=len(seq_entries)))

        src_fnums = [_extract_frame_num(bn) for bn, _ in src_entries]
        tgt_fnums = [_extract_frame_num(bn) for bn, _ in tgt_entries]
        has_fnums = all(n is not None for n in src_fnums) and all(n is not None for n in tgt_fnums)

        # 1. Missing Frame / Unequal Frame Count Detection (with exact missing frame number report!)
        if len(src_entries) != len(tgt_entries):
            detail = ""
            if has_fnums:
                s_set, t_set = set(src_fnums), set(tgt_fnums)
                miss_in_tgt = sorted(s_set - t_set)
                miss_in_src = sorted(t_set - s_set)
                parts = []
                if miss_in_tgt:
                    parts.append(i18n.t("server.missingInTarget", frames=miss_in_tgt[:10]))
                if miss_in_src:
                    parts.append(i18n.t("server.missingInSource", frames=miss_in_src[:10]))
                if parts:
                    detail = i18n.t("server.missingDetailPrefix", parts="; ".join(parts))
            raise ValueError(
                i18n.t("server.pairCountMismatch", src_count=len(src_entries), tgt_count=len(tgt_entries), detail=detail)
            )

        # 2. Frame Number Alignment Check (when enable_check is ON)
        if enable_check and has_fnums:
            for idx, (sn, tn, (s_bn, _), (t_bn, _)) in enumerate(
                zip(src_fnums, tgt_fnums, src_entries, tgt_entries)
            ):
                if sn != tn:
                    s_set, t_set = set(src_fnums), set(tgt_fnums)
                    miss_in_tgt = sorted(s_set - t_set)
                    miss_in_src = sorted(t_set - s_set)
                    extra = ""
                    if miss_in_tgt or miss_in_src:
                        extra = i18n.t("server.pairFrameExtraMissing", miss_tgt=miss_in_tgt[:8], miss_src=miss_in_src[:8])
                    raise ValueError(
                        i18n.t(
                            "server.pairFrameNumMismatch",
                            index=idx + 1,
                            src_name=s_bn,
                            src_num=sn,
                            tgt_name=t_bn,
                            tgt_num=tn,
                            extra=extra,
                        )
                    )

        # Stream pair-by-pair when keyframe count > 32 so large sequences never hold hundreds of float32 frames in RAM
        keep_in_ram = len(src_entries) <= 32
        cached_pairs = []
        h0, w0 = 0, 0
        in_ch = 99
        out_ch = 99
        src_a_min, src_a_max = float("inf"), float("-inf")
        tgt_a_min, tgt_a_max = float("inf"), float("-inf")
        maes = []
        worst_mae = -1.0
        worst_idx = 0
        worst_pair_names = ("", "")
        thumbs_s = []
        thumbs_t = []

        total_pairs = len(src_entries)
        set_upload_progress(upload_id, "checking", 0, total_pairs)
        for idx, ((s_bn_raw, s_reader), (t_bn_raw, t_reader)) in enumerate(zip(src_entries, tgt_entries), start=1):
            _, s_bn, s_im = _decode_checked_image(s_reader(), s_bn_raw, src_role, idx, restricted=restricted)
            _, t_bn, t_im = _decode_checked_image(t_reader(), t_bn_raw, tgt_role, idx, restricted=restricted)
            set_upload_progress(upload_id, "checking", idx, total_pairs)
            sh, sw = int(s_im.shape[0]), int(s_im.shape[1])
            th, tw = int(t_im.shape[0]), int(t_im.shape[1])
            if idx == 1:
                h0, w0 = sh, sw
            if sh != th or sw != tw:
                raise ValueError(
                    i18n.t(
                        "server.pairSizeMismatch",
                        index=idx,
                        src_name=s_bn,
                        sw=sw,
                        sh=sh,
                        tgt_name=t_bn,
                        tw=tw,
                        th=th,
                    )
                )
            if sh != h0 or sw != w0:
                raise ValueError(
                    i18n.t("server.seqInternalSizeMismatch", w0=w0, h0=h0, index=idx, sw=sw, sh=sh)
                )

            sc = 1 if s_im.ndim == 2 else int(s_im.shape[2])
            tc = 1 if t_im.ndim == 2 else int(t_im.shape[2])
            in_ch = min(in_ch, sc)
            out_ch = min(out_ch, tc)
            if s_im.ndim == 3 and s_im.shape[2] >= 4:
                sa = s_im[..., 3]
                src_a_min = min(src_a_min, float(np.min(sa)))
                src_a_max = max(src_a_max, float(np.max(sa)))
            if t_im.ndim == 3 and t_im.shape[2] >= 4:
                ta = t_im[..., 3]
                tgt_a_min = min(tgt_a_min, float(np.min(ta)))
                tgt_a_max = max(tgt_a_max, float(np.max(ta)))

            s_f = _norm_img_float(s_im)
            t_f = _norm_img_float(t_im)
            c_cmp = min(s_f.shape[2], t_f.shape[2], 3)
            frame_mae = float(np.mean(np.abs(s_f[..., :c_cmp] - t_f[..., :c_cmp])))
            maes.append(frame_mae)
            if frame_mae > worst_mae:
                worst_mae = frame_mae
                worst_idx = idx - 1
                worst_pair_names = (s_bn, t_bn)

            if enable_check and len(src_entries) > 1:
                thumbs_s.append(cv2.resize(s_f[..., :c_cmp], (64, 64), interpolation=cv2.INTER_AREA))
                thumbs_t.append(cv2.resize(t_f[..., :c_cmp], (64, 64), interpolation=cv2.INTER_AREA))

            if keep_in_ram:
                cached_pairs.append((s_bn, s_im, t_bn, t_im))
            else:
                del s_im, t_im
            del s_f, t_f

        src_dummy_alpha = _is_dummy_constant_alpha(src_a_min, src_a_max)
        tgt_dummy_alpha = _is_dummy_constant_alpha(tgt_a_min, tgt_a_max)
        if in_ch >= 3 and out_ch >= 3:
            if in_ch == 4 and out_ch == 4 and strategy != "alpha_matting":
                if src_dummy_alpha and tgt_dummy_alpha and abs(src_a_min - tgt_a_min) <= 1e-3:
                    in_ch = 3
                    out_ch = 3
            elif in_ch != out_ch:
                if in_ch == 4 and src_dummy_alpha:
                    in_ch = 3
                if out_ch == 4 and tgt_dummy_alpha:
                    out_ch = 3

        avg_mae = float(np.mean(maes)) if maes else 0.0

        if enable_check:
            # 3. Reject 100% identical duplicate sequences (only when channels match)
            if in_ch == out_ch and worst_mae < 1e-5:
                raise ValueError(i18n.t("server.pairDuplicateSeq"))

            # 4. Cross-Frame Content Shift Detection (detects if frame i is misaligned and actually matches frame j != i)
            if in_ch == out_ch and len(thumbs_s) > 1 and strategy != "alpha_matting":
                for i in range(len(thumbs_s)):
                    self_diff = float(np.mean(np.abs(thumbs_s[i] - thumbs_t[i])))
                    if self_diff > 0.02:
                        for j in range(len(thumbs_t)):
                            if i == j:
                                continue
                            cross_diff = float(np.mean(np.abs(thumbs_s[i] - thumbs_t[j])))
                            if cross_diff * 1.6 < self_diff and (self_diff - cross_diff) > 0.012:
                                s_name = src_entries[i][0]
                                t_i_name = tgt_entries[i][0]
                                t_j_name = tgt_entries[j][0]
                                raise ValueError(
                                    i18n.t(
                                        "server.pairCrossFrameShift",
                                        i=i + 1,
                                        src_name=s_name,
                                        tgt_i_name=t_i_name,
                                        self_pct=f"{self_diff * 100:.1f}",
                                        j=j + 1,
                                        tgt_j_name=t_j_name,
                                        cross_pct=f"{cross_diff * 100:.1f}",
                                    )
                                )

            # 5. Generous Pixel Error Threshold Check
            if strategy != "alpha_matting" and max_error_pct > 0:
                limit = float(max_error_pct) / 100.0
                if worst_mae > limit:
                    raise ValueError(
                        i18n.t(
                            "server.pairMaeExceeded",
                            index=worst_idx + 1,
                            src_name=worst_pair_names[0],
                            tgt_name=worst_pair_names[1],
                            worst_pct=f"{worst_mae * 100:.2f}",
                            avg_pct=f"{avg_mae * 100:.2f}",
                            limit_pct=f"{max_error_pct:.1f}",
                        )
                    )

        # Validation passed -> write clean paired dataset directory in .exr (ACEScg 16-bit half-float)
        ds_abs = user_dataset_dir(username, dataset_name)
        src_abs = os.path.join(ds_abs, "source")
        tgt_abs = os.path.join(ds_abs, "target")
        if os.path.exists(ds_abs):
            shutil.rmtree(ds_abs, ignore_errors=True)
        os.makedirs(src_abs, exist_ok=True)
        os.makedirs(tgt_abs, exist_ok=True)

        norm_in_ch = 3 if in_ch == 3 else (4 if in_ch >= 4 else 1)
        norm_out_ch = 3 if out_ch == 3 else (4 if out_ch >= 4 else 1)

        if keep_in_ram:
            pair_iter = cached_pairs
        else:
            pair_iter = (
                (
                    s_bn_raw,
                    _decode_checked_image(s_reader(), s_bn_raw, src_role, idx, restricted=restricted)[2],
                    t_bn_raw,
                    _decode_checked_image(t_reader(), t_bn_raw, tgt_role, idx, restricted=restricted)[2],
                )
                for idx, ((s_bn_raw, s_reader), (t_bn_raw, t_reader)) in enumerate(zip(src_entries, tgt_entries), start=1)
            )

        used_src_names: set[str] = set()
        used_tgt_names: set[str] = set()
        set_upload_progress(upload_id, "saving_pairs", 0, total_pairs)
        for idx, (s_bn, s_im, t_bn, t_im) in enumerate(pair_iter, start=1):
            s_out = s_im[..., :norm_in_ch] if (s_im.ndim == 3 and s_im.shape[2] > norm_in_ch) else s_im
            t_out = t_im[..., :norm_out_ch] if (t_im.ndim == 3 and t_im.shape[2] > norm_out_ch) else t_im
            if src_cs not in ("Raw", eff_work_cs) and eff_work_cs != "Raw" and s_out.shape[2] >= 3:
                s_out = color.transform_colorspace(s_out, src_cs, eff_work_cs)
            if src_cs not in ("Raw", eff_work_cs) and eff_work_cs != "Raw" and t_out.shape[2] >= 3:
                t_out = color.transform_colorspace(t_out, src_cs, eff_work_cs)
            s_fn = _unique_exr_filename(s_bn, idx, used_src_names)
            t_fn = _unique_exr_filename(t_bn, idx, used_tgt_names)
            color.write_exr_file(os.path.join(src_abs, s_fn), s_out, bit_depth="half", colorspace=eff_work_cs)
            color.write_exr_file(os.path.join(tgt_abs, t_fn), t_out, bit_depth="half", colorspace=eff_work_cs)
            set_upload_progress(upload_id, "saving_pairs", idx, total_pairs)

        frame_total = len(src_entries)

    seq_rel = ""
    seq_count = 0
    if seq_tuples:
        eval_abs = os.path.join(ds_abs, "eval")
        try:
            seq_count = _save_eval_frames_to_dir(
                eval_abs,
                seq_tuples,
                expected_w=w0,
                expected_h=h0,
                expected_in_ch=norm_in_ch,
                input_colorspace=src_cs,
                working_colorspace=eff_work_cs,
                restricted=restricted,
                upload_id=upload_id,
            )
            seq_rel = rel_to_root(eval_abs)
        except Exception:
            shutil.rmtree(ds_abs, ignore_errors=True)
            raise

    set_upload_progress(upload_id, "done", 1, 1)
    ds_rel = rel_to_root(ds_abs)
    src_rel = rel_to_root(src_abs)
    tgt_rel = rel_to_root(tgt_abs)
    tot_bytes = dir_total_bytes(ds_rel)

    return {
        "rel_path": ds_rel,
        "source_path": src_rel,
        "target_path": tgt_rel,
        "seq_path": seq_rel,
        "frame_count": frame_total,
        "seq_frame_count": seq_count,
        "width": w0,
        "height": h0,
        "in_channels": norm_in_ch,
        "out_channels": norm_out_ch,
        "avg_mae": round(avg_mae, 6),
        "max_mae": round(worst_mae, 6),
        "strategy": strategy if enable_check else "off",
        "input_colorspace": src_cs,
        "working_colorspace": eff_work_cs,
        "size_bytes": tot_bytes,
    }


def attach_eval_sequence_to_dataset(
    username: str,
    dataset_name: str,
    ds_rel: str,
    old_seq_rel: str,
    expected_w: int,
    expected_h: int,
    expected_in_ch: int,
    seq_tuples: list,
    input_colorspace: str = "",
    working_colorspace: str = "ACEScg",
    restricted: bool = False,
    upload_id: str = "",
) -> dict:
    """Attach or replace an optional full validation sequence (`eval/`) on an existing dataset."""
    src_cs, eff_work_cs = _require_explicit_colorspace(input_colorspace, working_colorspace)
    abs_ds = os.path.abspath(abs_from_root(ds_rel))
    users_abs = os.path.abspath(USERS_STORAGE)
    if os.path.normcase(abs_ds).startswith(os.path.normcase(users_abs + os.sep)):
        eval_abs = os.path.join(abs_ds, "eval")
    else:
        # Built-in shared demo dataset under dataset/: store user override inside user's isolated storage
        eval_abs = os.path.join(user_dataset_dir(username, f"{dataset_name}_eval"), "eval")

    if old_seq_rel and os.path.abspath(abs_from_root(old_seq_rel)) != os.path.abspath(eval_abs):
        delete_asset_files(old_seq_rel)

    seq_count = _save_eval_frames_to_dir(
        eval_abs,
        seq_tuples,
        expected_w=expected_w,
        expected_h=expected_h,
        expected_in_ch=expected_in_ch,
        input_colorspace=src_cs,
        working_colorspace=eff_work_cs,
        restricted=restricted,
        upload_id=upload_id,
    )
    set_upload_progress(upload_id, "done", 1, 1)
    seq_rel = rel_to_root(eval_abs)
    _, seq_bytes = dir_stats(seq_rel)
    return {
        "seq_path": seq_rel,
        "seq_frame_count": seq_count,
        "seq_bytes": seq_bytes,
    }


def inspect_paired_dirs(source_rel: str, target_rel: str, seq_rel: str = "") -> dict:
    """Compute paired & optional eval sequence stats for existing directories."""
    s_files = expand_sequence_files(source_rel)
    t_files = expand_sequence_files(target_rel)
    seq_cnt, seq_b = dir_stats(seq_rel) if seq_rel else (0, 0)
    n = min(len(s_files), len(t_files))
    if n == 0:
        return {
            "frame_count": 0,
            "seq_frame_count": seq_cnt,
            "width": 0,
            "height": 0,
            "in_channels": 3,
            "out_channels": 3,
            "avg_mae": 0.0,
            "max_mae": 0.0,
            "size_bytes": seq_b,
        }
    try:
        s0 = color.read_image_float(s_files[0], input_colorspace="ACEScg", working_colorspace="ACEScg")
        t0 = color.read_image_float(t_files[0], input_colorspace="ACEScg", working_colorspace="ACEScg")
    except Exception:
        s0, t0 = None, None
    h, w = (int(s0.shape[0]), int(s0.shape[1])) if s0 is not None else (0, 0)
    in_ch = (1 if s0.ndim == 2 else int(s0.shape[2])) if s0 is not None else 3
    out_ch = (1 if t0.ndim == 2 else int(t0.shape[2])) if t0 is not None else 3
    if in_ch >= 3 and out_ch >= 3:
        if in_ch == 4 and out_ch == 4 and not _has_varying_alpha([s0]) and not _has_varying_alpha([t0]):
            sa0 = float(np.min(s0[..., 3])) if (s0 is not None and s0.ndim == 3 and s0.shape[2] >= 4) else 1.0
            ta0 = float(np.min(t0[..., 3])) if (t0 is not None and t0.ndim == 3 and t0.shape[2] >= 4) else 1.0
            if abs(sa0 - ta0) <= 1e-3:
                in_ch = 3
                out_ch = 3
        elif in_ch != out_ch:
            if in_ch == 4 and not _has_varying_alpha([s0]):
                in_ch = 3
            if out_ch == 4 and not _has_varying_alpha([t0]):
                out_ch = 3

    maes = []
    for sp, tp in zip(s_files[:n], t_files[:n]):
        try:
            si = color.read_image_float(sp, input_colorspace="ACEScg", working_colorspace="ACEScg")
            ti = color.read_image_float(tp, input_colorspace="ACEScg", working_colorspace="ACEScg")
        except Exception:
            si, ti = None, None
        if si is not None and ti is not None and si.shape[:2] == ti.shape[:2]:
            sf = _norm_img_float(si)
            tf = _norm_img_float(ti)
            c = min(sf.shape[2], tf.shape[2], 3)
            maes.append(float(np.mean(np.abs(sf[..., :c] - tf[..., :c]))))

    avg_m = float(np.mean(maes)) if maes else 0.0
    max_m = float(np.max(maes)) if maes else 0.0
    _, sb = dir_stats(source_rel)
    _, tb = dir_stats(target_rel)
    return {
        "frame_count": n,
        "seq_frame_count": seq_cnt,
        "width": w,
        "height": h,
        "in_channels": in_ch,
        "out_channels": out_ch,
        "avg_mae": round(avg_m, 6),
        "max_mae": round(max_m, 6),
        "size_bytes": sb + tb + seq_b,
    }


def delete_asset_files(rel_path: str):
    """Safely delete asset directory if it resides inside active or default users storage."""
    if not rel_path:
        return
    abs_p = os.path.abspath(abs_from_root(rel_path))
    for base in {os.path.abspath(get_users_storage()), os.path.abspath(DEFAULT_USERS_STORAGE)}:
        if os.path.normcase(abs_p).startswith(os.path.normcase(base + os.sep)) and os.path.exists(abs_p):
            shutil.rmtree(abs_p, ignore_errors=True)
            break


def delete_task_files(run_dir_rel: str, config_path_rel: str):
    """Safely delete task run directory under run/, dedicated infer outputs under infer/, and config file under config/."""
    node_bn = ""
    if run_dir_rel:
        abs_run = os.path.abspath(abs_from_root(run_dir_rel))
        node_bn = os.path.basename(abs_run)
        for base in {os.path.abspath(get_run_root()), os.path.abspath(DEFAULT_RUN_ROOT)}:
            if (
                os.path.normcase(abs_run).startswith(os.path.normcase(base + os.sep))
                and node_bn != "verification_report"
                and os.path.exists(abs_run)
            ):
                shutil.rmtree(abs_run, ignore_errors=True)
                break
    if node_bn and node_bn != "verification_report":
        for inf_base in {os.path.abspath(get_infer_dir()), os.path.abspath(DEFAULT_INFER_DIR)}:
            cand_inf = os.path.abspath(os.path.join(inf_base, node_bn))
            if os.path.normcase(cand_inf).startswith(os.path.normcase(inf_base + os.sep)) and os.path.isdir(cand_inf):
                shutil.rmtree(cand_inf, ignore_errors=True)
    if config_path_rel:
        abs_cfg = os.path.abspath(abs_from_root(config_path_rel))
        for base in {os.path.abspath(get_config_root()), os.path.abspath(DEFAULT_CONFIG_ROOT)}:
            if (
                os.path.normcase(abs_cfg).startswith(os.path.normcase(base + os.sep))
                and os.path.basename(abs_cfg) != "default.json"
                and os.path.isfile(abs_cfg)
            ):
                try:
                    os.remove(abs_cfg)
                except OSError:
                    pass
                break


def _fast_dir_bytes(path_abs: str, max_entries: int = 80_000) -> int:
    if not path_abs or not os.path.exists(path_abs):
        return 0
    if os.path.isfile(path_abs):
        try:
            return os.path.getsize(path_abs)
        except OSError:
            return 0
    total = 0
    seen_inodes = set()
    stack = [path_abs]
    visited = 0
    while stack and visited < max_entries:
        cur = stack.pop()
        try:
            with os.scandir(cur) as it:
                for entry in it:
                    visited += 1
                    if visited >= max_entries:
                        break
                    try:
                        if entry.is_symlink():
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                        elif entry.is_file(follow_symlinks=False):
                            st = entry.stat(follow_symlinks=False)
                            if st.st_ino and st.st_nlink > 1:
                                ik = (st.st_dev, st.st_ino)
                                if ik in seen_inodes:
                                    continue
                                seen_inodes.add(ik)
                            total += st.st_size
                    except OSError:
                        pass
        except OSError:
            pass
    return total


def get_storage_disk_stats() -> dict:
    """Return active storage directory paths and disk space breakdown."""
    croot = get_data_cache_root()
    users_dir = get_users_storage()
    run_dir = get_run_root()
    cfg_dir = get_config_root()
    cache_dir = get_cache_dir()
    infer_dir = get_infer_dir()
    probe_dir = croot if croot else STORAGE_ROOT
    os.makedirs(probe_dir, exist_ok=True)
    try:
        du = shutil.disk_usage(probe_dir)
        total_gb = round(du.total / (1024**3), 2)
        used_gb = round(du.used / (1024**3), 2)
        free_gb = round(du.free / (1024**3), 2)
    except OSError:
        total_gb = used_gb = free_gb = 0.0

    db_bytes = 0
    for sfx in ("", "-wal", "-shm"):
        p = db.DB_PATH + sfx
        if os.path.isfile(p):
            try:
                db_bytes += os.path.getsize(p)
            except OSError:
                pass

    return {
        "data_cache_root": croot,
        "is_custom": bool(croot),
        "default_root_hint": ROOT,
        "effective_paths": {
            "users_dir": users_dir,
            "run_dir": run_dir,
            "config_dir": cfg_dir,
            "cache_dir": cache_dir,
            "infer_dir": infer_dir,
            "db_path": db.DB_PATH,
        },
        "disk": {
            "total_gb": total_gb,
            "used_gb": used_gb,
            "free_gb": free_gb,
            "users_mb": round(_fast_dir_bytes(users_dir) / (1024**2), 1),
            "run_mb": round(_fast_dir_bytes(run_dir) / (1024**2), 1),
            "infer_mb": round(_fast_dir_bytes(infer_dir) / (1024**2), 1),
            "cache_mb": round(_fast_dir_bytes(cache_dir) / (1024**2), 1),
            "db_mb": round(db_bytes / (1024**2), 2),
        },
    }


def _merge_directory_contents(src_dir: str, dst_dir: str, skip_names: set[str] | None = None) -> int:
    """Safely move top-level entries from `src_dir` into `dst_dir` without overwriting existing entries."""
    if not os.path.isdir(src_dir):
        return 0
    s_abs = os.path.abspath(src_dir)
    d_abs = os.path.abspath(dst_dir)
    if os.path.normcase(s_abs) == os.path.normcase(d_abs):
        return 0
    os.makedirs(d_abs, exist_ok=True)
    moved = 0
    skips = skip_names or set()
    for name in os.listdir(s_abs):
        if name in skips or name.startswith(".tmp_"):
            continue
        sp = os.path.join(s_abs, name)
        dp = os.path.join(d_abs, name)
        if os.path.exists(dp):
            continue
        shutil.move(sp, dp)
        moved += 1
    return moved


def configure_data_cache_root(
    new_path: str,
    migrate_existing: bool = False,
    has_active_jobs: bool = False,
) -> dict:
    """Validate, create, optionally migrate, and persist the custom data & cache root directory."""
    norm_new = _normalize_custom_root(new_path)
    old_users = get_users_storage()
    old_run = get_run_root()
    old_cfg = get_config_root()
    old_cache = get_cache_dir()
    old_infer = get_infer_dir()

    if norm_new:
        # Guard against pointing cache directory inside core source code folders
        for forbidden_sub in ("wildcat", "backend", "frontend", "dataset", ".git", ".venv", "runtime"):
            forb_abs = os.path.abspath(os.path.join(ROOT, forbidden_sub))
            n_case = os.path.normcase(norm_new)
            f_case = os.path.normcase(forb_abs)
            if n_case == f_case or n_case.startswith(f_case + os.sep):
                raise ValueError(i18n.t("server.cacheDirForbiddenCodePath", path=norm_new))
        if os.path.normcase(norm_new) == os.path.normcase(os.path.abspath(ROOT)):
            norm_new = ""

    if norm_new:
        try:
            for sub in ("users", "run", "config", "infer", os.path.join("cache", "jpg")):
                os.makedirs(os.path.join(norm_new, sub), exist_ok=True)
            probe = os.path.join(norm_new, f".wc_write_probe_{os.getpid()}")
            with open(probe, "w", encoding="utf-8") as fh:
                fh.write("ok")
            os.remove(probe)
        except OSError as exc:
            raise ValueError(i18n.t("server.cacheDirNotWritable", path=norm_new, err=str(exc)))

    migrated_items = 0
    if migrate_existing:
        if has_active_jobs:
            raise ValueError(i18n.t("server.cacheDirMigrateBusy"))
        target_users = os.path.join(norm_new, "users") if norm_new else DEFAULT_USERS_STORAGE
        target_run = os.path.join(norm_new, "run") if norm_new else DEFAULT_RUN_ROOT
        target_cfg = os.path.join(norm_new, "config") if norm_new else DEFAULT_CONFIG_ROOT
        target_cache = os.path.join(norm_new, "cache") if norm_new else DEFAULT_CACHE_DIR
        target_infer = os.path.join(norm_new, "infer") if norm_new else DEFAULT_INFER_DIR

        migrated_items += _merge_directory_contents(old_users, target_users)
        migrated_items += _merge_directory_contents(
            old_run, target_run, skip_names={"verification_report"}
        )
        migrated_items += _merge_directory_contents(
            old_cfg, target_cfg, skip_names={"default.json"}
        )
        migrated_items += _merge_directory_contents(old_cache, target_cache)
        migrated_items += _merge_directory_contents(old_infer, target_infer)

    db.set_setting("data_cache_root", norm_new)
    refresh_storage_paths(force_db=True)
    stats = get_storage_disk_stats()
    stats["migrated_items"] = migrated_items
    return stats
