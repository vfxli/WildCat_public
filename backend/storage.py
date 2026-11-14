# -*- coding: utf-8 -*-
"""User-Isolated Storage Manager for Assets (Sequences) and Task Run Directories."""
import glob
import io
import os
from pathlib import Path, PureWindowsPath
import re
import shutil
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
DATASET_ROOT = os.path.join(ROOT, "dataset")

USERS_STORAGE = DEFAULT_USERS_STORAGE
RUN_ROOT = DEFAULT_RUN_ROOT
CONFIG_ROOT = DEFAULT_CONFIG_ROOT
CACHE_DIR = DEFAULT_CACHE_DIR

_PATH_CACHE = {"loaded": False, "custom_root": ""}


def _normalize_custom_root(raw_path: str) -> str:
    s = str(raw_path or "").strip()
    if not s:
        return ""
    expanded = os.path.expanduser(s)
    return os.path.abspath(expanded)


def refresh_storage_paths(force_db: bool = True) -> str:
    """Reload `data_cache_root` from SQLite settings and update active storage root globals."""
    global USERS_STORAGE, RUN_ROOT, CONFIG_ROOT, CACHE_DIR
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
    else:
        USERS_STORAGE = DEFAULT_USERS_STORAGE
        RUN_ROOT = DEFAULT_RUN_ROOT
        CONFIG_ROOT = DEFAULT_CONFIG_ROOT
        CACHE_DIR = DEFAULT_CACHE_DIR
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


def get_jpg_cache_dir() -> str:
    d = os.path.join(get_cache_dir(), "jpg")
    os.makedirs(d, exist_ok=True)
    return d


def apply_subprocess_storage_env(env: dict) -> dict:
    """Populate WC_* storage root env vars for child training / evaluation / export processes."""
    env["WC_RUN_ROOT"] = get_run_root()
    env["WC_USERS_STORAGE"] = get_users_storage()
    env["WC_CONFIG_ROOT"] = get_config_root()
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

    return str(Path(ROOT) / p)


def _dedup_sorted(paths: list) -> list:
    seen = set()
    out = []
    for p in sorted(paths):
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


def _extract_upload_images(file_tuples: list, role_label: str) -> list:
    """Decode uploaded files (.exr/.png/.tif/.jpg/.zip) into sorted list of (stem_name, filename, float32_rgb_ndarray)
    with strict anti-DoS caps on archive entries, decompressed bytes, and image dimensions."""
    MAX_ROLE_FILES = 300
    MAX_SINGLE_BYTES = 128 * 1024 * 1024  # 128 MB
    MAX_TOTAL_BYTES = 1024 * 1024 * 1024  # 1 GB
    MAX_IMG_DIM = 8192

    raw_items = []
    total_uncompressed = 0
    valid_exts = (".exr", ".png", ".tif", ".tiff", ".jpg", ".jpeg")
    for fname, content in file_tuples:
        lower = (fname or "").lower()
        if lower.endswith(".zip"):
            if len(content) > MAX_TOTAL_BYTES:
                raise ValueError(i18n.t("server.uploadTooLarge", role=role_label))
            with zipfile.ZipFile(io.BytesIO(content), "r") as z:
                infos = [zi for zi in sorted(z.infolist(), key=lambda x: x.filename) if not zi.is_dir()]
                if len(infos) > MAX_ROLE_FILES * 2:
                    raise ValueError(i18n.t("server.uploadTooManyFiles", role=role_label, limit=MAX_ROLE_FILES))
                for zi in infos:
                    bn = PureWindowsPath(zi.filename).name
                    if bn.startswith(".") or "__MACOSX" in zi.filename:
                        continue
                    ext = PureWindowsPath(bn).suffix.lower()
                    if ext in valid_exts:
                        if zi.file_size > MAX_SINGLE_BYTES:
                            raise ValueError(i18n.t("server.uploadTooLarge", role=role_label))
                        total_uncompressed += int(zi.file_size)
                        if total_uncompressed > MAX_TOTAL_BYTES:
                            raise ValueError(i18n.t("server.uploadTooLarge", role=role_label))
                        if len(raw_items) >= MAX_ROLE_FILES:
                            raise ValueError(i18n.t("server.uploadTooManyFiles", role=role_label, limit=MAX_ROLE_FILES))
                        data = z.read(zi.filename)
                        if len(data) > MAX_SINGLE_BYTES:
                            raise ValueError(i18n.t("server.uploadTooLarge", role=role_label))
                        raw_items.append((bn, data))
        else:
            bn = PureWindowsPath(fname or "").name
            ext = PureWindowsPath(bn).suffix.lower()
            if ext in valid_exts:
                if len(content) > MAX_SINGLE_BYTES:
                    raise ValueError(i18n.t("server.uploadTooLarge", role=role_label))
                total_uncompressed += len(content)
                if total_uncompressed > MAX_TOTAL_BYTES:
                    raise ValueError(i18n.t("server.uploadTooLarge", role=role_label))
                if len(raw_items) >= MAX_ROLE_FILES:
                    raise ValueError(i18n.t("server.uploadTooManyFiles", role=role_label, limit=MAX_ROLE_FILES))
                raw_items.append((bn, content))

    if not raw_items:
        raise ValueError(i18n.t("server.uploadNoValidImages", role=role_label))

    raw_items.sort(key=lambda x: x[0])
    decoded = []
    for idx, (bn, data_bytes) in enumerate(raw_items, start=1):
        try:
            img = color.decode_image_bytes(data_bytes, bn)
        except Exception as e:
            raise ValueError(i18n.t("server.uploadDecodeFailed", role=role_label, filename=bn, err=e))
        ih, iw = int(img.shape[0]), int(img.shape[1])
        if ih <= 0 or iw <= 0 or ih > MAX_IMG_DIM or iw > MAX_IMG_DIM:
            raise ValueError(i18n.t("server.uploadResolutionExceeded", filename=bn, w=iw, h=ih, max_dim=MAX_IMG_DIM))
        stem = sanitize_name(PureWindowsPath(bn).stem, f"frame_{idx:04d}")
        decoded.append((stem, bn, img))
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


def _resolve_upload_colorspace(filename: str, input_colorspace: str, working_colorspace: str) -> str:
    ext = PureWindowsPath(filename or "").suffix.lower()
    if not input_colorspace or input_colorspace == "auto":
        return working_colorspace if ext == ".exr" else "sRGB"
    return input_colorspace


def _has_varying_alpha(imgs: list) -> bool:
    for im in imgs:
        if im is not None and im.ndim == 3 and im.shape[2] >= 4:
            a = im[..., 3]
            if float(np.min(a)) < 0.999 or float(np.max(a)) > 1.001:
                return True
    return False


def _save_eval_frames_to_dir(
    eval_abs: str,
    seq_tuples: list,
    expected_w: int = 0,
    expected_h: int = 0,
    expected_in_ch: int = 0,
    input_colorspace: str = "auto",
    working_colorspace: str = "ACEScg",
) -> int:
    seq_imgs = _extract_upload_images(seq_tuples, i18n.t("server.roleEvalSeq"))
    if len(seq_imgs) < 2:
        raise ValueError(i18n.t("server.evalSeqMinFrames", count=len(seq_imgs)))

    h0 = expected_h or int(seq_imgs[0][2].shape[0])
    w0 = expected_w or int(seq_imgs[0][2].shape[1])
    for idx, (_, s_bn, s_im) in enumerate(seq_imgs, start=1):
        sh, sw = int(s_im.shape[0]), int(s_im.shape[1])
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

    if os.path.exists(eval_abs):
        shutil.rmtree(eval_abs, ignore_errors=True)
    os.makedirs(eval_abs, exist_ok=True)

    for idx, (_, s_bn, s_im) in enumerate(seq_imgs, start=1):
        s_out = s_im
        if expected_in_ch in (1, 3, 4) and s_out.ndim == 3 and s_out.shape[2] > expected_in_ch:
            s_out = s_out[..., :expected_in_ch]
        s_cs = _resolve_upload_colorspace(s_bn, input_colorspace, working_colorspace)
        if s_cs not in ("", "Raw", working_colorspace) and working_colorspace != "Raw" and s_out.shape[2] >= 3:
            s_out = color.transform_colorspace(s_out, s_cs, working_colorspace)
        color.write_exr_file(os.path.join(eval_abs, f"frame_{idx:04d}.exr"), s_out, bit_depth="half")
    return len(seq_imgs)


def validate_and_save_paired_dataset(
    username: str,
    dataset_name: str,
    source_tuples: list,
    target_tuples: list,
    max_error_pct: float = 35.0,
    strategy: str = "standard",
    enable_check: bool = True,
    input_colorspace: str = "auto",
    working_colorspace: str = "ACEScg",
    seq_tuples: list | None = None,
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
    src_imgs = _extract_upload_images(source_tuples, i18n.t("server.roleSource"))
    tgt_imgs = _extract_upload_images(target_tuples, i18n.t("server.roleTarget"))

    src_fnums = [_extract_frame_num(x[1]) for x in src_imgs]
    tgt_fnums = [_extract_frame_num(x[1]) for x in tgt_imgs]
    has_fnums = all(n is not None for n in src_fnums) and all(n is not None for n in tgt_fnums)

    # 1. Missing Frame / Unequal Frame Count Detection (with exact missing frame number report!)
    if len(src_imgs) != len(tgt_imgs):
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
            i18n.t("server.pairCountMismatch", src_count=len(src_imgs), tgt_count=len(tgt_imgs), detail=detail)
        )

    # 2. Frame Number Alignment Check (when enable_check is ON)
    if enable_check and has_fnums:
        for idx, (sn, tn, (_, s_bn, _), (_, t_bn, _)) in enumerate(
            zip(src_fnums, tgt_fnums, src_imgs, tgt_imgs)
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

    h0, w0 = int(src_imgs[0][2].shape[0]), int(src_imgs[0][2].shape[1])
    in_ch = min(1 if im.ndim == 2 else int(im.shape[2]) for _, _, im in src_imgs)
    out_ch = min(1 if im.ndim == 2 else int(im.shape[2]) for _, _, im in tgt_imgs)
    if in_ch >= 3 and out_ch >= 3 and in_ch != out_ch:
        if in_ch == 4 and not _has_varying_alpha([im for _, _, im in src_imgs]):
            in_ch = 3
        if out_ch == 4 and not _has_varying_alpha([im for _, _, im in tgt_imgs]):
            out_ch = 3

    maes = []
    worst_mae = -1.0
    worst_idx = 0
    worst_pair_names = ("", "")
    thumbs_s = []
    thumbs_t = []

    for idx, ((s_stem, s_bn, s_im), (t_stem, t_bn, t_im)) in enumerate(zip(src_imgs, tgt_imgs)):
        sh, sw = int(s_im.shape[0]), int(s_im.shape[1])
        th, tw = int(t_im.shape[0]), int(t_im.shape[1])
        if sh != th or sw != tw:
            raise ValueError(
                i18n.t(
                    "server.pairSizeMismatch",
                    index=idx + 1,
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
                i18n.t("server.seqInternalSizeMismatch", w0=w0, h0=h0, index=idx + 1, sw=sw, sh=sh)
            )

        s_f = _norm_img_float(s_im)
        t_f = _norm_img_float(t_im)
        c_cmp = min(s_f.shape[2], t_f.shape[2], 3)
        frame_mae = float(np.mean(np.abs(s_f[..., :c_cmp] - t_f[..., :c_cmp])))
        maes.append(frame_mae)
        if frame_mae > worst_mae:
            worst_mae = frame_mae
            worst_idx = idx
            worst_pair_names = (s_bn, t_bn)

        if enable_check and len(src_imgs) > 1 and in_ch == out_ch:
            thumbs_s.append(cv2.resize(s_f[..., :c_cmp], (64, 64), interpolation=cv2.INTER_AREA))
            thumbs_t.append(cv2.resize(t_f[..., :c_cmp], (64, 64), interpolation=cv2.INTER_AREA))

    avg_mae = float(np.mean(maes)) if maes else 0.0

    if enable_check:
        # 3. Reject 100% identical duplicate sequences (only when channels match)
        if in_ch == out_ch and worst_mae < 1e-5:
            raise ValueError(i18n.t("server.pairDuplicateSeq"))

        # 4. Cross-Frame Content Shift Detection (detects if frame i is misaligned and actually matches frame j != i)
        if len(thumbs_s) > 1 and strategy != "alpha_matting":
            for i in range(len(thumbs_s)):
                self_diff = float(np.mean(np.abs(thumbs_s[i] - thumbs_t[i])))
                if self_diff > 0.02:
                    for j in range(len(thumbs_t)):
                        if i == j:
                            continue
                        cross_diff = float(np.mean(np.abs(thumbs_s[i] - thumbs_t[j])))
                        if cross_diff * 1.6 < self_diff and (self_diff - cross_diff) > 0.012:
                            s_name = src_imgs[i][1]
                            t_i_name = tgt_imgs[i][1]
                            t_j_name = tgt_imgs[j][1]
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

    for idx, ((_, s_bn, s_im), (_, t_bn, t_im)) in enumerate(zip(src_imgs, tgt_imgs), start=1):
        s_out = s_im[..., :norm_in_ch] if (s_im.ndim == 3 and s_im.shape[2] > norm_in_ch) else s_im
        t_out = t_im[..., :norm_out_ch] if (t_im.ndim == 3 and t_im.shape[2] > norm_out_ch) else t_im
        s_cs = _resolve_upload_colorspace(s_bn, input_colorspace, working_colorspace)
        t_cs = _resolve_upload_colorspace(t_bn, input_colorspace, working_colorspace)
        if s_cs not in ("", "Raw", working_colorspace) and working_colorspace != "Raw" and s_out.shape[2] >= 3:
            s_out = color.transform_colorspace(s_out, s_cs, working_colorspace)
        if t_cs not in ("", "Raw", working_colorspace) and working_colorspace != "Raw" and t_out.shape[2] >= 3:
            t_out = color.transform_colorspace(t_out, t_cs, working_colorspace)
        color.write_exr_file(os.path.join(src_abs, f"frame_{idx:04d}.exr"), s_out, bit_depth="half")
        color.write_exr_file(os.path.join(tgt_abs, f"frame_{idx:04d}.exr"), t_out, bit_depth="half")

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
                input_colorspace=input_colorspace,
                working_colorspace=working_colorspace,
            )
            seq_rel = rel_to_root(eval_abs)
        except Exception:
            shutil.rmtree(ds_abs, ignore_errors=True)
            raise

    ds_rel = rel_to_root(ds_abs)
    src_rel = rel_to_root(src_abs)
    tgt_rel = rel_to_root(tgt_abs)
    tot_bytes = dir_total_bytes(ds_rel)

    return {
        "rel_path": ds_rel,
        "source_path": src_rel,
        "target_path": tgt_rel,
        "seq_path": seq_rel,
        "frame_count": len(src_imgs),
        "seq_frame_count": seq_count,
        "width": w0,
        "height": h0,
        "in_channels": norm_in_ch,
        "out_channels": norm_out_ch,
        "avg_mae": round(avg_mae, 6),
        "max_mae": round(worst_mae, 6),
        "strategy": strategy if enable_check else "off",
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
    input_colorspace: str = "auto",
    working_colorspace: str = "ACEScg",
) -> dict:
    """Attach or replace an optional full validation sequence (`eval/`) on an existing dataset."""
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
        input_colorspace=input_colorspace,
        working_colorspace=working_colorspace,
    )
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
        s0 = color.read_image_float(s_files[0], input_colorspace="auto", working_colorspace="ACEScg")
        t0 = color.read_image_float(t_files[0], input_colorspace="auto", working_colorspace="ACEScg")
    except Exception:
        s0, t0 = None, None
    h, w = (int(s0.shape[0]), int(s0.shape[1])) if s0 is not None else (0, 0)
    in_ch = (1 if s0.ndim == 2 else int(s0.shape[2])) if s0 is not None else 3
    out_ch = (1 if t0.ndim == 2 else int(t0.shape[2])) if t0 is not None else 3
    if in_ch >= 3 and out_ch >= 3 and in_ch != out_ch:
        if in_ch == 4 and not _has_varying_alpha([s0]):
            in_ch = 3
        if out_ch == 4 and not _has_varying_alpha([t0]):
            out_ch = 3

    maes = []
    for sp, tp in zip(s_files[:n], t_files[:n]):
        try:
            si = color.read_image_float(sp, input_colorspace="auto", working_colorspace="ACEScg")
            ti = color.read_image_float(tp, input_colorspace="auto", working_colorspace="ACEScg")
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
    """Safely delete task run directory under run/ and config file under config/."""
    if run_dir_rel:
        abs_run = os.path.abspath(abs_from_root(run_dir_rel))
        for base in {os.path.abspath(get_run_root()), os.path.abspath(DEFAULT_RUN_ROOT)}:
            if (
                os.path.normcase(abs_run).startswith(os.path.normcase(base + os.sep))
                and os.path.basename(abs_run) != "verification_report"
                and os.path.exists(abs_run)
            ):
                shutil.rmtree(abs_run, ignore_errors=True)
                break
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
            "db_path": db.DB_PATH,
        },
        "disk": {
            "total_gb": total_gb,
            "used_gb": used_gb,
            "free_gb": free_gb,
            "users_mb": round(_fast_dir_bytes(users_dir) / (1024**2), 1),
            "run_mb": round(_fast_dir_bytes(run_dir) / (1024**2), 1),
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
            for sub in ("users", "run", "config", os.path.join("cache", "jpg")):
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

        migrated_items += _merge_directory_contents(old_users, target_users)
        migrated_items += _merge_directory_contents(
            old_run, target_run, skip_names={"verification_report"}
        )
        migrated_items += _merge_directory_contents(
            old_cfg, target_cfg, skip_names={"default.json"}
        )
        migrated_items += _merge_directory_contents(old_cache, target_cache)

    db.set_setting("data_cache_root", norm_new)
    refresh_storage_paths(force_db=True)
    stats = get_storage_disk_stats()
    stats["migrated_items"] = migrated_items
    return stats
