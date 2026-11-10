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

from . import db

ROOT = db.ROOT
STORAGE_ROOT = db.STORAGE_ROOT
USERS_STORAGE = os.path.join(STORAGE_ROOT, "users")
RUN_ROOT = os.path.join(ROOT, "run")
CONFIG_ROOT = os.path.join(ROOT, "config")
DATASET_ROOT = os.path.join(ROOT, "dataset")


def sanitize_name(name: str, default="item") -> str:
    s = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", (name or "").strip())
    return s.strip("._") or default


def user_root(username: str) -> str:
    p = os.path.join(USERS_STORAGE, sanitize_name(username, "user"))
    os.makedirs(os.path.join(p, "assets"), exist_ok=True)
    return p


def user_asset_dir(username: str, asset_name: str) -> str:
    return os.path.join(user_root(username), "assets", sanitize_name(asset_name, "asset"))


def rel_to_root(abs_path: str) -> str:
    try:
        return Path(os.path.relpath(os.path.abspath(abs_path), ROOT)).as_posix()
    except ValueError:
        return Path(abs_path).as_posix()


def abs_from_root(path_str: str) -> str:
    if not path_str:
        return ROOT
    p = Path(PureWindowsPath(path_str).as_posix())
    return str(p if p.is_absolute() else (Path(ROOT) / p))


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
    """Decode uploaded files (.exr/.png/.tif/.jpg/.zip) into sorted list of (stem_name, filename, float32_rgb_ndarray)."""
    raw_items = []
    valid_exts = (".exr", ".png", ".tif", ".tiff", ".jpg", ".jpeg")
    for fname, content in file_tuples:
        lower = (fname or "").lower()
        if lower.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(content), "r") as z:
                for zi in sorted(z.infolist(), key=lambda x: x.filename):
                    if zi.is_dir():
                        continue
                    bn = PureWindowsPath(zi.filename).name
                    if bn.startswith(".") or "__MACOSX" in zi.filename:
                        continue
                    ext = PureWindowsPath(bn).suffix.lower()
                    if ext in valid_exts:
                        raw_items.append((bn, z.read(zi.filename)))
        else:
            bn = PureWindowsPath(fname or "").name
            ext = PureWindowsPath(bn).suffix.lower()
            if ext in valid_exts:
                raw_items.append((bn, content))

    if not raw_items:
        raise ValueError(f"{role_label}未检测到有效的图像文件 (.exr / .png / .tif / .jpg / .zip)！")

    raw_items.sort(key=lambda x: x[0])
    decoded = []
    for idx, (bn, data_bytes) in enumerate(raw_items, start=1):
        try:
            img = color.decode_image_bytes(data_bytes, bn)
        except Exception as e:
            raise ValueError(f"{role_label}中的文件 '{bn}' 无法解码为有效图像 ({e})！")
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


def validate_and_save_paired_dataset(
    username: str,
    dataset_name: str,
    source_tuples: list,
    target_tuples: list,
    max_error_pct: float = 35.0,
    strategy: str = "standard",
    enable_check: bool = True,
    input_colorspace: str = "ACEScg",
    working_colorspace: str = "ACEScg",
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
    src_imgs = _extract_upload_images(source_tuples, "输入原片序列 (Source)")
    tgt_imgs = _extract_upload_images(target_tuples, "目标效果图序列 (Target)")

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
                parts.append(f"目标序列缺失帧号 {miss_in_tgt[:10]}")
            if miss_in_src:
                parts.append(f"输入序列缺失帧号 {miss_in_src[:10]}")
            if parts:
                detail = " — 检测到缺帧：" + "；".join(parts)
        raise ValueError(
            f"配对校验未通过（缺帧/数量不一致）：输入原片序列共 {len(src_imgs)} 帧，目标效果图序列共 {len(tgt_imgs)} 帧{detail}！"
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
                    extra = f"（目标缺帧 {miss_in_tgt[:8]}，输入缺帧 {miss_in_src[:8]}）"
                raise ValueError(
                    f"配对校验未通过（帧号未对齐）：第 {idx + 1} 对文件帧号不一致 — 输入为 '{s_bn}' (帧号 {sn})，目标为 '{t_bn}' (帧号 {tn}){extra}！如需忽略文件名帧号可关闭上方检测开关。"
                )

    h0, w0 = int(src_imgs[0][2].shape[0]), int(src_imgs[0][2].shape[1])
    in_ch = min(1 if im.ndim == 2 else int(im.shape[2]) for _, _, im in src_imgs)
    out_ch = min(1 if im.ndim == 2 else int(im.shape[2]) for _, _, im in tgt_imgs)
    if in_ch >= 3 and out_ch >= 3 and in_ch != out_ch:
        in_ch = min(in_ch, out_ch)
        out_ch = in_ch

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
                f"配对校验未通过（尺寸不一致）：第 {idx + 1} 帧分辨率不匹配 — 输入 '{s_bn}' 为 {sw}×{sh}，目标 '{t_bn}' 为 {tw}×{th}！"
            )
        if sh != h0 or sw != w0:
            raise ValueError(
                f"配对校验未通过（序列内尺寸不统一）：第 1 帧为 {w0}×{h0}，而第 {idx + 1} 帧为 {sw}×{sh}！"
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
            raise ValueError(
                "配对校验未通过（素材完全重复）：输入原片序列与目标效果图序列像素 100% 完全相同（误差为 0.00%），请检查是否把同一份序列上传了两次！"
            )

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
                                f"配对校验未通过（检测到画面错帧/未对齐）：第 {i + 1} 帧输入 '{s_name}' 与第 {i + 1} 帧目标 '{t_i_name}' 差异较大 ({self_diff * 100:.1f}%)，"
                                f"反而与第 {j + 1} 帧目标 '{t_j_name}' 高度吻合 (差异仅 {cross_diff * 100:.1f}%)！请检查序列是否漏帧或顺序错位。"
                            )

        # 5. Generous Pixel Error Threshold Check
        if strategy != "alpha_matting" and max_error_pct > 0:
            limit = float(max_error_pct) / 100.0
            if worst_mae > limit:
                raise ValueError(
                    f"配对校验未通过（像素差异超限）：第 {worst_idx + 1} 帧 ({worst_pair_names[0]} ↔ {worst_pair_names[1]}) "
                    f"全图平均像素差达 {worst_mae * 100:.2f}%（全序列平均差 {avg_mae * 100:.2f}%），"
                    f"超过当前设定的容差上限 {max_error_pct:.1f}%！疑似非同一镜头素材。若确需使用大差异素材，可调高容差上限或关闭上方配对检测开关。"
                )

    # Validation passed -> write clean paired dataset directory in .exr (ACEScg 16-bit half-float)
    ds_abs = user_dataset_dir(username, dataset_name)
    src_abs = os.path.join(ds_abs, "source")
    tgt_abs = os.path.join(ds_abs, "target")
    if os.path.exists(ds_abs):
        shutil.rmtree(ds_abs, ignore_errors=True)
    os.makedirs(src_abs, exist_ok=True)
    os.makedirs(tgt_abs, exist_ok=True)

    for idx, ((_, s_bn, s_im), (_, t_bn, t_im)) in enumerate(zip(src_imgs, tgt_imgs), start=1):
        s_out = s_im
        t_out = t_im
        if input_colorspace and input_colorspace not in ("auto", "Raw", working_colorspace):
            if s_out.shape[2] >= 3:
                s_out = color.transform_colorspace(s_out, input_colorspace, working_colorspace)
            if t_out.shape[2] >= 3:
                t_out = color.transform_colorspace(t_out, input_colorspace, working_colorspace)
        color.write_exr_file(os.path.join(src_abs, f"frame_{idx:04d}.exr"), s_out, bit_depth="half")
        color.write_exr_file(os.path.join(tgt_abs, f"frame_{idx:04d}.exr"), t_out, bit_depth="half")

    ds_rel = rel_to_root(ds_abs)
    src_rel = rel_to_root(src_abs)
    tgt_rel = rel_to_root(tgt_abs)
    tot_bytes = dir_total_bytes(ds_rel)

    return {
        "rel_path": ds_rel,
        "source_path": src_rel,
        "target_path": tgt_rel,
        "frame_count": len(src_imgs),
        "width": w0,
        "height": h0,
        "in_channels": 3 if in_ch == 3 else (4 if in_ch == 4 else 1),
        "out_channels": 3 if out_ch == 3 else (4 if out_ch == 4 else 1),
        "avg_mae": round(avg_mae, 6),
        "max_mae": round(worst_mae, 6),
        "strategy": strategy if enable_check else "off",
        "size_bytes": tot_bytes,
    }


def inspect_paired_dirs(source_rel: str, target_rel: str) -> dict:
    """Compute paired stats (frame_count, width, height, channels, avg_mae, max_mae, size_bytes) for existing directories."""
    s_files = expand_sequence_files(source_rel)
    t_files = expand_sequence_files(target_rel)
    n = min(len(s_files), len(t_files))
    if n == 0:
        return {
            "frame_count": 0,
            "width": 0,
            "height": 0,
            "in_channels": 3,
            "out_channels": 3,
            "avg_mae": 0.0,
            "max_mae": 0.0,
            "size_bytes": 0,
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
        in_ch = min(in_ch, out_ch)
        out_ch = in_ch

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
        "width": w,
        "height": h,
        "in_channels": in_ch,
        "out_channels": out_ch,
        "avg_mae": round(avg_m, 6),
        "max_mae": round(max_m, 6),
        "size_bytes": sb + tb,
    }


def save_uploaded_files_to_asset(
    username: str, asset_name: str, file_tuples: list
) -> tuple:
    """
    Save uploaded files [(filename, bytes), ...] into user's asset directory.
    Supports individual .png/.jpg/.jpeg images or .zip archives containing images.
    Converts .jpg/.jpeg to .png automatically so Core Compute Engine (`wildcat`)
    can load 100% standard `.png` sequences.
    Returns (rel_path, frame_count, size_bytes).
    """
    target_dir = user_asset_dir(username, asset_name)
    os.makedirs(target_dir, exist_ok=True)

    raw_items = []
    for fname, content in file_tuples:
        lower = (fname or "").lower()
        if lower.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(content), "r") as z:
                for zi in sorted(z.infolist(), key=lambda x: x.filename):
                    if zi.is_dir():
                        continue
                    bn = PureWindowsPath(zi.filename).name
                    if bn.startswith(".") or "__MACOSX" in zi.filename:
                        continue
                    ext = PureWindowsPath(bn).suffix.lower()
                    if ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr"):
                        raw_items.append((bn, z.read(zi.filename)))
        else:
            bn = PureWindowsPath(fname or "").name
            ext = PureWindowsPath(bn).suffix.lower()
            if ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr"):
                raw_items.append((bn, content))

    if not raw_items:
        raise ValueError("No valid image files (.png/.jpg/.zip) found in upload.")

    saved_count = 0
    for idx, (bn, data_bytes) in enumerate(raw_items, start=1):
        ext = PureWindowsPath(bn).suffix.lower()
        stem = sanitize_name(PureWindowsPath(bn).stem, f"frame_{idx:04d}")
        out_png = os.path.join(target_dir, f"{stem}.png")
        if ext == ".png":
            with open(out_png, "wb") as f:
                f.write(data_bytes)
            saved_count += 1
        else:
            arr = np.frombuffer(data_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            if img is not None:
                color.imwrite_unicode(out_png, img)
                saved_count += 1

    rel_p = rel_to_root(target_dir)
    n_frames, n_bytes = dir_stats(rel_p)
    return rel_p, n_frames, n_bytes


def delete_asset_files(rel_path: str):
    """Safely delete asset directory if it resides inside STORAGE_ROOT/users."""
    abs_p = os.path.abspath(abs_from_root(rel_path))
    users_abs = os.path.abspath(USERS_STORAGE)
    if os.path.normcase(abs_p).startswith(os.path.normcase(users_abs + os.sep)) and os.path.exists(abs_p):
        shutil.rmtree(abs_p, ignore_errors=True)


def delete_task_files(run_dir_rel: str, config_path_rel: str):
    """Safely delete task run directory under run/ and config file under config/."""
    if run_dir_rel:
        abs_run = os.path.abspath(abs_from_root(run_dir_rel))
        run_root_abs = os.path.abspath(RUN_ROOT)
        if (
            os.path.normcase(abs_run).startswith(os.path.normcase(run_root_abs + os.sep))
            and os.path.basename(abs_run) != "verification_report"
            and os.path.exists(abs_run)
        ):
            shutil.rmtree(abs_run, ignore_errors=True)
    if config_path_rel:
        abs_cfg = os.path.abspath(abs_from_root(config_path_rel))
        cfg_root_abs = os.path.abspath(CONFIG_ROOT)
        if (
            os.path.normcase(abs_cfg).startswith(os.path.normcase(cfg_root_abs + os.sep))
            and os.path.basename(abs_cfg) != "default.json"
            and os.path.isfile(abs_cfg)
        ):
            try:
                os.remove(abs_cfg)
            except OSError:
                pass
