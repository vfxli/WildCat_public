# -*- coding: utf-8 -*-
"""
WildCat OCIO (OpenColorIO) Color Management & OpenEXR (.exr) Pipeline.

Default VFX Production Pipeline:
  - Working Colorspace: ACEScg (Linear AP1)
  - File Format: OpenEXR (.exr, 16-bit half-float ZIP compressed by default, preserving HDR values > 1.0)
  - Display / View Transform (for Web UI Viewer & Training Previews):
    OCIO ACES 1.3 Display/View (`sRGB - Display` / `ACES 1.0 - SDR Video`)
"""
import io
import os
import tempfile

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np

try:
    import PyOpenColorIO as OCIO

    HAS_OCIO = True
except Exception:
    OCIO = None
    HAS_OCIO = False

try:
    import Imath
    import OpenEXR

    HAS_OPENEXR = True
except Exception:
    OpenEXR = None
    Imath = None
    HAS_OPENEXR = False


# Built-in ACES 1.3 OCIO v2 URI (compiled inside PyOpenColorIO 2.2+)
DEFAULT_OCIO_CONFIG_URI = "ocio://cg-config-v1.0.0_aces-v1.3_ocio-v2.1"

# Canonical VFX Colorspace Names (compatible with ACES 1.2 / 1.3 OCIO configs)
COLORSPACE_ALIASES = {
    "ACEScg": ["ACEScg", "ACES - ACEScg", "lin_ap1"],
    "ACES2065-1": ["ACES2065-1", "ACES - ACES2065-1", "aces", "lin_ap0"],
    "ACEScct": ["ACEScct", "ACES - ACEScct"],
    "ACEScc": ["ACEScc", "ACES - ACEScc"],
    "sRGB": ["sRGB - Texture", "Output - sRGB", "Utility - sRGB - Texture", "srgb_texture", "sRGB"],
    "Rec.709": ["Rec.1886 Rec.709 - Display", "Output - Rec.709", "Utility - Rec.709 - Camera", "Rec.709"],
    "Linear sRGB": ["Linear Rec.709 (sRGB)", "Utility - Linear - sRGB", "lin_rec709", "lin_srgb"],
    "Raw": ["Raw", "Utility - Raw", "nc_raw"],
}

_OCIO_CFG_CACHE = {}
_CPU_PROC_CACHE = {}


def get_ocio_config(ocio_config_path: str = ""):
    """Load OCIO Config from explicit path, $OCIO env var, or built-in ACES 1.3 config."""
    if not HAS_OCIO:
        return None
    cfg_key = (ocio_config_path or os.environ.get("OCIO") or DEFAULT_OCIO_CONFIG_URI).strip()
    if cfg_key in _OCIO_CFG_CACHE:
        return _OCIO_CFG_CACHE[cfg_key]

    cfg = None
    try:
        if ocio_config_path and os.path.isfile(ocio_config_path):
            cfg = OCIO.Config.CreateFromFile(ocio_config_path)
        elif os.environ.get("OCIO") and os.path.isfile(os.environ["OCIO"]):
            cfg = OCIO.Config.CreateFromEnv()
        else:
            cfg = OCIO.Config.CreateFromBuiltinConfig(DEFAULT_OCIO_CONFIG_URI)
    except Exception:
        try:
            cfg = OCIO.Config.CreateFromBuiltinConfig("cg-config-v1.0.0_aces-v1.3_ocio-v2.1")
        except Exception:
            cfg = None

    if cfg is not None:
        _OCIO_CFG_CACHE[cfg_key] = cfg
    return cfg


def resolve_ocio_colorspace_name(cfg, name: str) -> str:
    """Resolve user-friendly colorspace name (e.g. 'ACEScg', 'sRGB') against active OCIO config."""
    if not name:
        return "ACEScg"
    if cfg is None:
        return name
    # Exact match first
    try:
        if cfg.getColorSpace(name) is not None:
            return name
    except Exception:
        pass
    # Check canonical aliases
    for canonical, candidates in COLORSPACE_ALIASES.items():
        if name.lower() == canonical.lower() or name in candidates:
            for cand in candidates:
                try:
                    if cfg.getColorSpace(cand) is not None:
                        return cand
                except Exception:
                    pass
    return name


def get_ocio_info(ocio_config_path: str = "") -> dict:
    """Return active OCIO configuration metadata and available colorspaces/views for Web UI."""
    cfg = get_ocio_config(ocio_config_path)
    source = ocio_config_path or os.environ.get("OCIO") or "Built-in ACES 1.3 (cg-config-v1.0.0_aces-v1.3_ocio-v2.1)"
    spaces = ["ACEScg", "ACES2065-1", "ACEScct", "sRGB", "Linear sRGB", "Rec.709", "Raw"]
    views = ["ACES 1.0 - SDR Video (sRGB)", "Un-tone-mapped (sRGB)", "Raw"]
    if cfg is not None:
        try:
            cs_names = list(cfg.getColorSpaceNames())
            for cs in cs_names:
                if cs not in spaces:
                    spaces.append(cs)
        except Exception:
            pass
    return {
        "has_ocio": HAS_OCIO,
        "ocio_version": getattr(OCIO, "__version__", "fallback") if HAS_OCIO else "analytical_aces_fallback",
        "has_openexr": HAS_OPENEXR,
        "config_source": source,
        "working_colorspace": "ACEScg",
        "default_file_format": "exr",
        "colorspaces": spaces,
        "views": views,
    }


def _get_cpu_processor(src_space: str, dst_space: str, ocio_config_path: str = ""):
    if not HAS_OCIO or src_space == dst_space or src_space == "Raw" or dst_space == "Raw":
        return None
    key = (ocio_config_path, src_space, dst_space)
    if key in _CPU_PROC_CACHE:
        return _CPU_PROC_CACHE[key]
    cfg = get_ocio_config(ocio_config_path)
    if cfg is None:
        return None
    try:
        s_real = resolve_ocio_colorspace_name(cfg, src_space)
        d_real = resolve_ocio_colorspace_name(cfg, dst_space)
        if s_real == d_real:
            return None
        proc = cfg.getProcessor(s_real, d_real).getDefaultCPUProcessor()
        _CPU_PROC_CACHE[key] = proc
        return proc
    except Exception:
        return None


def _build_display_view_proc_for_cfg(cfg, src_space: str, view_mode: str):
    if cfg is None:
        return None
    s_real = resolve_ocio_colorspace_name(cfg, src_space)
    displays = list(cfg.getDisplays()) if hasattr(cfg, "getDisplays") else []
    display = ""
    for cand_d in ("sRGB - Display", "sRGB", "ACES"):
        if cand_d in displays:
            display = cand_d
            break
    if not display:
        display = cfg.getDefaultDisplay() or (displays[0] if displays else "sRGB - Display")

    avail_views = list(cfg.getViews(display)) if hasattr(cfg, "getViews") else []
    want_untone = "un-tone-mapped" in (view_mode or "").lower() or view_mode in ("sRGB", "sRGB - Texture")
    view = ""
    if want_untone:
        for v in avail_views:
            if "un-tone-mapped" in v.lower() or v.lower() in ("raw", "srgb - texture"):
                if "raw" not in v.lower():
                    view = v
                    break
        if not view:
            return None
    else:
        for cand_v in ("ACES 1.0 - SDR Video", "sRGB", "Output - sRGB", "ACES 1.0 SDR-video"):
            for v in avail_views:
                if v == cand_v or cand_v.lower() in v.lower():
                    view = v
                    break
            if view:
                break
        if not view:
            view = cfg.getDefaultView(display) or (avail_views[0] if avail_views else "ACES 1.0 - SDR Video")
    return cfg.getProcessor(s_real, display, view, OCIO.TRANSFORM_DIR_FORWARD).getDefaultCPUProcessor()


def _get_display_view_processor(
    src_space: str = "ACEScg",
    view_mode: str = "ACES 1.0 - SDR Video",
    ocio_config_path: str = "",
):
    if not HAS_OCIO or view_mode == "Raw":
        return None
    key = ("DISPLAY_VIEW", ocio_config_path, src_space, view_mode)
    if key in _CPU_PROC_CACHE:
        return _CPU_PROC_CACHE[key]

    proc = None
    cfg = get_ocio_config(ocio_config_path)
    if cfg is not None:
        try:
            proc = _build_display_view_proc_for_cfg(cfg, src_space, view_mode)
        except Exception:
            proc = None
    if proc is None and not ocio_config_path:
        try:
            builtin_cfg = OCIO.Config.CreateFromBuiltinConfig(DEFAULT_OCIO_CONFIG_URI)
            proc = _build_display_view_proc_for_cfg(builtin_cfg, src_space, view_mode)
        except Exception:
            proc = None
    if proc is not None:
        _CPU_PROC_CACHE[key] = proc
    return proc


# Analytical ACEScg <-> sRGB fallback matrices & ACES Narkowicz/Hill RRT+ODT curve
# (Used only if PyOpenColorIO is unavailable or Raw/custom space fails)
_AP1_TO_SRGB_MAT = np.array(
    [
        [1.70505099, -0.62179212, -0.08325887],
        [-0.13025642, 1.14080473, -0.01054831],
        [-0.02400336, -0.12896898, 1.15297234],
    ],
    dtype=np.float32,
)
_SRGB_TO_AP1_MAT = np.linalg.inv(_AP1_TO_SRGB_MAT).astype(np.float32)


def transform_colorspace(
    img: np.ndarray,
    src_space: str = "ACEScg",
    dst_space: str = "ACEScg",
    ocio_config_path: str = "",
) -> np.ndarray:
    """
    Transform float32 image (H, W, C) in RGB(A) order from `src_space` to `dst_space` using OCIO.
    Note: Alpha channel (channel index 3) or 1-channel masks are never color-transformed (always linear/Raw).
    """
    arr = np.ascontiguousarray(img, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]
    if not src_space or not dst_space or src_space == dst_space or src_space == "Raw" or dst_space == "Raw":
        return arr
    if arr.shape[2] < 3:
        return arr

    proc = _get_cpu_processor(src_space, dst_space, ocio_config_path)
    if proc is not None:
        out = arr.copy()
        rgb = np.ascontiguousarray(out[..., :3], dtype=np.float32)
        h, w, _ = rgb.shape
        desc = OCIO.PackedImageDesc(rgb, w, h, 3)
        proc.apply(desc)
        out[..., :3] = rgb
        return out

    # Fallback analytical transform between sRGB and ACEScg
    out = arr.copy()
    rgb = out[..., :3]
    if src_space in ("sRGB", "sRGB - Texture") and dst_space == "ACEScg":
        lin = np.where(rgb <= 0.04045, rgb / 12.92, np.power(np.maximum(rgb, 0.0) + 0.055, 2.4) / (1.055 ** 2.4))
        out[..., :3] = np.einsum("hwc,dc->hwd", lin, _SRGB_TO_AP1_MAT)
    elif src_space == "ACEScg" and dst_space in ("sRGB", "sRGB - Texture"):
        lin = np.clip(np.einsum("hwc,dc->hwd", rgb, _AP1_TO_SRGB_MAT), 0.0, None)
        out[..., :3] = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1.0 / 2.4) - 0.055)
    return out


def _aces_rrt_odt_srgb_fit(rgb_ap1: np.ndarray) -> np.ndarray:
    """Analytical ACEScg -> ACES 1.0 SDR Video (sRGB) fallback when PyOpenColorIO is unavailable."""
    lin_srgb = np.maximum(np.einsum("hwc,dc->hwd", rgb_ap1, _AP1_TO_SRGB_MAT), 0.0)
    # Stephen Hill RRT+ODT fit (input/output matrices + rational curve)
    in_mat = np.array(
        [
            [0.59719, 0.35458, 0.04823],
            [0.07600, 0.90834, 0.01566],
            [0.02840, 0.13383, 0.83777],
        ],
        dtype=np.float32,
    )
    out_mat = np.array(
        [
            [1.60475, -0.53108, -0.07367],
            [-0.10208, 1.10813, -0.00605],
            [-0.00327, -0.07276, 1.07602],
        ],
        dtype=np.float32,
    )
    v = np.einsum("hwc,dc->hwd", lin_srgb, in_mat)
    a = v * (v + 0.0245786) - 0.000090537
    b = v * (0.983729 * v + 0.4329510) + 0.238081
    tonemapped = np.clip(np.einsum("hwc,dc->hwd", a / np.maximum(b, 1e-12), out_mat), 0.0, 1.0)
    return np.where(tonemapped <= 0.0031308, tonemapped * 12.92, 1.055 * np.power(tonemapped, 1.0 / 2.4) - 0.055)


def acescg_to_display_srgb(
    img: np.ndarray,
    src_space: str = "ACEScg",
    view_mode: str = "ACES 1.0 - SDR Video",
    ocio_config_path: str = "",
) -> np.ndarray:
    """
    Convert an image in `src_space` (default `ACEScg` float32 data) to `[0, 1]` display `sRGB`
    using OCIO `ACES 1.0 - SDR Video` Display/View (1:1 parity with Foundry Nuke's ACES Viewer).
    """
    arr = np.ascontiguousarray(img, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.shape[2] == 1:
        return np.clip(np.repeat(arr, 3, axis=2), 0.0, 1.0)

    rgb = np.ascontiguousarray(arr[..., :3].copy(), dtype=np.float32)
    if view_mode == "Raw":
        return np.clip(rgb, 0.0, 1.0)

    want_untone = "un-tone-mapped" in (view_mode or "").lower() or view_mode in ("sRGB", "sRGB - Texture")
    proc = None
    if not want_untone:
        proc = _get_display_view_processor(src_space, view_mode or "ACES 1.0 - SDR Video", ocio_config_path)
    else:
        proc = _get_display_view_processor(src_space, "Un-tone-mapped", ocio_config_path)
    if proc is None:
        proc = _get_cpu_processor(src_space, "sRGB", ocio_config_path)
    if proc is not None:
        h, w, _ = rgb.shape
        desc = OCIO.PackedImageDesc(rgb, w, h, 3)
        proc.apply(desc)
        return np.clip(rgb, 0.0, 1.0)

    if not want_untone:
        return np.clip(_aces_rrt_odt_srgb_fit(rgb), 0.0, 1.0).astype(np.float32)
    lin = np.clip(np.einsum("hwc,dc->hwd", rgb, _AP1_TO_SRGB_MAT), 0.0, None)
    srgb = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1.0 / 2.4) - 0.055)
    return np.clip(srgb, 0.0, 1.0).astype(np.float32)


# ─────────────────────── OpenEXR (.exr) & General Image I/O ───────────────────────


def imread_unicode(path: str, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray | None:
    """Cross-platform cv2.imread supporting non-ASCII (e.g. Chinese) file paths on Windows."""
    if os.name == "nt" and not path.isascii():
        try:
            raw = np.fromfile(path, dtype=np.uint8)
            return cv2.imdecode(raw, flags) if raw.size > 0 else None
        except Exception:
            return None
    im = cv2.imread(path, flags)
    if im is None and os.name == "nt" and os.path.isfile(path):
        try:
            raw = np.fromfile(path, dtype=np.uint8)
            return cv2.imdecode(raw, flags) if raw.size > 0 else None
        except Exception:
            return None
    return im


def imwrite_unicode(path: str, img: np.ndarray, params=None) -> bool:
    """Cross-platform cv2.imwrite supporting non-ASCII (e.g. Chinese) file paths on Windows."""
    if os.name == "nt" and not path.isascii():
        try:
            ext = os.path.splitext(path)[1] or ".png"
            ok, buf = cv2.imencode(ext, img, params or [])
            if ok:
                buf.tofile(path)
                return True
            return False
        except Exception:
            return False
    return bool(cv2.imwrite(path, img, params) if params is not None else cv2.imwrite(path, img))


def _pick_exr_channel_order(ch_names) -> list[str]:
    """Select exact (R, G, B[, A]) or 1-ch mask channel names from EXR headers, supporting Nuke layer prefixes."""
    names = list(ch_names)
    name_set = set(names)
    if all(c in name_set for c in ("R", "G", "B", "A")):
        return ["R", "G", "B", "A"]
    if all(c in name_set for c in ("R", "G", "B")):
        return ["R", "G", "B"]

    # Group channels by layer prefix (e.g. 'rgba.red', 'beauty.R', 'rgb.green', 'main.B')
    groups: dict[str, dict[str, str]] = {}
    role_map = {
        "r": "R",
        "red": "R",
        "g": "G",
        "green": "G",
        "b": "B",
        "blue": "B",
        "a": "A",
        "alpha": "A",
        "y": "Y",
        "luma": "Y",
    }
    for cn in names:
        if "." in cn:
            pfx, sfx = cn.rsplit(".", 1)
        else:
            pfx, sfx = "", cn
        role = role_map.get(sfx.lower())
        if role:
            groups.setdefault(pfx, {})[role] = cn

    pref_layers = ["", "rgba", "rgb", "beauty", "color", "default", "main"]
    ordered_layers = [lp for lp in pref_layers if lp in groups] + sorted(
        k for k in groups if k not in pref_layers
    )
    for lp in ordered_layers:
        g = groups[lp]
        if "R" in g and "G" in g and "B" in g:
            return [g["R"], g["G"], g["B"], g["A"]] if "A" in g else [g["R"], g["G"], g["B"]]
    for lp in ordered_layers:
        g = groups[lp]
        if "Y" in g:
            return [g["Y"]]
        if "A" in g:
            return [g["A"]]
    if "Y" in name_set:
        return ["Y"]
    if "A" in name_set:
        return ["A"]
    return sorted(names)[:4]


def _unpack_exr_box(box) -> tuple[int, int, int, int] | None:
    if box is None:
        return None
    if hasattr(box, "min") and hasattr(box, "max"):
        return int(box.min.x), int(box.min.y), int(box.max.x), int(box.max.y)
    if isinstance(box, (tuple, list)) and len(box) == 2:
        return int(box[0][0]), int(box[0][1]), int(box[1][0]), int(box[1][1])
    return None


def _align_dw_to_disp(arr: np.ndarray, dw_box, disp_box) -> np.ndarray:
    """Canvas-align an EXR `dataWindow` array onto its `displayWindow` (matching Nuke's format canvas)."""
    dw = _unpack_exr_box(dw_box)
    disp = _unpack_exr_box(disp_box)
    if dw is None or disp is None or dw == disp:
        return arr
    dw_x0, dw_y0, dw_x1, dw_y1 = dw
    dp_x0, dp_y0, dp_x1, dp_y1 = disp
    dp_w = dp_x1 - dp_x0 + 1
    dp_h = dp_y1 - dp_y0 + 1
    if dp_w <= 0 or dp_h <= 0 or dp_w > 32768 or dp_h > 32768:
        return arr
    c = 1 if arr.ndim == 2 else arr.shape[2]
    src = arr[..., None] if arr.ndim == 2 else arr
    canvas = np.zeros((dp_h, dp_w, c), dtype=np.float32)

    ix0 = max(dw_x0, dp_x0)
    iy0 = max(dw_y0, dp_y0)
    ix1 = min(dw_x1, dp_x1)
    iy1 = min(dw_y1, dp_y1)
    if ix0 <= ix1 and iy0 <= iy1:
        sx0, sy0 = ix0 - dw_x0, iy0 - dw_y0
        sx1, sy1 = sx0 + (ix1 - ix0 + 1), sy0 + (iy1 - iy0 + 1)
        dx0, dy0 = ix0 - dp_x0, iy0 - dp_y0
        dx1, dy1 = dx0 + (ix1 - ix0 + 1), dy0 + (iy1 - iy0 + 1)
        canvas[dy0:dy1, dx0:dx1, :] = src[sy0:sy1, sx0:sx1, :]
    return canvas[..., 0] if arr.ndim == 2 else canvas


def read_exr_file(path: str) -> np.ndarray:
    """Read an OpenEXR (.exr) file (single-part, multi-part, prefixed layers, or BBox) into float32 (H, W, C) RGB(A)."""
    if HAS_OPENEXR:
        # 1. Fast single-part scanline/tiled reader when primary RGB(A) is in part 0
        try:
            exr = OpenEXR.InputFile(path)
            hdr = exr.header()
            ch_map = hdr["channels"]
            picked = _pick_exr_channel_order(ch_map.keys())
            # If EXR is multi-part (e.g. Nuke interleave='channels' where part 0 only holds 'R' or 'A'), defer to OpenEXR.File
            is_split_multipart = len(picked) == 1 and picked[0] in ("R", "G", "B") and hasattr(OpenEXR, "File")
            if not is_split_multipart and picked:
                dw = hdr["dataWindow"]
                disp = hdr.get("displayWindow", dw)
                w = dw.max.x - dw.min.x + 1
                h = dw.max.y - dw.min.y + 1
                pt = Imath.PixelType(Imath.PixelType.FLOAT)
                chans = [np.frombuffer(exr.channel(c, pt), dtype=np.float32).reshape(h, w) for c in picked]
                exr.close()
                arr = np.stack(chans, axis=-1)
                arr = _align_dw_to_disp(arr, dw, disp)
                return np.nan_to_num(arr, nan=0.0, posinf=65504.0, neginf=-65504.0).astype(np.float32)
            exr.close()
        except Exception:
            pass

        # 2. Multi-part / deep-header OpenEXR 3.x reader (collects channels across parts)
        if hasattr(OpenEXR, "File"):
            try:
                with OpenEXR.File(path) as ef:
                    all_ch: dict[str, np.ndarray] = {}
                    ref_disp = None
                    for part in ef.parts:
                        phdr = part.header
                        pdw = phdr.get("dataWindow")
                        pdisp = phdr.get("displayWindow", pdw)
                        if ref_disp is None and pdisp is not None:
                            ref_disp = pdisp
                        for cname, ch_obj in part.channels.items():
                            if cname not in all_ch and hasattr(ch_obj, "pixels"):
                                px = np.array(ch_obj.pixels, dtype=np.float32, copy=True)
                                if px.ndim == 2:
                                    px = _align_dw_to_disp(px, pdw, ref_disp if ref_disp is not None else pdisp)
                                    all_ch[cname] = px
                    if all_ch:
                        picked = _pick_exr_channel_order(all_ch.keys())
                        shapes = {all_ch[c].shape for c in picked}
                        if len(shapes) == 1:
                            arr = np.stack([all_ch[c] for c in picked], axis=-1)
                            return np.nan_to_num(arr, nan=0.0, posinf=65504.0, neginf=-65504.0).astype(np.float32)
            except Exception:
                pass

    # Fallback to OpenCV EXR reader (which returns BGR(A) float32)
    im = imread_unicode(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise IOError(f"Cannot read EXR file: {path}")
    im = np.nan_to_num(im.astype(np.float32), nan=0.0, posinf=65504.0, neginf=-65504.0)
    if im.ndim == 2:
        return im[..., None]
    if im.shape[2] >= 3:
        im = im[..., [2, 1, 0, 3][: im.shape[2]]]
    return np.ascontiguousarray(im)


def write_exr_file(path: str, img_rgb: np.ndarray, bit_depth: str = "half") -> None:
    """
    Write a float32 numpy array (H, W, C) in RGB(A) order to an OpenEXR (.exr) file.
    Default `bit_depth='half'` (16-bit half-float ZIP compressed VFX pipeline standard).
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    use_half = bit_depth.lower() in ("half", "fp16", "float16", "16")
    inf_cap = 65504.0 if use_half else 1e20
    arr = np.nan_to_num(np.ascontiguousarray(img_rgb, dtype=np.float32), nan=0.0, posinf=inf_cap, neginf=-inf_cap)
    if arr.ndim == 2:
        arr = arr[..., None]
    h, w, c = arr.shape

    if HAS_OPENEXR:
        try:
            pt = Imath.PixelType(Imath.PixelType.HALF if use_half else Imath.PixelType.FLOAT)
            np_dt = np.float16 if use_half else np.float32

            hdr = OpenEXR.Header(w, h)
            hdr["compression"] = Imath.Compression(Imath.Compression.ZIP_COMPRESSION)
            if c >= 3 and hasattr(Imath, "Chromaticities") and hasattr(Imath, "V2f"):
                # Standard ACEScg (AP1 + D60) chromaticities matching Foundry Nuke ACEScg Write node
                hdr["chromaticities"] = Imath.Chromaticities(
                    Imath.V2f(0.713, 0.293),
                    Imath.V2f(0.165, 0.830),
                    Imath.V2f(0.128, 0.044),
                    Imath.V2f(0.32168, 0.33767),
                )

            if c == 1:
                ch_names = ["Y"]
            elif c == 3:
                ch_names = ["R", "G", "B"]
            else:
                ch_names = ["R", "G", "B", "A"][:c]

            hdr["channels"] = {name: Imath.Channel(pt) for name in ch_names}
            out = OpenEXR.OutputFile(path, hdr)
            ch_data = {name: np.ascontiguousarray(arr[..., i], dtype=np_dt).tobytes() for i, name in enumerate(ch_names)}
            out.writePixels(ch_data)
            out.close()
            return
        except Exception:
            if not os.name == "nt":
                raise

    # Fallback to OpenCV EXR writer (convert RGB(A) -> BGR(A)) with lossless ZIP compression
    bgr = arr[..., [2, 1, 0, 3][:c]] if c >= 3 else arr[..., 0]
    exr_params = []
    if hasattr(cv2, "IMWRITE_EXR_TYPE"):
        exr_params += [
            cv2.IMWRITE_EXR_TYPE,
            cv2.IMWRITE_EXR_TYPE_HALF if use_half else cv2.IMWRITE_EXR_TYPE_FLOAT,
        ]
    if hasattr(cv2, "IMWRITE_EXR_COMPRESSION") and hasattr(cv2, "IMWRITE_EXR_COMPRESSION_ZIP"):
        exr_params += [cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_ZIP]
    imwrite_unicode(path, bgr.astype(np.float32), exr_params or None)


def read_image_float(
    path: str,
    input_colorspace: str = "auto",
    working_colorspace: str = "ACEScg",
    ocio_config_path: str = "",
    is_data_pass: bool = False,
) -> np.ndarray:
    """
    Unified image reader for `.exr`, `.png`, `.jpg`, `.tif`.
    Returns `(H, W, C)` float32 in RGB(A) order converted to `working_colorspace` (default `ACEScg`).
      - `.exr` files default (`input_colorspace='auto'`) to `ACEScg` linear (full float HDR range preserved, no [0,1] clamp!).
      - `.png`/`.jpg`/`.tif` files default (`input_colorspace='auto'`) to `sRGB -> working_colorspace` (matching Nuke Read node; 1-ch masks & alpha stay linear).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".exr":
        im = read_exr_file(path)
        src_cs = "ACEScg" if input_colorspace in ("", "auto", None) else input_colorspace
        if not is_data_pass and im.shape[2] >= 3 and src_cs != working_colorspace and working_colorspace != "Raw":
            im = transform_colorspace(im, src_cs, working_colorspace, ocio_config_path)
        return im

    im = imread_unicode(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise IOError("cannot read " + path)
    if im.dtype == np.uint16:
        im = im.astype(np.float32) / 65535.0
    elif im.dtype == np.uint8:
        im = im.astype(np.float32) / 255.0
    else:
        im = im.astype(np.float32)
    if im.ndim == 2:
        im = im[..., None]
    if im.shape[2] >= 3:
        im = im[..., [2, 1, 0, 3][: im.shape[2]]]

    eff_cs = "sRGB" if input_colorspace in ("", "auto", None) else input_colorspace
    if (
        not is_data_pass
        and im.shape[2] >= 3
        and eff_cs != "Raw"
        and working_colorspace != "Raw"
        and eff_cs != working_colorspace
    ):
        im = transform_colorspace(im, eff_cs, working_colorspace, ocio_config_path)
    return np.ascontiguousarray(im, dtype=np.float32)


def decode_image_bytes(data_bytes: bytes, filename: str) -> np.ndarray:
    """Decode uploaded image bytes (`.exr`, `.png`, `.jpg`, `.tif`) into `(H, W, C)` float32 RGB(A) array."""
    ext = os.path.splitext(filename or "")[1].lower()
    if ext == ".exr":
        fd, tmp_name = tempfile.mkstemp(suffix=".exr")
        try:
            with os.fdopen(fd, "wb") as tmp:
                tmp.write(data_bytes)
            return read_exr_file(tmp_name)
        finally:
            if os.path.exists(tmp_name):
                try:
                    os.remove(tmp_name)
                except OSError:
                    pass

    arr = np.frombuffer(data_bytes, dtype=np.uint8)
    im = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise ValueError(f"Failed to decode image file: {filename}")
    if im.dtype == np.uint16:
        im = im.astype(np.float32) / 65535.0
    elif im.dtype == np.uint8:
        im = im.astype(np.float32) / 255.0
    else:
        im = im.astype(np.float32)
    if im.ndim == 2:
        im = im[..., None]
    if im.shape[2] >= 3:
        im = im[..., [2, 1, 0, 3][: im.shape[2]]]
    return np.ascontiguousarray(im, dtype=np.float32)
