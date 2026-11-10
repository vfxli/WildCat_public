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
    cfg = get_ocio_config(ocio_config_path)
    if cfg is None:
        return None
    try:
        s_real = resolve_ocio_colorspace_name(cfg, src_space)
        display = cfg.getDefaultDisplay() or "sRGB - Display"
        view = cfg.getDefaultView(display) or "ACES 1.0 - SDR Video"
        if "Un-tone-mapped" in view_mode:
            for v in cfg.getViews(display):
                if "Un-tone-mapped" in v:
                    view = v
                    break
        proc = cfg.getProcessor(s_real, display, view, OCIO.TRANSFORM_DIR_FORWARD).getDefaultCPUProcessor()
        _CPU_PROC_CACHE[key] = proc
        return proc
    except Exception:
        return None


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


def acescg_to_display_srgb(
    img: np.ndarray,
    src_space: str = "ACEScg",
    view_mode: str = "sRGB",
    ocio_config_path: str = "",
) -> np.ndarray:
    """
    Convert an image in `src_space` (default `ACEScg` float32 data) to `[0, 1]` `sRGB`
    using OCIO (for Web UI Preview Window, Thumbnails, and Training Previews).
    """
    arr = np.ascontiguousarray(img, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.shape[2] == 1:
        return np.clip(np.repeat(arr, 3, axis=2), 0.0, 1.0)

    rgb = np.ascontiguousarray(arr[..., :3].copy(), dtype=np.float32)
    if view_mode == "Raw":
        return np.clip(rgb, 0.0, 1.0)

    # Standard CG pipeline: Data is ACEScg -> Preview Window is sRGB (managed by OCIO)
    proc = _get_cpu_processor(src_space, "sRGB", ocio_config_path)
    if proc is not None:
        h, w, _ = rgb.shape
        desc = OCIO.PackedImageDesc(rgb, w, h, 3)
        proc.apply(desc)
        return np.clip(rgb, 0.0, 1.0)

    # Analytical ACEScg -> sRGB fallback
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


def read_exr_file(path: str) -> np.ndarray:
    """Read an OpenEXR (.exr) file into a float32 numpy array (H, W, C) in RGB(A) channel order."""
    if HAS_OPENEXR:
        try:
            exr = OpenEXR.InputFile(path)
            hdr = exr.header()
            dw = hdr["dataWindow"]
            w = dw.max.x - dw.min.x + 1
            h = dw.max.y - dw.min.y + 1
            ch_map = hdr["channels"]
            pt = Imath.PixelType(Imath.PixelType.FLOAT)

            def _read_ch(cname):
                raw = exr.channel(cname, pt)
                return np.frombuffer(raw, dtype=np.float32).reshape(h, w)

            if all(c in ch_map for c in ("R", "G", "B", "A")):
                chans = [_read_ch("R"), _read_ch("G"), _read_ch("B"), _read_ch("A")]
            elif all(c in ch_map for c in ("R", "G", "B")):
                chans = [_read_ch("R"), _read_ch("G"), _read_ch("B")]
            elif "Y" in ch_map:
                chans = [_read_ch("Y")]
            elif "A" in ch_map:
                chans = [_read_ch("A")]
            else:
                sorted_names = sorted(ch_map.keys())[:4]
                chans = [_read_ch(c) for c in sorted_names]
            exr.close()
            arr = np.stack(chans, axis=-1)
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
    arr = np.nan_to_num(np.ascontiguousarray(img_rgb, dtype=np.float32), nan=0.0, posinf=65504.0, neginf=-65504.0)
    if arr.ndim == 2:
        arr = arr[..., None]
    h, w, c = arr.shape

    if HAS_OPENEXR:
        try:
            use_half = bit_depth.lower() in ("half", "fp16", "float16", "16")
            pt = Imath.PixelType(Imath.PixelType.HALF if use_half else Imath.PixelType.FLOAT)
            np_dt = np.float16 if use_half else np.float32

            hdr = OpenEXR.Header(w, h)
            hdr["compression"] = Imath.Compression(Imath.Compression.ZIP_COMPRESSION)

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

    # Fallback to OpenCV EXR writer (convert RGB(A) -> BGR(A))
    bgr = arr[..., [2, 1, 0, 3][:c]] if c >= 3 else arr[..., 0]
    imwrite_unicode(path, bgr.astype(np.float32))


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
      - `.exr` files default to `input_colorspace='ACEScg'` (full float HDR range preserved, no [0,1] clamp!).
      - `.png`/`.jpg` files: if `input_colorspace='auto'`, kept in [0,1] normalized float (or transformed if explicit OCIO space given).
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

    if (
        not is_data_pass
        and im.shape[2] >= 3
        and input_colorspace not in ("", "auto", None, "Raw")
        and input_colorspace != working_colorspace
    ):
        im = transform_colorspace(im, input_colorspace, working_colorspace, ocio_config_path)
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
        raise ValueError(f"无法解码图像文件: {filename}")
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
