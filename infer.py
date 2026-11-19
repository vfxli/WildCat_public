# -*- coding: utf-8 -*-
"""WildCat inference CLI & streaming sequence runner.
Usage:
  python infer.py config/xxx.json [sequence_dir_or_pattern] [step]
  python infer.py --spec /path/to/spec.json
"""
from concurrent.futures import ThreadPoolExecutor
import json
import os
import signal
import sys
import time

_ROOT = os.path.dirname(os.path.abspath(__file__))
for _cand in (
    os.path.join(_ROOT, ".venv", "Lib", "site-packages"),
    os.path.join(_ROOT, ".venv", "lib", "python3.12", "site-packages"),
    os.path.join(_ROOT, "runtime", "python", "Lib", "site-packages"),
    os.path.join(_ROOT, "runtime", "python", "lib", "python3.12", "site-packages"),
    os.path.join(_ROOT, "..", "runtime", "python", "Lib", "site-packages"),
    os.path.join(_ROOT, "..", "runtime", "python", "lib", "python3.12", "site-packages"),
    os.path.join(_ROOT, "..", ".venv", "Lib", "site-packages"),
    os.path.join(_ROOT, "..", ".venv", "lib", "python3.12", "site-packages"),
):
    if os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.insert(0, _cand)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import cv2
import numpy as np
import torch

from wildcat import color, models
from wildcat.data import _read, expand_pattern
from wildcat.models import build_generator
from wildcat.preview import infer_image


def _atomic_json(path: str, payload: dict) -> None:
    if not path:
        return
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
        os.replace(tmp, path)
    except OSError:
        pass


def _match_channels(im: np.ndarray, channels: int) -> np.ndarray:
    arr = np.asarray(im, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]
    c = int(channels)
    if arr.shape[2] == c:
        return np.ascontiguousarray(arr, dtype=np.float32)
    if arr.shape[2] > c:
        return np.ascontiguousarray(arr[..., :c], dtype=np.float32)
    if c == 4 and arr.shape[2] == 3:
        alpha = np.ones((arr.shape[0], arr.shape[1], 1), dtype=np.float32)
        return np.ascontiguousarray(np.concatenate([arr, alpha], axis=2), dtype=np.float32)
    pad = np.zeros((arr.shape[0], arr.shape[1], c - arr.shape[2]), dtype=np.float32)
    return np.ascontiguousarray(np.concatenate([arr, pad], axis=2), dtype=np.float32)


def _load_model_for_infer(ckpt_path: str, task_cfg: dict, device: str, opt_nhwc: bool = False):
    mo = dict((task_cfg or {}).get("model") or {})
    ck = None
    try:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception:
        ck = None

    if isinstance(ck, dict) and "G" in ck:
        ck_mo = (ck.get("config") or {}).get("model") or {}
        if ck_mo:
            mo = dict(ck_mo)
        net, _ = build_generator(mo, cudnn_pad=True)
        net.load_state_dict(ck["G"], strict=True)
        net = net.to(device).eval()
        if opt_nhwc and str(device).startswith("cuda"):
            net = net.to(memory_format=torch.channels_last)
        return net, mo, int(ck.get("step") or 0), (ck.get("config") or {})

    # Fallback: exported model file (TorchScript .pt or preset/exported model via load_weights_into_model)
    try:
        w_spec = models.inspect_weight_source(
            ckpt_path,
            fallback_model_size=mo.get("model_size", "Medium"),
            fallback_in=int(mo.get("in_channels", 3)),
            fallback_out=int(mo.get("out_channels", mo.get("in_channels", 3))),
        )
        if w_spec.get("model_size"):
            mo["model_size"] = w_spec["model_size"]
            mo["global_attention"] = bool(w_spec.get("global_attention", mo.get("global_attention", False)))
        if w_spec.get("in_channels"):
            mo["in_channels"] = w_spec["in_channels"]
        if w_spec.get("out_channels"):
            mo["out_channels"] = w_spec["out_channels"]
        net, _ = build_generator(mo, cudnn_pad=True)
        models.load_weights_into_model(net, ckpt_path, model_size=mo.get("model_size", "Medium"), device="cpu")
        net = net.to(device).eval()
        if opt_nhwc and str(device).startswith("cuda"):
            net = net.to(memory_format=torch.channels_last)
        return net, mo, 0, {}
    except Exception:
        jit_net = torch.jit.load(ckpt_path, map_location=device)
        jit_net.eval()
        return jit_net, mo, 0, {}


def _save_output_atomic(out_path: str, arr_rgb: np.ndarray, write_as_exr: bool, exr_depth: str, colorspace: str = "ACEScg") -> None:
    stem, ext = os.path.splitext(out_path)
    tmp_path = f"{stem}.tmp_{os.getpid()}{ext}"
    try:
        if write_as_exr:
            color.write_exr_file(tmp_path, arr_rgb, bit_depth=exr_depth, colorspace=colorspace)
        else:
            clipped = np.clip(arr_rgb, 0.0, 1.0)
            ext_low = ext.lower()
            if ext_low in (".jpg", ".jpeg"):
                if clipped.shape[2] >= 3:
                    bgr = clipped[..., [2, 1, 0]]
                else:
                    bgr = clipped[:, :, 0]
                u8 = (bgr * 255.0 + 0.5).astype(np.uint8)
                color.imwrite_unicode(tmp_path, u8, [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                if clipped.shape[2] >= 3:
                    clipped = clipped[..., [2, 1, 0] + list(range(3, clipped.shape[2]))]
                elif clipped.shape[2] == 1:
                    clipped = clipped[:, :, 0]
                u16 = (clipped * 65535.0 + 0.5).astype(np.uint16)
                color.imwrite_unicode(tmp_path, u16)
        os.replace(tmp_path, out_path)
    except BaseException:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


@torch.no_grad()
def run_spec(spec_path: str) -> int:
    with open(spec_path, "r", encoding="utf-8") as fh:
        spec = json.load(fh)

    ckpt_path = spec["ckpt_path"]
    task_cfg = spec.get("task_config") or {}
    input_dir = spec["input_dir"]
    out_dir = spec["output_dir"]
    prog_path = spec.get("progress_path") or ""
    cancel_path = spec.get("cancel_path") or ""
    out_fmt = str(spec.get("output_format") or "exr_float")
    os.makedirs(out_dir, exist_ok=True)

    stop = {"flag": False}

    def _on_sig(*_):
        stop["flag"] = True

    for sig_name in ("SIGTERM", "SIGINT", "SIGBREAK"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            try:
                signal.signal(sig, _on_sig)
            except Exception:
                pass

    req_dev = str(spec.get("device") or "cuda:0")
    if torch.cuda.is_available() and req_dev.startswith("cuda"):
        device = req_dev
    elif torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    gpu_label = str(spec.get("gpu_label") or device)

    # Production delivery sequence inference: strict IEEE 754 FP32 (no TF32 rounding, no AMP quantization)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    net, mo, ck_step, ck_cfg = _load_model_for_infer(ckpt_path, task_cfg, device, opt_nhwc=False)
    c_in = int(mo.get("in_channels", 3))
    c_out = int(mo.get("out_channels", c_in))
    col = (task_cfg.get("color") or ck_cfg.get("color") or {})
    working_cs = col.get("working_colorspace", "ACEScg")
    source_cs = spec.get("input_colorspace") or col.get("source_colorspace") or working_cs
    if source_cs == "auto":
        source_cs = working_cs
    target_cs = col.get("target_colorspace") or working_cs
    if target_cs == "auto":
        target_cs = working_cs
    ocio_cfg = col.get("ocio_config", "")
    view_mode = col.get("view_transform", "ACES 1.0 - SDR Video")

    files = expand_pattern(input_dir)
    if not files:
        raise ValueError(f"No sequence frames found in: {input_dir}")

    raw_fidx = spec.get("frame_index")
    if raw_fidx is not None:
        fidx = max(0, min(len(files) - 1, int(raw_fidx)))
        files = [files[fidx]]

    resolved_step = int(spec.get("ckpt_step") or ck_step)
    total = len(files)
    t0 = time.time()
    _atomic_json(prog_path, {"done": 0, "total": total, "fps": 0.0, "device": gpu_label, "step": resolved_step})

    eval_collector = None
    eval_spec = spec.get("eval_spec") if isinstance(spec.get("eval_spec"), dict) else None
    if eval_spec and eval_spec.get("eval_out_dir") and raw_fidx is None and total >= 2:
        from wildcat import evaluator as _evaluator

        eval_collector = _evaluator.SequenceEvalCollector(
            eval_out_dir=str(eval_spec["eval_out_dir"]),
            eval_key=str(eval_spec.get("eval_key") or f"step{resolved_step}"),
            ckpt_step=resolved_step,
            ckpt_file=os.path.basename(ckpt_path),
            c_in=c_in,
            c_out=c_out,
            working_cs=working_cs,
            ocio_cfg=ocio_cfg,
            eval_asset_id=eval_spec.get("eval_asset_id"),
            eval_asset_name=str(eval_spec.get("eval_asset_name") or ""),
            ckpt_mode=str(eval_spec.get("ckpt_mode") or "latest"),
            device_label=gpu_label,
        )

    write_exr = out_fmt in ("exr", "exr_half", "exr_float")
    exr_depth = "float" if out_fmt == "exr_float" else "half"
    out_ext = ".exr" if write_exr else (".jpg" if out_fmt == "jpg_95" else ".png")

    for idx, path in enumerate(files, start=1):
        if stop["flag"] or (cancel_path and os.path.exists(cancel_path)):
            return 130
        im = _read(path, input_colorspace=source_cs, working_colorspace=working_cs, ocio_config_path=ocio_cfg, is_data_pass=(c_in == 1))
        im = _match_channels(im, c_in)
        x = torch.from_numpy(im).permute(2, 0, 1)[None].to(device)
        y_raw = infer_image(net, x, amp_dtype=None, channels_last=False)[0].permute(1, 2, 0).cpu().numpy()

        if eval_collector is not None:
            eval_collector.add_frame(idx - 1, path, im, y_raw)

        y = y_raw
        if not write_exr and y.shape[2] >= 3 and working_cs != "Raw":
            rgb_disp = color.acescg_to_display_srgb(y[..., :3], src_space=working_cs, view_mode=view_mode, ocio_config_path=ocio_cfg)
            if y.shape[2] >= 4:
                y = np.concatenate([rgb_disp, np.clip(y[..., 3:4], 0.0, 1.0)], axis=2)
            else:
                y = rgb_disp

        stem = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(out_dir, stem + out_ext)
        _save_output_atomic(out_path, y, write_exr, exr_depth, colorspace=working_cs)

        dt = max(time.time() - t0, 1e-6)
        _atomic_json(
            prog_path,
            {
                "done": idx,
                "total": total,
                "fps": round(idx / dt, 2),
                "seconds": round(dt, 2),
                "device": gpu_label,
                "width": int(y.shape[1]),
                "height": int(y.shape[0]),
                "out_channels": int(y.shape[2]),
            },
        )

    if eval_collector is not None:
        if stop["flag"] or (cancel_path and os.path.exists(cancel_path)):
            return 130
        eval_collector.finalize(
            sup_source_dir=str(eval_spec.get("sup_source_dir") or ""),
            sup_target_dir=str(eval_spec.get("sup_target_dir") or ""),
            source_cs=str(col.get("source_colorspace") or working_cs),
            target_cs=target_cs,
            infer_fn=lambda arr: infer_image(
                net,
                torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).permute(2, 0, 1)[None].to(device),
                amp_dtype=None,
                channels_last=False,
            )[0].permute(1, 2, 0).cpu().numpy(),
        )
    return 0


@torch.no_grad()
def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--spec":
        sys.exit(run_spec(sys.argv[2]))

    with open(sys.argv[1], "r", encoding="utf-8") as _f:
        cfg = json.load(_f)
    args = sys.argv[2:]
    step_arg = None
    if args and args[-1].isdigit():
        step_arg = int(args.pop())
    pattern = args[0] if args else cfg["paths"]["source"]

    paths_cfg = cfg.get("paths") or {}
    run_dir = paths_cfg.get("run_dir") or os.path.join("run", cfg["run_name"])
    ck_name = "latest.pt" if step_arg is None else "step_%06d.pt" % step_arg
    ckpt_path = os.path.join(run_dir, "ckpt", ck_name)
    out_dir = paths_cfg.get("output") or os.path.join(run_dir, "infer")
    os.makedirs(out_dir, exist_ok=True)

    opts = cfg.get("optimizations", {}) or {}
    lossless = opts.get("lossless", {}) or {}

    opt_async_io = bool(lossless.get("async_io", False))
    prec = (cfg.get("train") or {}).get("precision", "fp32")
    opt_nhwc = bool(lossless.get("channels_last", False))

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.allow_tf32 = True

    req_dev = str((cfg.get("train") or {}).get("device") or "")
    if torch.cuda.is_available():
        device = req_dev if req_dev.startswith("cuda") else "cuda"
    else:
        device = "cpu"

    net, mo, ck_step, ck_cfg = _load_model_for_infer(ckpt_path, cfg, device, opt_nhwc=opt_nhwc)
    amp_dtype = {"fp16": torch.float16, "fp32": None}.get(prec, None)
    ac_enabled = amp_dtype is not None and device.startswith("cuda")

    c_in = int(mo.get("in_channels", 3))
    col = cfg.get("color") or ck_cfg.get("color") or {}
    working_cs = col.get("working_colorspace", "ACEScg")
    source_cs = col.get("source_colorspace") or working_cs
    if source_cs == "auto":
        source_cs = working_cs
    raw_output_cs = col.get("output_colorspace", "")
    ocio_cfg = col.get("ocio_config", "")
    out_fmt = col.get("output_format", "exr_half")  # 'exr_half' | 'exr_float' | 'png_16'

    print(
        "ckpt step %d | input %s | color=%s->%s | prec=%s nhwc=%s"
        % (ck_step if ck_step else -1, pattern, source_cs, working_cs, prec, opt_nhwc)
    )

    files = expand_pattern(pattern)
    if not files:
        raise IOError("no frames match: " + str(pattern))

    pool = ThreadPoolExecutor(max_workers=4) if opt_async_io else None
    t0 = time.time()

    def _save_cli(out_path, arr_rgb, write_as_exr, exr_depth):
        _save_output_atomic(out_path, arr_rgb, write_as_exr, exr_depth, colorspace=working_cs)
        print(out_path)

    for path in files:
        im = _read(path, input_colorspace=source_cs, working_colorspace=working_cs, ocio_config_path=ocio_cfg, is_data_pass=(c_in == 1))
        im = _match_channels(im, c_in)
        x = torch.from_numpy(im).permute(2, 0, 1)[None].to(device)
        y = infer_image(net, x, amp_dtype if ac_enabled else None, opt_nhwc and device.startswith("cuda"))
        y = y[0].permute(1, 2, 0).cpu().numpy()

        stem, ext = os.path.splitext(os.path.basename(path))
        write_exr = out_fmt in ("exr", "exr_half", "exr_float") or (out_fmt == "auto" and ext.lower() == ".exr")
        exr_depth = "float" if out_fmt == "exr_float" else "half"
        if raw_output_cs and raw_output_cs != "auto":
            eff_out_cs = raw_output_cs
        elif write_exr:
            eff_out_cs = working_cs
        else:
            eff_out_cs = "sRGB" if (source_cs != "Raw" and working_cs != "Raw") else working_cs

        if y.shape[2] >= 3 and eff_out_cs and eff_out_cs != working_cs and eff_out_cs != "Raw":
            y = color.transform_colorspace(y, working_cs, eff_out_cs, ocio_cfg)

        out_ext = ".exr" if write_exr else (".png" if out_fmt == "png_16" else ext)
        out = os.path.join(out_dir, stem + out_ext)
        if pool is not None:
            pool.submit(_save_cli, out, y, write_exr, exr_depth)
        else:
            _save_cli(out, y, write_exr, exr_depth)

    if pool is not None:
        pool.shutdown(wait=True)
    dt = max(time.time() - t0, 1e-6)
    print("Rendered %d frames in %.2fs (%.1f fps)" % (len(files), dt, len(files) / dt))


if __name__ == "__main__":
    import traceback

    try:
        main()
    except SystemExit:
        raise
    except KeyboardInterrupt:
        sys.exit(130)
    except BaseException:
        tb = traceback.format_exc()
        sys.stderr.write(tb + "\n")
        sys.stderr.flush()
        try:
            if len(sys.argv) > 1 and sys.argv[1] != "--spec" and os.path.isfile(sys.argv[1]):
                with open(sys.argv[1], "r", encoding="utf-8") as _f:
                    _c = json.load(_f)
                if isinstance(_c, dict) and _c.get("run_name"):
                    _rd = os.path.join(_ROOT, "run", str(_c["run_name"]))
                    os.makedirs(_rd, exist_ok=True)
                    with open(os.path.join(_rd, "train.log"), "a", encoding="utf-8") as _lf:
                        _lf.write(tb + "\n")
        except Exception:
            pass
        sys.exit(1)
