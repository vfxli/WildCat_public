# -*- coding: utf-8 -*-
"""Streaming GPU Sequence & Fidelity Evaluator for WildCat Experiment Lab.

Computes:
  1. Supervised keyframe fidelity metrics on paired frames (PSNR in ACEScg & display sRGB,
     MSE, MAE, Sobel/gradient Edge-L1, untouched Background Drift MAE, P99 pixel error).
  2. Full-sequence inference + unsupervised temporal stability metrics on arbitrary shot plates
     without ground truth:
       - 1st-order Edit-Layer Jitter:  mean |(y_hat_t - x_t) - (y_hat_{t-1} - x_{t-1})|
       - 2nd-order HF Chatter (HFI):   mean |delta_t - 2*delta_{t-1} + delta_{t-2}|
       - Static-Region Boiling:        mean |y_hat_t - y_hat_{t-1}| on |x_t - x_{t-1}| < 0.005
       - Per-frame timeline curve + amplified visual flicker/error heatmaps (O(1) RAM/VRAM).

Usage:
  python -m wildcat.evaluator <spec_json_path>
"""
import json
import math
import os
import signal
import sys
import time

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _cand in (
    os.path.join(_ROOT, ".venv", "Lib", "site-packages"),
    os.path.join(_ROOT, ".venv", "lib", "python3.12", "site-packages"),
    os.path.join(_ROOT, "runtime", "python", "Lib", "site-packages"),
    os.path.join(_ROOT, "runtime", "python", "lib", "python3.12", "site-packages"),
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

from wildcat import color
from wildcat.data import _read, expand_pattern
from wildcat.models import build_generator
from wildcat.preview import infer_image


def _atomic_write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _match_channels(im: np.ndarray, channels: int) -> np.ndarray:
    c = int(channels)
    if im.shape[2] == c:
        return im
    if im.shape[2] > c:
        return np.ascontiguousarray(im[..., :c], dtype=np.float32)
    if c == 4 and im.shape[2] == 3:
        alpha = np.ones((im.shape[0], im.shape[1], 1), dtype=np.float32)
        return np.concatenate([im, alpha], axis=2)
    pad = np.zeros((im.shape[0], im.shape[1], c - im.shape[2]), dtype=np.float32)
    return np.concatenate([im, pad], axis=2)


def _resize_preview_u8(bgr_u8: np.ndarray, max_side: int = 1280) -> np.ndarray:
    if max_side <= 0:
        return bgr_u8
    h, w = bgr_u8.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return bgr_u8
    scale = max_side / float(m)
    nw = max(2, int(round(w * scale)))
    nh = max(2, int(round(h * scale)))
    return cv2.resize(bgr_u8, (nw, nh), interpolation=cv2.INTER_AREA)


def _render_rgba_checker_srgb(rgb_srgb_01: np.ndarray, alpha_01: np.ndarray, tile: int = 16) -> np.ndarray:
    """Composite premultiplied display sRGB [0, 1] with alpha [0, 1] over a dark Nuke-style checkerboard."""
    a = np.clip(alpha_01, 0.0, 1.0)
    if a.ndim == 2:
        a = a[..., None]
    if float(np.min(a)) >= 0.995:
        return rgb_srgb_01
    h, w = rgb_srgb_01.shape[:2]
    yy = (np.arange(h, dtype=np.int32) // tile)[:, None]
    xx = (np.arange(w, dtype=np.int32) // tile)[None, :]
    chk = np.where(((yy + xx) & 1) == 0, 0.11, 0.165).astype(np.float32)[..., None]
    return np.clip(rgb_srgb_01 + (1.0 - a) * chk, 0.0, 1.0)


def _render_heatmap_bgr(diff_map_01: np.ndarray, ref_srgb_01: np.ndarray, gain: float = 12.0) -> np.ndarray:
    """Blend an Inferno false-color heatmap of `diff_map_01 * gain` over a darkened grayscale plate."""
    amp = np.clip(diff_map_01 * float(gain), 0.0, 1.0)
    amp_u8 = (amp * 255.0 + 0.5).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(amp_u8, cv2.COLORMAP_INFERNO)
    gray = np.mean(ref_srgb_01[..., :3], axis=2, keepdims=True)
    base_bgr = (np.clip(gray, 0.0, 1.0) * (255.0 * 0.28)).astype(np.float32)
    weight = np.clip(amp[..., None] * 1.6, 0.15, 0.92).astype(np.float32)
    blended = base_bgr * (1.0 - weight) + heat_bgr.astype(np.float32) * weight
    return np.clip(blended + 0.5, 0.0, 255.0).astype(np.uint8)


def _edge_l1(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean finite-difference spatial gradient L1 error across channels."""
    if pred.shape[0] < 2 or pred.shape[1] < 2:
        return float(np.mean(np.abs(pred - gt)))
    dx_p = pred[:, 1:, :] - pred[:, :-1, :]
    dx_g = gt[:, 1:, :] - gt[:, :-1, :]
    dy_p = pred[1:, :, :] - pred[:-1, :, :]
    dy_g = gt[1:, :, :] - gt[:-1, :, :]
    return 0.5 * (float(np.mean(np.abs(dx_p - dx_g))) + float(np.mean(np.abs(dy_p - dy_g))))


def _compute_pair_metrics(
    src_lin: np.ndarray,
    tgt_lin: np.ndarray,
    pred_lin: np.ndarray,
    src_srgb: np.ndarray,
    tgt_srgb: np.ndarray,
    pred_srgb: np.ndarray,
) -> dict:
    diff_lin = pred_lin - tgt_lin
    mse_lin = float(np.mean(diff_lin * diff_lin))
    mae_lin = float(np.mean(np.abs(diff_lin)))
    psnr_aces = float(10.0 * math.log10(1.0 / max(mse_lin, 1e-12)))

    diff_srgb = pred_srgb - tgt_srgb
    mse_srgb = float(np.mean(diff_srgb * diff_srgb))
    psnr_srgb = float(10.0 * math.log10(1.0 / max(mse_srgb, 1e-12)))

    edge_err = _edge_l1(pred_lin, tgt_lin)
    pix_l1 = np.mean(np.abs(diff_lin), axis=2)
    p99_err = float(np.percentile(pix_l1, 99.0))

    bg_drift = None
    if src_lin.shape == tgt_lin.shape:
        untouched = np.max(np.abs(tgt_lin - src_lin), axis=2) < 1e-4
        if float(np.mean(untouched)) >= 0.05:
            bg_drift = float(np.mean(pix_l1[untouched]))

    return {
        "mse": mse_lin,
        "mae": mae_lin,
        "psnr_aces": psnr_aces,
        "psnr_srgb": psnr_srgb,
        "edge_l1": edge_err,
        "p99_err": p99_err,
        "bg_drift_mae": bg_drift,
    }


def _load_eval_model(ckpt_path: str, task_cfg: dict, device: str):
    """Load generator weights from either a WildCat state-dict checkpoint or a TorchScript .pt."""
    mo = dict((task_cfg or {}).get("model") or {})
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
        step = int(ck.get("step") or 0)
        return net, mo, step, False

    # Fallback: TorchScript exported model (Training_YYMMDD_HHMMSS.<step>.pt)
    jit_net = torch.jit.load(ckpt_path, map_location=device)
    jit_net.eval()
    return jit_net, mo, 0, True


@torch.no_grad()
def run_evaluation(spec: dict) -> dict:
    t0 = time.time()
    ckpt_path = spec["ckpt_path"]
    task_cfg = spec.get("task_config") or {}
    eval_out_dir = spec["eval_out_dir"]
    os.makedirs(eval_out_dir, exist_ok=True)
    prog_path = os.path.join(eval_out_dir, "progress.json")

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

    # Disable TF32 during scientific evaluation so metrics are bit-accurate across GPU architectures
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    net, mo, ck_step, _is_jit = _load_eval_model(ckpt_path, task_cfg, device)
    ckpt_step = int(spec.get("ckpt_step") or ck_step or 0)
    c_in = int(mo.get("in_channels", 3))
    c_out = int(mo.get("out_channels", 3))

    col = (task_cfg.get("color") or {})
    working_cs = col.get("working_colorspace", "ACEScg")
    train_src_cs = col.get("source_colorspace", "auto")
    train_tgt_cs = col.get("target_colorspace", "auto")
    ocio_cfg = col.get("ocio_config", "")
    max_side = int(spec.get("max_preview_side") or 1280)

    def _infer_frame(src_np: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(src_np, dtype=np.float32)).permute(2, 0, 1)[None].to(device)
        out = infer_image(net, x, amp_dtype=None, channels_last=False)
        return out[0].permute(1, 2, 0).cpu().numpy()

    # ── 1. Supervised Keyframe Fidelity on Task's Paired Training Set ──
    paths_cfg = task_cfg.get("paths") or {}
    train_src_files = expand_pattern(paths_cfg.get("source", "")) if paths_cfg.get("source") else []
    train_tgt_files = expand_pattern(paths_cfg.get("target", "")) if paths_cfg.get("target") else []
    eval_src_pattern = str(spec.get("eval_source_dir") or "").strip()
    eval_tgt_pattern = str(spec.get("eval_target_dir") or "").strip()
    seq_src_files = expand_pattern(eval_src_pattern) if eval_src_pattern else []
    seq_tgt_files = expand_pattern(eval_tgt_pattern) if eval_tgt_pattern else []
    seq_is_paired = bool(seq_src_files and len(seq_tgt_files) == len(seq_src_files))

    n_sup = min(len(train_src_files), len(train_tgt_files))
    n_seq = len(seq_src_files)
    total_work = n_sup + n_seq
    done_work = 0

    def _report_progress(phase: str):
        _atomic_write_json(
            prog_path,
            {
                "status": "running",
                "phase": phase,
                "done": done_work,
                "total": total_work,
                "updated_at": time.time(),
            },
        )

    _report_progress("supervised")
    sup_summary = None
    if n_sup > 0:
        sup_rows = []
        for idx in range(n_sup):
            if stop["flag"]:
                raise InterruptedError("Evaluation cancelled")
            s_im = _match_channels(
                _read(train_src_files[idx], train_src_cs, working_cs, ocio_cfg, is_data_pass=(c_in == 1)),
                c_in,
            )
            t_im = _match_channels(
                _read(train_tgt_files[idx], train_tgt_cs, working_cs, ocio_cfg, is_data_pass=(c_out == 1)),
                c_out,
            )
            p_im = _match_channels(_infer_frame(s_im), c_out)

            s_srgb = color.acescg_to_display_srgb(s_im, src_space=working_cs, ocio_config_path=ocio_cfg)
            t_srgb = color.acescg_to_display_srgb(t_im, src_space=working_cs, ocio_config_path=ocio_cfg)
            p_srgb = color.acescg_to_display_srgb(p_im, src_space=working_cs, ocio_config_path=ocio_cfg)
            m = _compute_pair_metrics(s_im, t_im, p_im, s_srgb, t_srgb, p_srgb)
            m["name"] = os.path.basename(train_src_files[idx])
            sup_rows.append(m)
            done_work += 1
            _report_progress("supervised")

        bg_vals = [r["bg_drift_mae"] for r in sup_rows if r["bg_drift_mae"] is not None]
        sup_summary = {
            "frame_count": n_sup,
            "mse": float(np.mean([r["mse"] for r in sup_rows])),
            "mae": float(np.mean([r["mae"] for r in sup_rows])),
            "psnr_aces": float(np.mean([r["psnr_aces"] for r in sup_rows])),
            "psnr_srgb": float(np.mean([r["psnr_srgb"] for r in sup_rows])),
            "edge_l1": float(np.mean([r["edge_l1"] for r in sup_rows])),
            "p99_err": float(np.mean([r["p99_err"] for r in sup_rows])),
            "bg_drift_mae": float(np.mean(bg_vals)) if bg_vals else None,
        }

    # ── 2. Streaming Full-Sequence Inference & Unsupervised Temporal Flicker ──
    seq_summary = None
    if n_seq > 0:
        out_jpg_dir = os.path.join(eval_out_dir, "out")
        flk_jpg_dir = os.path.join(eval_out_dir, "flicker")
        err_jpg_dir = os.path.join(eval_out_dir, "err")
        os.makedirs(out_jpg_dir, exist_ok=True)
        os.makedirs(flk_jpg_dir, exist_ok=True)
        if seq_is_paired:
            os.makedirs(err_jpg_dir, exist_ok=True)

        eval_cs = str(spec.get("eval_colorspace") or train_src_cs or "auto")
        per_frame = []
        prev_src_srgb = None
        prev_out_srgb = None
        prev_delta = None
        prev2_delta = None

        delta_jitters = []
        hf_chatters = []
        static_boils = []
        edit_mags = []
        seq_pair_rows = []

        peak_flicker_idx = 0
        peak_flicker_val = -1.0

        _report_progress("sequence")
        for idx in range(n_seq):
            if stop["flag"]:
                raise InterruptedError("Evaluation cancelled")
            s_path = seq_src_files[idx]
            s_im = _match_channels(
                _read(s_path, eval_cs, working_cs, ocio_cfg, is_data_pass=(c_in == 1)),
                c_in,
            )
            p_im = _match_channels(_infer_frame(s_im), c_out)

            s_srgb = color.acescg_to_display_srgb(s_im, src_space=working_cs, ocio_config_path=ocio_cfg)
            p_srgb = color.acescg_to_display_srgb(p_im, src_space=working_cs, ocio_config_path=ocio_cfg)
            p_disp_srgb = _render_rgba_checker_srgb(p_srgb, p_im[..., 3:4]) if c_out == 4 else p_srgb
            out_sig = np.concatenate([p_srgb, np.clip(p_im[..., 3:4], 0.0, 1.0)], axis=2) if c_out == 4 else p_srgb

            # Channel-aware edit residual / matte signal in perceptual display space [0, 1]
            if c_in == c_out:
                if c_out == 4:
                    a_diff = np.clip(p_im[..., 3:4], 0.0, 1.0) - np.clip(s_im[..., 3:4], 0.0, 1.0)
                    cur_delta = np.concatenate([p_srgb - s_srgb, a_diff], axis=2)
                else:
                    cur_delta = p_srgb - s_srgb
            elif c_out == 1:
                cur_delta = np.clip(p_im[..., :1], 0.0, 1.0)
            elif c_in == 3 and c_out == 4:
                cur_delta = np.concatenate([p_srgb - s_srgb, np.clip(p_im[..., 3:4], 0.0, 1.0)], axis=2)
            else:
                cur_delta = p_srgb
            edit_l1 = float(np.mean(np.abs(cur_delta)))
            edit_mags.append(edit_l1)

            flk_delta = None
            flk_hf = None
            flk_static = None

            if prev_delta is not None:
                dd = np.abs(cur_delta - prev_delta)
                flk_delta = float(np.mean(dd))
                delta_jitters.append(flk_delta)
                if flk_delta > peak_flicker_val:
                    peak_flicker_val = flk_delta
                    peak_flicker_idx = idx
                flk_map_01 = np.max(dd, axis=2)

                plate_motion = np.max(np.abs(s_srgb - prev_src_srgb), axis=2)
                static_mask = plate_motion < 0.005
                if float(np.mean(static_mask)) >= 0.02:
                    out_diff = np.mean(np.abs(out_sig - prev_out_srgb), axis=2)
                    flk_static = float(np.mean(out_diff[static_mask]))
                    static_boils.append(flk_static)
            else:
                flk_map_01 = np.zeros((p_srgb.shape[0], p_srgb.shape[1]), dtype=np.float32)

            if prev2_delta is not None and prev_delta is not None:
                acc = np.abs(cur_delta - 2.0 * prev_delta + prev2_delta)
                flk_hf = float(np.mean(acc))
                hf_chatters.append(flk_hf)

            # Save compact sRGB output & flicker heatmap frames
            fname_jpg = f"frame_{idx:04d}.jpg"
            out_bgr = _resize_preview_u8(
                np.clip(p_disp_srgb[..., ::-1] * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8),
                max_side=max_side,
            )
            color.imwrite_unicode(
                os.path.join(out_jpg_dir, fname_jpg), out_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90]
            )

            flk_bgr = _resize_preview_u8(
                _render_heatmap_bgr(flk_map_01, s_srgb, gain=12.0),
                max_side=max_side,
            )
            color.imwrite_unicode(
                os.path.join(flk_jpg_dir, fname_jpg), flk_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90]
            )

            row = {
                "idx": idx,
                "name": os.path.basename(s_path),
                "file": fname_jpg,
                "edit_l1": edit_l1,
                "flicker_delta": flk_delta,
                "flicker_hf": flk_hf,
                "flicker_static": flk_static,
            }

            if seq_is_paired:
                t_im = _match_channels(
                    _read(seq_tgt_files[idx], eval_cs, working_cs, ocio_cfg, is_data_pass=(c_out == 1)),
                    c_out,
                )
                t_srgb = color.acescg_to_display_srgb(t_im, src_space=working_cs, ocio_config_path=ocio_cfg)
                pm = _compute_pair_metrics(s_im, t_im, p_im, s_srgb, t_srgb, p_srgb)
                seq_pair_rows.append(pm)
                row["mse"] = pm["mse"]
                row["psnr_srgb"] = pm["psnr_srgb"]
                row["psnr_aces"] = pm["psnr_aces"]

                err_map_01 = np.max(np.abs(p_srgb - t_srgb), axis=2)
                err_bgr = _resize_preview_u8(
                    _render_heatmap_bgr(err_map_01, t_srgb, gain=10.0),
                    max_side=max_side,
                )
                color.imwrite_unicode(
                    os.path.join(err_jpg_dir, fname_jpg), err_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90]
                )

            per_frame.append(row)
            prev2_delta = prev_delta
            prev_delta = cur_delta
            prev_src_srgb = s_srgb
            prev_out_srgb = out_sig

            done_work += 1
            _report_progress("sequence")

        seq_summary = {
            "frame_count": n_seq,
            "is_paired": seq_is_paired,
            "edit_l1": float(np.mean(edit_mags)) if edit_mags else 0.0,
            "flicker_delta": float(np.mean(delta_jitters)) if delta_jitters else None,
            "flicker_delta_max": float(np.max(delta_jitters)) if delta_jitters else None,
            "flicker_hf": float(np.mean(hf_chatters)) if hf_chatters else None,
            "flicker_static": float(np.mean(static_boils)) if static_boils else None,
            "peak_flicker_idx": int(peak_flicker_idx) if delta_jitters else 0,
            "peak_flicker_val": float(peak_flicker_val) if delta_jitters else None,
            "per_frame": per_frame,
        }
        if seq_is_paired and seq_pair_rows:
            seq_bg = [r["bg_drift_mae"] for r in seq_pair_rows if r["bg_drift_mae"] is not None]
            seq_summary.update(
                {
                    "mse": float(np.mean([r["mse"] for r in seq_pair_rows])),
                    "mae": float(np.mean([r["mae"] for r in seq_pair_rows])),
                    "psnr_aces": float(np.mean([r["psnr_aces"] for r in seq_pair_rows])),
                    "psnr_srgb": float(np.mean([r["psnr_srgb"] for r in seq_pair_rows])),
                    "edge_l1": float(np.mean([r["edge_l1"] for r in seq_pair_rows])),
                    "p99_err": float(np.mean([r["p99_err"] for r in seq_pair_rows])),
                    "bg_drift_mae": float(np.mean(seq_bg)) if seq_bg else None,
                }
            )

    elapsed = round(time.time() - t0, 2)
    result = {
        "eval_key": spec.get("eval_key", ""),
        "eval_asset_id": spec.get("eval_asset_id"),
        "eval_asset_name": spec.get("eval_asset_name", ""),
        "ckpt_mode": spec.get("ckpt_mode", "best"),
        "ckpt_step": ckpt_step,
        "ckpt_file": os.path.basename(ckpt_path),
        "device": device,
        "elapsed_sec": elapsed,
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "supervised": sup_summary,
        "sequence": seq_summary,
    }
    _atomic_write_json(os.path.join(eval_out_dir, "metrics.json"), result)
    _atomic_write_json(
        prog_path,
        {
            "status": "completed",
            "phase": "done",
            "done": total_work,
            "total": total_work,
            "updated_at": time.time(),
        },
    )
    return result


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m wildcat.evaluator <spec_json_path>")
    spec_path = sys.argv[1]
    with open(spec_path, "r", encoding="utf-8") as fh:
        spec = json.load(fh)
    res = run_evaluation(spec)
    sup_psnr = (res.get("supervised") or {}).get("psnr_srgb")
    seq_flk = (res.get("sequence") or {}).get("flicker_hf")
    print(
        f"[Evaluator] step={res['ckpt_step']} elapsed={res['elapsed_sec']}s "
        f"sup_psnr_srgb={sup_psnr} seq_hfi={seq_flk}"
    )


if __name__ == "__main__":
    main()
