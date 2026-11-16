# -*- coding: utf-8 -*-
"""Validation-sequence checkpoint evaluator, checkpoint recommendation, auto-stop score and auto-extend judge.

All of this is optional: nothing here runs unless a job has a validation sequence (`paths.eval_sequence`),
`train.stop_after_optimal_epochs` or `train.auto_extend_optimal`.  Default training is not affected.

What is evaluated per saved checkpoint:
  1. Paired keyframes (the frames the model was trained on): `psnr_aces`, `psnr_srgb`, `roi_psnr`, `edge_l1`, `bg_drift_mae`
  2. Validation sequence (no ground truth): `seq_static`, `seq_delta`, `seq_hfi`, `seq_edit_l1`

Rules after the 2026-11 re-validation (264 runs, 3 seeds, 24-83 unseen test frames per dataset, 5 fixed evaluation
points per run):

  - Recommended checkpoint (`balanced`) = the latest full-resolution (stage 1/1) checkpoint.  On the records the final
    step was the best of the evaluated steps in 77 % of the runs and otherwise within 0.36 dB of the best one
    (0.67 dB on a dataset that was excluded as a failure case); no rule based on the training keyframes could tell
    which runs those were.  The former composite score is still stored for display but no longer chooses anything.
  - `detail` = the stage-1/1 checkpoint that fits the edit region of the keyframes best (`roi_psnr`, `edge_l1`).
    It describes the fit to the training frames only.
  - The "Seen PSNR is saturated" rules (46 / 58 dB in scoring and auto-stop, 41 / 56 dB in auto-extend) were removed:
    they fired on a third of the runs, made the auto-stop lock a step that was 1.5 dB (up to 5.1 dB) worse on unseen
    frames, and vetoed extensions on the dataset where training longer gained up to 4.8 dB on unseen frames.
  - Everything that depended on `seq_hfi` was removed.  It is the mean 2nd-order temporal difference of the edit
    layer (output - input); it follows how much input grain the model alters and how much the subject moves and is
    NOT a flicker measure.  It is still computed, stored and shown.  `train.hfi_in_decisions` is ignored.
  - Auto-extend continues only while the fit to the training keyframes still improves (>= 0.20 dB per chunk) and never
    past its caps.  That criterion follows the training frames: on the 18 runs with a longer reference run it said
    "extend" every time; training longer helped unseen frames in 15 of them and hurt in 3 (4-px grid, degrain).
  - Not testable on the records (kept as they were): the untouched-background / static-boil stability terms, the
    `edge_l1` terms, the matte-collapse guard, the 0.015 dB new-best margin of the auto-stop, and the learning-rate
    shape of an extension round (wildcat/trainer.py).
"""
import glob
import json
import math
import os
import re
import time

import numpy as np
import torch

from . import advanced, color, data, log as _log_mod, models
from .preview import infer_image

_STEP_PT_RE = re.compile(r"step_(\d+)\.pt$")
RECOMMENDATION_FILENAME = "recommended_models.json"

# `seq_hfi` is not a flicker measure (see the module docstring).  Since the 2026-11 re-validation it takes no part in any
# automated decision and the opt-in switch `train.hfi_in_decisions` is ignored; the names below are kept so that old
# configs, old `recommended_models.json` files and callers keep working.
HFI_IN_DECISIONS_DEFAULT = False


def hfi_in_decisions(cfg: dict | None) -> bool:
    """Always False: the rules that used `seq_hfi` were removed (a config that still sets the flag is accepted)."""
    return False


def _atomic_write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _match_channels(im: np.ndarray, channels: int) -> np.ndarray:
    c = int(channels)
    if im.ndim == 2:
        im = im[..., None]
    if im.shape[2] == c:
        return np.ascontiguousarray(im, dtype=np.float32)
    if im.shape[2] > c:
        return np.ascontiguousarray(im[..., :c], dtype=np.float32)
    if c == 4 and im.shape[2] == 3:
        alpha = np.ones((im.shape[0], im.shape[1], 1), dtype=np.float32)
        return np.concatenate([im, alpha], axis=2).astype(np.float32, copy=False)
    pad = np.zeros((im.shape[0], im.shape[1], c - im.shape[2]), dtype=np.float32)
    return np.concatenate([im, pad], axis=2).astype(np.float32, copy=False)


def _edge_l1(pred: np.ndarray, gt: np.ndarray) -> float:
    if pred.shape[0] < 2 or pred.shape[1] < 2:
        return float(np.mean(np.abs(pred - gt)))
    dx_p = pred[:, 1:, :] - pred[:, :-1, :]
    dx_g = gt[:, 1:, :] - gt[:, :-1, :]
    dy_p = pred[1:, :, :] - pred[:-1, :, :]
    dy_g = gt[1:, :, :] - gt[:-1, :, :]
    return 0.5 * (float(np.mean(np.abs(dx_p - dx_g))) + float(np.mean(np.abs(dy_p - dy_g))))


def resolve_eval_sequence_files(cfg: dict) -> list[str]:
    """Expand `paths.eval_sequence` into sorted frame paths (>= 2 frames required)."""
    paths_cfg = (cfg or {}).get("paths") or {}
    raw_seq = str(paths_cfg.get("eval_sequence") or "").strip()
    if not raw_seq:
        return []
    files = data.expand_pattern(raw_seq)
    return files if len(files) >= 2 else []


def select_contiguous_eval_windows(seq_files: list[str], max_frames: int = 60) -> list[list[str]]:
    """Select contiguous frame windows so 1st/2nd-order frame-to-frame flicker remains physically exact."""
    n = len(seq_files)
    if n < 2:
        return []
    if max_frames <= 0 or n <= max_frames:
        return [list(seq_files)]
    # Split into 3 contiguous temporal windows (start, middle, end) preserving 1-frame adjacency
    win_len = max(6, max_frames // 3)
    if win_len * 3 >= n:
        return [list(seq_files)]
    mid_start = max(win_len, (n - win_len) // 2)
    end_start = max(mid_start + win_len, n - win_len)
    w1 = list(seq_files[0:win_len])
    w2 = list(seq_files[mid_start : mid_start + win_len])
    w3 = list(seq_files[end_start:n])
    return [w for w in (w1, w2, w3) if len(w) >= 3]


def load_saved_recommendations(run_dir: str) -> dict | None:
    rec_path = os.path.join(run_dir, RECOMMENDATION_FILENAME)
    if not os.path.isfile(rec_path):
        return None
    try:
        with open(rec_path, "r", encoding="utf-8") as fh:
            data_obj = json.load(fh)
        if isinstance(data_obj, dict) and data_obj.get("available"):
            return data_obj
    except Exception:
        pass
    return None


class CheckpointRecommender:
    """Fast GPU streaming multi-metric checkpoint evaluator and auto-extend judge."""

    def __init__(
        self,
        run_dir: str,
        cfg: dict,
        device: str = None,
        src_full: list | None = None,
        tgt_full: list | None = None,
        max_seq_frames: int = 60,
        max_ckpts: int = 10,
    ):
        self.run_dir = os.path.abspath(run_dir)
        self.cfg = cfg or {}
        if device:
            self.device = str(device)
        elif torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        self.mo = dict(self.cfg.get("model") or {})
        self.c_in = int(self.mo.get("in_channels", 3))
        self.c_out = int(self.mo.get("out_channels", self.c_in))
        col = self.cfg.get("color") or {}
        self.working_cs = col.get("working_colorspace", "ACEScg")
        self.src_cs = col.get("source_colorspace", "auto")
        self.tgt_cs = col.get("target_colorspace", "auto")
        self.ocio_cfg = col.get("ocio_config", "")

        self.max_seq_frames = int(max_seq_frames)
        self.max_ckpts = max(4, int(max_ckpts))
        self.seq_all_files = resolve_eval_sequence_files(self.cfg)
        self.has_eval_sequence = len(self.seq_all_files) >= 2
        self.use_hfi = False  # kept for callers; `seq_hfi` no longer takes part in any decision

        self._src_full_init = src_full
        self._tgt_full_init = tgt_full
        self._prepared = False
        self._pair_records = []  # list of dicts with src_f32, tgt_f32, roi_mask, bg_mask
        self._seq_windows = []   # list of windows, each list of (src_f16, src_srgb_f16)
        self._eval_net = None

        self.rec_path = os.path.join(self.run_dir, RECOMMENDATION_FILENAME)
        self.evaluated_steps: dict[int, dict] = {}
        self._live_s1_history: list[dict] = []
        self._load_cached_steps()

    def _compute_stage_bounds(self, total_steps: int, initial_total_steps: int = 0) -> list[tuple[int, int]]:
        tr_cfg = self.cfg.get("train") or {}
        adv_cfg = advanced.parse_advanced_config(self.cfg)
        base_tot = int(initial_total_steps or tr_cfg.get("initial_total_steps") or total_steps or 1)
        return advanced.compute_stage_list(
            max(1, base_tot),
            bool(tr_cfg.get("multi_resolution", True)),
            adv_cfg["stage_schedule"],
        )

    @staticmethod
    def _scale_for_step(step: int, stage_bounds: list[tuple[int, int]] | None) -> int:
        if not stage_bounds:
            return 1
        st = int(step)
        for sc, b_end in stage_bounds:
            if st <= int(b_end):
                return int(sc)
        return 1

    def _load_cached_steps(self) -> None:
        if not os.path.isfile(self.rec_path):
            return
        try:
            with open(self.rec_path, "r", encoding="utf-8") as fh:
                prev = json.load(fh)
            raw_steps = (prev or {}).get("evaluated_steps") or {}
            for k, v in raw_steps.items():
                st = int(k)
                if st > 0 and isinstance(v, dict) and "psnr_aces" in v:
                    self.evaluated_steps[st] = v
        except Exception:
            pass

    def _ensure_prepared(self) -> None:
        if self._prepared:
            return
        paths_cfg = self.cfg.get("paths") or {}
        if self._src_full_init is not None and self._tgt_full_init is not None:
            s_frames = [_match_channels(np.asarray(a, dtype=np.float32), self.c_in) for a in self._src_full_init]
            t_frames = [_match_channels(np.asarray(b, dtype=np.float32), self.c_out) for b in self._tgt_full_init]
        else:
            s_files = data.expand_pattern(paths_cfg.get("source", "")) if paths_cfg.get("source") else []
            t_files = data.expand_pattern(paths_cfg.get("target", "")) if paths_cfg.get("target") else []
            n_p = min(len(s_files), len(t_files))
            s_frames = [
                _match_channels(
                    data._read(s_files[i], self.src_cs, self.working_cs, self.ocio_cfg, is_data_pass=(self.c_in == 1)),
                    self.c_in,
                )
                for i in range(n_p)
            ]
            t_frames = [
                _match_channels(
                    data._read(t_files[i], self.tgt_cs, self.working_cs, self.ocio_cfg, is_data_pass=(self.c_out == 1)),
                    self.c_out,
                )
                for i in range(n_p)
            ]

        self._pair_records = []
        for s_im, t_im in zip(s_frames, t_frames):
            t_srgb = color.acescg_to_display_srgb(t_im, src_space=self.working_cs, ocio_config_path=self.ocio_cfg)
            roi_mask = None
            bg_mask = None
            if self.c_out == 1:
                fg = t_im[..., 0] > 0.01
                if 0.0005 <= float(np.mean(fg)) <= 0.92:
                    roi_mask = fg
                bg = t_im[..., 0] < 1e-4
                if float(np.mean(bg)) >= 0.05:
                    bg_mask = bg
            elif s_im.shape == t_im.shape:
                diff_mag = np.max(np.abs(t_im - s_im), axis=2)
                fg = diff_mag >= 0.005
                if 0.0005 <= float(np.mean(fg)) <= 0.90:
                    roi_mask = fg
                bg = diff_mag < 1e-4
                if float(np.mean(bg)) >= 0.05:
                    bg_mask = bg
            elif self.c_in == 3 and self.c_out == 4:
                diff_rgb = np.max(np.abs(t_im[..., :3] - s_im[..., :3]), axis=2)
                alpha_edge = (t_im[..., 3] > 0.01) & (t_im[..., 3] < 0.99)
                fg = (diff_rgb >= 0.005) | alpha_edge
                if 0.0005 <= float(np.mean(fg)) <= 0.92:
                    roi_mask = fg
                bg = (diff_rgb < 1e-4) & (t_im[..., 3] < 1e-4)
                if float(np.mean(bg)) >= 0.05:
                    bg_mask = bg

            self._pair_records.append(
                {
                    "src_f16": s_im.astype(np.float16, copy=False),
                    "tgt_f32": t_im,
                    "tgt_srgb_f16": t_srgb.astype(np.float16, copy=False),
                    "roi_mask": roi_mask,
                    "bg_mask": bg_mask,
                }
            )

        self._seq_windows = []
        if self.has_eval_sequence:
            windows = select_contiguous_eval_windows(self.seq_all_files, max_frames=self.max_seq_frames)
            for win_files in windows:
                w_frames = []
                for fp in win_files:
                    s_im = _match_channels(
                        data._read(fp, self.src_cs, self.working_cs, self.ocio_cfg, is_data_pass=(self.c_in == 1)),
                        self.c_in,
                    )
                    s_srgb = color.acescg_to_display_srgb(
                        s_im, src_space=self.working_cs, ocio_config_path=self.ocio_cfg
                    )
                    w_frames.append(
                        (
                            s_im.astype(np.float16, copy=False),
                            s_srgb.astype(np.float16, copy=False),
                        )
                    )
                if len(w_frames) >= 2:
                    self._seq_windows.append(w_frames)

        self._prepared = True

    def discover_checkpoint_paths(self) -> dict[int, str]:
        """Return `{step: ckpt_path}` for saved checkpoints in `ckpt/`."""
        ckpt_dir = os.path.join(self.run_dir, "ckpt")
        found: dict[int, str] = {}
        if os.path.isdir(ckpt_dir):
            for p in glob.glob(os.path.join(ckpt_dir, "step_*.pt")):
                if p.endswith(".tmp"):
                    continue
                m = _STEP_PT_RE.search(os.path.basename(p))
                if m:
                    st = int(m.group(1))
                    if st > 0:
                        found[st] = p
            lat_p = os.path.join(ckpt_dir, "latest.pt")
            if os.path.isfile(lat_p):
                try:
                    sp = models.inspect_weight_source(lat_p)
                    st_lat = int(sp.get("step") or 0)
                    if st_lat > 0 and st_lat not in found:
                        found[st_lat] = lat_p
                except Exception:
                    pass
        if os.path.isdir(self.run_dir):
            for fn in os.listdir(self.run_dir):
                if fn.endswith(".tmp") or not fn.endswith(".pt"):
                    continue
                m_nuke = re.search(r"\.(\d+)\.pt$", fn)
                if m_nuke:
                    st_n = int(m_nuke.group(1))
                    if st_n > 0 and st_n not in found:
                        found[st_n] = os.path.join(self.run_dir, fn)
        return found

    def select_candidate_steps(self, available_steps: list[int], total_steps: int) -> list[int]:
        """Pick strategic milestone steps (stage transitions + mature curve + latest two) up to `max_ckpts`."""
        steps = sorted(set(int(s) for s in available_steps if int(s) > 0))
        if len(steps) <= self.max_ckpts:
            return steps

        must = set()
        # Always include latest 2 checkpoints (for slope check)
        must.add(steps[-1])
        must.add(steps[-2])

        # Include stage transition milestones (end of 1/4 and especially end of 1/2!)
        tr_cfg = self.cfg.get("train") or {}
        adv_cfg = advanced.parse_advanced_config(self.cfg)
        stage_bounds = advanced.compute_stage_list(
            max(total_steps, steps[-1]),
            bool(tr_cfg.get("multi_resolution", True)),
            adv_cfg["stage_schedule"],
        )
        for _, b_end in stage_bounds:
            nearest = min(steps, key=lambda s: abs(s - b_end))
            must.add(nearest)

        # Prefer checkpoints in [30%..100%] of run, spaced evenly
        mature = [s for s in steps if s >= steps[-1] * 0.30] or steps
        rem_slots = max(2, self.max_ckpts - len(must))
        if len(mature) <= rem_slots:
            must.update(mature)
        else:
            for i in range(rem_slots):
                idx = int(round(i * (len(mature) - 1) / float(max(1, rem_slots - 1))))
                must.add(mature[idx])

        # Also keep any already-cached steps at zero extra GPU cost
        must.update(s for s in steps if s in self.evaluated_steps)
        return sorted(must)

    def _load_net_for_step(self, ckpt_path: str = None, state_dict: dict = None):
        if state_dict is not None:
            if self._eval_net is None:
                net, _ = models.build_generator(self.mo, cudnn_pad=True)
                self._eval_net = net.to(self.device).eval()
            self._eval_net.load_state_dict(state_dict, strict=True)
            return self._eval_net

        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(ck, dict) and "G" in ck:
            ck_mo = (ck.get("config") or {}).get("model") or self.mo
            if self._eval_net is None:
                net, _ = models.build_generator(ck_mo, cudnn_pad=True)
                self._eval_net = net.to(self.device).eval()
            self._eval_net.load_state_dict(ck["G"], strict=True)
            return self._eval_net
        jit_net = torch.jit.load(ckpt_path, map_location=self.device)
        jit_net.eval()
        return jit_net

    @torch.no_grad()
    def evaluate_step(self, step: int, ckpt_path: str = None, state_dict: dict = None) -> dict:
        st = int(step)
        if st in self.evaluated_steps and state_dict is None:
            return self.evaluated_steps[st]

        self._ensure_prepared()
        tf32_mm_prev = torch.backends.cuda.matmul.allow_tf32
        tf32_cd_prev = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            net = self._load_net_for_step(ckpt_path=ckpt_path, state_dict=state_dict)

            def _infer(src_f16: np.ndarray) -> np.ndarray:
                x = (
                    torch.from_numpy(np.ascontiguousarray(src_f16, dtype=np.float32))
                    .permute(2, 0, 1)[None]
                    .to(self.device)
                )
                out = infer_image(net, x, amp_dtype=None, channels_last=False)
                return _match_channels(out[0].permute(1, 2, 0).cpu().numpy(), self.c_out)

            # 1. Paired Keyframe Fidelity & Edit-ROI / Untouched-BG Metrics
            psnr_aces_list = []
            psnr_srgb_list = []
            roi_psnr_list = []
            edge_l1_list = []
            bg_drift_list = []

            for rec in self._pair_records:
                p_im = _infer(rec["src_f16"])
                t_im = rec["tgt_f32"]
                diff_lin = p_im - t_im
                mse_lin = float(np.mean(diff_lin * diff_lin))
                psnr_aces = float(10.0 * math.log10(1.0 / max(mse_lin, 1e-12)))
                psnr_aces_list.append(psnr_aces)

                p_srgb = color.acescg_to_display_srgb(
                    p_im, src_space=self.working_cs, ocio_config_path=self.ocio_cfg
                )
                t_srgb = rec["tgt_srgb_f16"].astype(np.float32)
                mse_srgb = float(np.mean((p_srgb - t_srgb) ** 2))
                psnr_srgb_list.append(float(10.0 * math.log10(1.0 / max(mse_srgb, 1e-12))))

                if rec["roi_mask"] is not None:
                    roi_diff = diff_lin[rec["roi_mask"]]
                    roi_mse = float(np.mean(roi_diff * roi_diff))
                    roi_psnr_list.append(float(10.0 * math.log10(1.0 / max(roi_mse, 1e-12))))
                else:
                    roi_psnr_list.append(psnr_aces)

                edge_l1_list.append(_edge_l1(p_im, t_im))
                if rec["bg_mask"] is not None:
                    pix_l1 = np.mean(np.abs(diff_lin), axis=2)
                    bg_drift_list.append(float(np.mean(pix_l1[rec["bg_mask"]])))

            # 2. Continuous Validation Sequence Metrics (seq_hfi = high-frequency change index, not a flicker measure)
            delta_jitters = []
            hf_chatters = []
            static_boils = []
            edit_mags = []
            eval_frame_total = 0

            for win in self._seq_windows:
                prev_s_srgb = None
                prev_out_sig = None
                prev_delta = None
                prev2_delta = None
                for s_f16, s_srgb_f16 in win:
                    eval_frame_total += 1
                    p_im = _infer(s_f16)
                    s_srgb = s_srgb_f16.astype(np.float32)
                    p_srgb = color.acescg_to_display_srgb(
                        p_im, src_space=self.working_cs, ocio_config_path=self.ocio_cfg
                    )
                    out_sig = (
                        np.concatenate([p_srgb, np.clip(p_im[..., 3:4], 0.0, 1.0)], axis=2)
                        if self.c_out == 4
                        else (np.clip(p_im[..., :1], 0.0, 1.0) if self.c_out == 1 else p_srgb)
                    )
                    if self.c_in == self.c_out:
                        if self.c_out == 4:
                            a_diff = np.clip(p_im[..., 3:4], 0.0, 1.0) - np.clip(
                                s_f16[..., 3:4].astype(np.float32), 0.0, 1.0
                            )
                            cur_delta = np.concatenate([p_srgb - s_srgb, a_diff], axis=2)
                        else:
                            cur_delta = p_srgb - s_srgb
                    elif self.c_out == 1:
                        cur_delta = np.clip(p_im[..., :1], 0.0, 1.0)
                    elif self.c_in == 3 and self.c_out == 4:
                        cur_delta = np.concatenate(
                            [p_srgb - s_srgb, np.clip(p_im[..., 3:4], 0.0, 1.0)], axis=2
                        )
                    else:
                        cur_delta = p_srgb

                    edit_mags.append(float(np.mean(np.abs(cur_delta))))
                    if prev_delta is not None:
                        delta_jitters.append(float(np.mean(np.abs(cur_delta - prev_delta))))
                        plate_motion = np.max(np.abs(s_srgb - prev_s_srgb), axis=2)
                        static_mask = plate_motion < 0.005
                        if float(np.mean(static_mask)) >= 0.02:
                            out_diff = np.mean(np.abs(out_sig - prev_out_sig), axis=2)
                            static_boils.append(float(np.mean(out_diff[static_mask])))
                    if prev2_delta is not None and prev_delta is not None:
                        hf_chatters.append(
                            float(np.mean(np.abs(cur_delta - 2.0 * prev_delta + prev2_delta)))
                        )
                    prev2_delta = prev_delta
                    prev_delta = cur_delta
                    prev_s_srgb = s_srgb
                    prev_out_sig = out_sig

            metrics = {
                "step": st,
                "psnr_aces": round(float(np.mean(psnr_aces_list)), 4) if psnr_aces_list else 0.0,
                "psnr_srgb": round(float(np.mean(psnr_srgb_list)), 4) if psnr_srgb_list else 0.0,
                "roi_psnr": round(float(np.mean(roi_psnr_list)), 4) if roi_psnr_list else 0.0,
                "edge_l1": round(float(np.mean(edge_l1_list)), 6) if edge_l1_list else 0.0,
                "bg_drift_mae": round(float(np.mean(bg_drift_list)), 6) if bg_drift_list else None,
                "seq_frames_evaluated": eval_frame_total,
                "seq_edit_l1": round(float(np.mean(edit_mags)), 6) if edit_mags else 0.0,
                "seq_delta": round(float(np.mean(delta_jitters)), 6) if delta_jitters else None,
                "seq_hfi": round(float(np.mean(hf_chatters)), 6) if hf_chatters else (
                    round(float(np.mean(delta_jitters)), 6) if delta_jitters else None
                ),
                "seq_static": round(float(np.mean(static_boils)), 6) if static_boils else None,
            }
            self.evaluated_steps[st] = metrics
            return metrics
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32_mm_prev
            torch.backends.cudnn.allow_tf32 = tf32_cd_prev

    @staticmethod
    def compute_live_score(
        metrics: dict,
        c_out: int = 3,
        s1_history: list[dict] | None = None,
        use_hfi: bool = HFI_IN_DECISIONS_DEFAULT,
    ) -> float:
        """dB-scale score of one stage-1/1 checkpoint for the live `stop_after_optimal_epochs` tracking.

        It follows the fit to the training keyframes (plus, when available, how little the untouched background /
        static areas move) and is lowered when a matte loses its foreground.  There is no "Seen PSNR is saturated"
        rule any more (2026-11 re-validation: it locked steps that were 1.5 dB worse on unseen frames on average)
        and `seq_hfi` is not used (`use_hfi` is accepted and ignored)."""
        if not metrics:
            return -1e9
        s_srgb = float(metrics.get("psnr_srgb") or metrics.get("roi_psnr") or 0.0)
        roi_p = float(metrics.get("roi_psnr") or 0.0)
        aces_p = float(metrics.get("psnr_aces") or 0.0)
        edge_e = float(metrics.get("edge_l1") or 0.0)
        fid_raw = 0.65 * roi_p + 0.35 * aces_p - 8.0 * edge_e

        hist = list(s1_history or [])
        edits = [float(m.get("seq_edit_l1") or 0.0) for m in hist if (m.get("seq_edit_l1") or 0.0) > 0]
        cur_edit = float(metrics.get("seq_edit_l1") or 0.0)
        if cur_edit > 0:
            edits.append(cur_edit)
        ref_edit = max(float(np.median(edits)), edits[0]) if edits else 0.0
        matte_collapse_pen = 0.0
        if ref_edit > 1e-4 and cur_edit < 0.85 * ref_edit:
            matte_collapse_pen = 12.0 * ((0.85 * ref_edit - cur_edit) / ref_edit)

        stb_raw = metrics.get("bg_drift_mae") if metrics.get("bg_drift_mae") is not None else metrics.get("seq_static")
        if metrics.get("seq_hfi") is not None:
            # a validation sequence was evaluated: display-domain keyframe PSNR + stability (weights as before)
            stb_db = -20.0 * math.log10(max(float(stb_raw), 1e-6)) if stb_raw is not None else 0.0
            return round(0.45 * s_srgb + 0.15 * stb_db - matte_collapse_pen, 4)
        # no validation sequence: keyframe fidelity (+ untouched-background stability when there is a background mask)
        if stb_raw is not None:
            stb_db = -20.0 * math.log10(max(float(stb_raw), 1e-6))
            return round(0.80 * fid_raw + 0.20 * stb_db - matte_collapse_pen, 4)
        return round(fid_raw - matte_collapse_pen, 4)

    def live_score_step(self, metrics: dict) -> float:
        """Instance-aware Stage-1/1 live score tracker for `stop_after_optimal_epochs`."""
        if not metrics:
            return -1e9
        sc = self.compute_live_score(
            metrics, c_out=self.c_out, s1_history=self._live_s1_history, use_hfi=self.use_hfi
        )
        st = int(metrics.get("step") or 0)
        if not self._live_s1_history or int(self._live_s1_history[-1].get("step") or 0) != st:
            self._live_s1_history.append(dict(metrics))
            self._live_s1_history.sort(key=lambda x: int(x.get("step") or 0))
        return sc

    @staticmethod
    def _score_evaluated_rows(
        rows: list[dict],
        epochs: int,
        total_steps: int,
        c_out: int = 3,
        stage_bounds: list[tuple[int, int]] | None = None,
        use_hfi: bool = HFI_IN_DECISIONS_DEFAULT,
    ) -> tuple[list[dict], dict]:
        """Annotate evaluated checkpoints and pick `balanced` and `detail`.

        `balanced` = the latest stage-1/1 checkpoint (2026-11 re-validation: the final step was the best evaluated
        step, or within 0.36 dB of it, on every dataset; rules that preferred an earlier step cost up to 5 dB).
        `detail`   = the stage-1/1 checkpoint with the best fit of the edit region on the training keyframes.
        A checkpoint whose matte lost its foreground (edit magnitude below 85 % of the stage-1/1 reference) is never
        picked.  `composite_score` is kept for display only.  `use_hfi` is accepted and ignored."""
        if not rows:
            return [], {}
        ordered = sorted((dict(r) for r in rows), key=lambda x: int(x["step"]))
        for r in ordered:
            if "stage_scale" not in r or not r["stage_scale"]:
                r["stage_scale"] = CheckpointRecommender._scale_for_step(int(r["step"]), stage_bounds)

        s1_rows = [r for r in ordered if int(r.get("stage_scale") or 1) == 1]
        pool = s1_rows if s1_rows else ordered
        has_seq = any(r.get("seq_hfi") is not None for r in pool)

        s1_edits = [float(r.get("seq_edit_l1") or 0.0) for r in pool if (r.get("seq_edit_l1") or 0.0) > 0]
        ref_edit = max(float(np.median(s1_edits)), s1_edits[0]) if s1_edits else 0.0

        for r in ordered:
            ep_est = int(round((r["step"] * epochs) / float(total_steps))) if (epochs > 0 and total_steps > 0) else 0
            r["epoch"] = ep_est
            fid_db = 0.65 * float(r["roi_psnr"]) + 0.35 * float(r["psnr_aces"])
            s_srgb = float(r.get("psnr_srgb") or fid_db)
            # Display-only score (same formulas as before, without the removed HFI and "Seen saturated" branches)
            if has_seq and r.get("seq_hfi") is not None:
                stb_raw = r.get("bg_drift_mae") if r.get("bg_drift_mae") is not None else r.get("seq_static")
                stb_db = -20.0 * math.log10(max(float(stb_raw), 1e-6)) if stb_raw is not None else 0.0
                prog_bonus = 0.35 * (float(r["step"]) / float(max(1, total_steps)))
                comp = 0.65 * s_srgb + 0.10 * stb_db + prog_bonus
            else:
                stb_raw = r.get("bg_drift_mae")
                stb_db = -20.0 * math.log10(max(float(stb_raw), 1e-6)) if (stb_raw is not None and float(stb_raw) > 0) else s_srgb
                comp = 0.85 * s_srgb + 0.15 * stb_db

            r["fid_db"] = round(fid_db, 4)
            r["composite_score"] = round(comp, 4)
            in_s1_or_fallback = (int(r.get("stage_scale") or 1) == 1) or not s1_rows
            edit_ok = ref_edit <= 1e-4 or float(r.get("seq_edit_l1") or 0.0) >= 0.85 * ref_edit
            r["qualified"] = bool(in_s1_or_fallback and edit_ok)

        qualified = [r for r in ordered if r["qualified"]] or pool
        best_balanced = max(qualified, key=lambda r: int(r["step"]))
        best_detail = max(
            qualified,
            key=lambda r: (float(r["roi_psnr"]) - 12.0 * float(r.get("edge_l1") or 0.0), int(r["step"])),
        )

        def _pack_choice(role: str, row: dict) -> dict:
            return {
                "role": role,
                "step": int(row["step"]),
                "epoch": int(row.get("epoch") or 0),
                "stage_scale": int(row.get("stage_scale") or 1),
                "roi_psnr": row["roi_psnr"],
                "psnr_aces": row["psnr_aces"],
                "psnr_srgb": row["psnr_srgb"],
                "edge_l1": row["edge_l1"],
                "bg_drift_mae": row.get("bg_drift_mae"),
                "seq_edit_l1": row.get("seq_edit_l1"),
                "seq_hfi": row.get("seq_hfi"),
                "seq_delta": row.get("seq_delta"),
                "seq_static": row.get("seq_static"),
                "composite_score": row["composite_score"],
            }

        picks = {"balanced": _pack_choice("balanced", best_balanced), "detail": _pack_choice("detail", best_detail)}
        return ordered, picks

    def evaluate_and_summarize(
        self,
        current_step: int = None,
        current_state_dict: dict = None,
        epochs: int = 0,
        total_steps: int = 0,
        initial_epochs: int = 0,
        initial_total_steps: int = 0,
        auto_extended_rounds: int = 0,
        stop_reason: str = "",
        stopped_after_optimal: bool = False,
        stop_after_optimal_epochs: int = 0,
        optimal_step: int = 0,
        persist: bool = True,
    ) -> dict:
        t0 = time.time()
        ckpt_map = self.discover_checkpoint_paths()
        avail = list(ckpt_map.keys())
        if current_step and current_step > 0 and current_step not in ckpt_map and current_state_dict is not None:
            avail.append(int(current_step))

        eff_total = max([int(total_steps or 0)] + avail + [1])
        eff_epochs = int(epochs or (self.cfg.get("train") or {}).get("epochs") or 0)
        cand_steps = self.select_candidate_steps(avail, eff_total)
        stage_bounds = self._compute_stage_bounds(eff_total, initial_total_steps=initial_total_steps)

        rows = []
        for st in cand_steps:
            if current_step and st == int(current_step) and current_state_dict is not None and st not in self.evaluated_steps:
                m = self.evaluate_step(st, state_dict=current_state_dict)
            elif st in self.evaluated_steps:
                m = self.evaluated_steps[st]
            elif st in ckpt_map:
                m = self.evaluate_step(st, ckpt_path=ckpt_map[st])
            else:
                continue
            rows.append(m)

        scored_rows, picks = self._score_evaluated_rows(
            rows,
            eff_epochs,
            eff_total,
            c_out=self.c_out,
            stage_bounds=stage_bounds,
            use_hfi=self.use_hfi,
        )
        summary = {
            "available": bool(picks and self.has_eval_sequence),
            "has_eval_sequence": bool(self.has_eval_sequence),
            "hfi_in_decisions": bool(self.use_hfi),
            "seq_total_frames": len(self.seq_all_files),
            "seq_evaluated_frames": int(scored_rows[-1]["seq_frames_evaluated"]) if scored_rows else 0,
            "evaluated_checkpoint_count": len(scored_rows),
            "initial_epochs": int(initial_epochs or eff_epochs),
            "final_epochs": eff_epochs,
            "total_steps": eff_total,
            "auto_extended": int(auto_extended_rounds) > 0,
            "auto_extended_rounds": int(auto_extended_rounds),
            "stopped_after_optimal": bool(stopped_after_optimal),
            "stop_after_optimal_epochs": int(stop_after_optimal_epochs or 0),
            "optimal_step": int(optimal_step or 0),
            "stop_reason": str(stop_reason or ""),
            "elapsed_sec": round(time.time() - t0, 2),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "picks": picks,
            "evaluated_steps": {str(r["step"]): r for r in scored_rows},
        }
        if persist and scored_rows:
            _atomic_write_json(self.rec_path, summary)
        return summary

    def decide_auto_extend(
        self,
        summary: dict,
        initial_epochs: int,
        initial_total_steps: int,
        current_epochs: int,
        current_total_steps: int,
        auto_extend_rounds: int,
        max_epochs_cap: int = 0,
        max_rounds: int = 4,
        max_mult: float = 2.5,
        stop_after_optimal_epochs: int = 0,
        best_opt_step: int = 0,
    ) -> tuple[bool, str, int]:
        """Decide whether to add another chunk of full-resolution training after the requested epochs.
        Returns `(should_extend, reason_code, next_target_epochs)`.

        The only quality criterion left is "the fit to the training keyframes still improves by >= 0.20 dB per
        chunk".  It cannot tell whether unseen frames benefit: on the re-validation records it would have asked
        for more training in nearly every run, and where a longer reference run exists that helped unseen frames
        in 15 of 18 runs and hurt in 3.  The caps (`max_rounds`, `max_mult`, the user's epoch cap) bound the cost."""
        if not self.has_eval_sequence and int(stop_after_optimal_epochs or 0) <= 0:
            return False, "no_eval_sequence", current_epochs
        if max_epochs_cap > 0 and current_epochs >= max_epochs_cap:
            return False, "user_epoch_cap_reached", current_epochs
        ceil_epochs = max(
            initial_epochs + 1,
            int(round(initial_epochs * float(max_mult))),
            initial_epochs + int(stop_after_optimal_epochs or 0) * 2,
        )
        if max_epochs_cap > 0:
            ceil_epochs = min(ceil_epochs, int(max_epochs_cap))
        if auto_extend_rounds >= max_rounds or current_epochs >= ceil_epochs:
            return False, "max_extension_reached", current_epochs

        # When Best-Combo `stop_after_optimal_epochs` is active and a recent optimum happened near the end of budget,
        # extend enough epochs to verify whether `+stop_after_optimal_epochs` passes without a better peak.
        stop_ep = int(stop_after_optimal_epochs or 0)
        if stop_ep > 0 and best_opt_step > 0 and current_total_steps > 0:
            best_ep_est = int(round((best_opt_step * current_epochs) / float(max(1, current_total_steps))))
            elapsed_ep = max(0, current_epochs - best_ep_est)
            if elapsed_ep < stop_ep:
                rem_ep = max(1, stop_ep - elapsed_ep)
                next_ep_pat = min(ceil_epochs, current_epochs + max(rem_ep, int(round(initial_epochs * 0.25))))
                if next_ep_pat > current_epochs:
                    return True, f"confirming_optimal_window({elapsed_ep}/{stop_ep}ep)", next_ep_pat

        if not self.has_eval_sequence:
            return False, "optimal_window_confirmed", current_epochs

        ev_map = (summary or {}).get("evaluated_steps") or {}
        rows = sorted(ev_map.values(), key=lambda r: int(r["step"]))
        picks = (summary or {}).get("picks") or {}
        if not rows:
            return False, "no_checkpoints", current_epochs

        chunk_ep = max(1, int(round(initial_epochs * 0.30)))
        next_epochs = min(ceil_epochs, current_epochs + chunk_ep)
        if next_epochs <= current_epochs:
            return False, "at_ceiling", current_epochs

        if len(rows) < 2:
            # Only 1 checkpoint saved across the entire run -> extend 1 chunk so slope/optimum can be verified
            return True, "single_checkpoint_need_slope", next_epochs

        last_r = rows[-1]
        s_last = int(last_r["step"])
        s1_rows = [r for r in rows if int(r.get("stage_scale") or 1) == 1] or rows

        # 0. Guard against Roto / matte foreground collapse (an almost empty mask would only look "stable")
        s1_edits = [float(r.get("seq_edit_l1") or 0.0) for r in s1_rows if (r.get("seq_edit_l1") or 0.0) > 0]
        ref_edit = max(float(np.median(s1_edits)), s1_edits[0]) if s1_edits else 0.0
        if ref_edit > 1e-4 and float(last_r.get("seq_edit_l1") or 0.0) < 0.88 * ref_edit:
            return False, "matte_collapse_guard", current_epochs

        # 1. The recommended checkpoint is the latest stage-1/1 step unless that step failed the collapse filter;
        #    if an earlier step was recommended, more training is not asked for.
        bal_pick = picks.get("balanced") or {}
        if int(bal_pick.get("step") or s_last) < s_last:
            return False, "interior_optimum_captured", current_epochs

        # 2. Improvement of the fit to the training keyframes over the latest mature stage-1/1 interval, normalised
        #    to one extension chunk.  (The "Seen PSNR is saturated" veto and the `seq_hfi` rules were removed in the
        #    2026-11 re-validation; see the module docstring.)
        ref_pool = s1_rows[:-1] if len(s1_rows) >= 2 else rows[:-1]
        ref_candidates = [r for r in ref_pool if int(r["step"]) <= int(s_last * 0.88)]
        ref_r = ref_candidates[-1] if ref_candidates else ref_pool[-1]
        step_span = max(1, s_last - int(ref_r["step"]))
        norm_scale = 0.25 / max(0.08, step_span / float(max(1, initial_total_steps)))
        fid_gain_per_chunk = (float(last_r.get("fid_db") or 0.0) - float(ref_r.get("fid_db") or 0.0)) * norm_scale

        # 3. Fidelity plateau on the training keyframes -> stop extending
        if fid_gain_per_chunk < 0.20:
            return False, "convergence_plateau_captured", current_epochs

        return True, f"still_improving(dFid={fid_gain_per_chunk:+.2f}dB)", next_epochs

    def release(self) -> None:
        self._eval_net = None
        self._pair_records = []
        self._seq_windows = []
        self._prepared = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main_cli(argv: list[str] = None) -> int:
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="Evaluate WildCat checkpoints on validation sequence and write recommended_models.json")
    ap.add_argument("run_dir", help="Path to training run directory")
    ap.add_argument("config_path", help="Path to config JSON")
    ap.add_argument("--eval-seq", default="", help="Optional override for paths.eval_sequence")
    ap.add_argument("--force", action="store_true", help="Clear cached evaluated_steps before evaluating")
    args = ap.parse_args(argv)

    with open(args.config_path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    if args.eval_seq:
        cfg.setdefault("paths", {})["eval_sequence"] = args.eval_seq

    rec_file = os.path.join(args.run_dir, RECOMMENDATION_FILENAME)
    prev_meta = {}
    if os.path.isfile(rec_file):
        try:
            with open(rec_file, "r", encoding="utf-8") as fh:
                prev_meta = json.load(fh) or {}
        except Exception:
            prev_meta = {}
        if args.force:
            try:
                os.remove(rec_file)
            except OSError:
                pass

    info_p = os.path.join(args.run_dir, "run_info.json")
    rinfo = {}
    if os.path.isfile(info_p):
        try:
            with open(info_p, "r", encoding="utf-8") as fh:
                rinfo = json.load(fh) or {}
        except Exception:
            rinfo = {}

    epochs = int(rinfo.get("epochs") or (cfg.get("train") or {}).get("epochs") or 0)
    total_steps = int(rinfo.get("total_steps") or (cfg.get("train") or {}).get("total_steps") or 0)
    init_epochs = int(rinfo.get("initial_epochs") or prev_meta.get("initial_epochs") or epochs)
    ext_rounds = int(rinfo.get("auto_extended_rounds") or prev_meta.get("auto_extended_rounds") or 0)
    stopped_after_opt = bool(rinfo.get("stopped_after_optimal") or prev_meta.get("stopped_after_optimal") or False)
    stop_after_opt_ep = int(
        rinfo.get("stop_after_optimal_epochs")
        or prev_meta.get("stop_after_optimal_epochs")
        or (cfg.get("train") or {}).get("stop_after_optimal_epochs")
        or 0
    )
    opt_step = int(rinfo.get("optimal_step") or prev_meta.get("optimal_step") or 0)

    rec = CheckpointRecommender(args.run_dir, cfg)
    try:
        summary = rec.evaluate_and_summarize(
            epochs=epochs,
            total_steps=total_steps,
            initial_epochs=init_epochs,
            auto_extended_rounds=ext_rounds,
            stop_reason=str(prev_meta.get("stop_reason") or "manual_eval"),
            stopped_after_optimal=stopped_after_opt,
            stop_after_optimal_epochs=stop_after_opt_ep,
            optimal_step=opt_step,
            persist=True,
        )
    finally:
        rec.release()

    sys.stdout.write(json.dumps({"available": summary.get("available", False), "count": summary.get("evaluated_checkpoint_count", 0)}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
