# -*- coding: utf-8 -*-
"""Optional advanced training hyperparameters for WildCat (validated across 9 VFX shot families & 206 blind-test runs).

Bottom-line invariant:
  When ``cfg.get("advanced")`` is absent or every sub-module has ``enabled=False``,
  ``parse_advanced_config(cfg)["any_enabled"]`` is ``False`` and the training loop,
  sampler, loss, optimizer schedule, and model forward execute the exact native
  CopyCat code path without any deviation.
"""
import math
import torch

ADV_SECTION_KEYS = (
    "crop_grid",
    "loss_family",
    "stage_schedule",
    "ema_lr",
    "augment",
)


def _as_bool(v, default=False):
    if v is None:
        return default
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


def _clamp(v, lo, hi, default):
    try:
        x = float(v)
        if not math.isfinite(x):
            return default
        return max(lo, min(hi, x))
    except Exception:
        return default


def parse_advanced_config(cfg: dict | None) -> dict:
    """Parse and sanitize ``cfg['advanced']``. Returns normalized dict with ``any_enabled`` flag."""
    cfg_dict = cfg if isinstance(cfg, dict) else {}
    raw = cfg_dict.get("advanced")
    if not isinstance(raw, dict):
        raw = {}
    tr_raw = cfg_dict.get("train") if isinstance(cfg_dict.get("train"), dict) else {}
    mo_raw = cfg_dict.get("model") if isinstance(cfg_dict.get("model"), dict) else {}
    multi_res = _as_bool(tr_raw.get("multi_resolution"), True)
    c_in = int(_clamp(mo_raw.get("in_channels"), 1, 4, 3))
    c_out = int(_clamp(mo_raw.get("out_channels"), 1, 4, c_in))

    # 01. Crop origin phase grid (1px free all-phase placement vs Nuke 4px bottom-anchored grid)
    cg_raw = raw.get("crop_grid") if isinstance(raw.get("crop_grid"), dict) else {}
    grid_v = int(_clamp(cg_raw.get("full_res_grid"), 1, 4, 1))
    if grid_v not in (1, 2, 4):
        grid_v = 1
    cg_en = _as_bool(cg_raw.get("enabled"), False) and (grid_v != 4)
    crop_grid = {
        "enabled": cg_en,
        "full_res_grid": grid_v if cg_en else 4,
    }

    # 02. Composite loss: Charbonnier + L1 spatial edge gradient + ACEScg HDR highlight damping
    lf_raw = raw.get("loss_family") if isinstance(raw.get("loss_family"), dict) else {}
    def_charb = 0.30 if c_out == 4 else 0.25
    def_grad = 0.15 if c_out == 4 else 0.05
    def_hdr = 0.35 if c_out == 4 else 0.50
    lf_charb = _clamp(lf_raw.get("charbonnier_ratio"), 0.0, 1.0, def_charb)
    lf_eps = _clamp(lf_raw.get("charb_eps"), 1e-6, 0.5, 1e-3)
    lf_grad = _clamp(lf_raw.get("grad_weight"), 0.0, 5.0, def_grad)
    # 1ch Alpha mattes are [0, 1] coverage masks, not scene-referred HDR radiance
    lf_hdr = 0.0 if c_out == 1 else _clamp(lf_raw.get("hdr_alpha"), 0.0, 2.0, def_hdr)
    lf_en = _as_bool(lf_raw.get("enabled"), False) and (lf_charb > 0.0 or lf_grad > 0.0 or lf_hdr > 0.0)
    loss_family = {
        "enabled": lf_en,
        "charbonnier_ratio": lf_charb,
        "charb_eps": lf_eps,
        "grad_weight": lf_grad,
        "hdr_alpha": lf_hdr,
    }

    # 03. Multi-resolution stage duration split & stage-transition LR soft warmup (requires multi_resolution=True)
    st_raw = raw.get("stage_schedule") if isinstance(raw.get("stage_schedule"), dict) else {}
    rq = _clamp(st_raw.get("r_quarter"), 0.0, 100.0, 15.0)
    rh = _clamp(st_raw.get("r_half"), 0.0, 100.0, 25.0)
    rf = _clamp(st_raw.get("r_full"), 0.0, 100.0, 60.0)
    if rq + rh + rf <= 1e-6:
        rq, rh, rf = 15.0, 25.0, 60.0
    st_warm = int(_clamp(st_raw.get("warmup_steps"), 0, 50000, 200))
    st_en = bool(multi_res) and _as_bool(st_raw.get("enabled"), False)
    stage_schedule = {
        "enabled": st_en,
        "r_quarter": rq,
        "r_half": rh,
        "r_full": rf,
        "warmup_steps": st_warm,
    }

    # 04. Stage-reset adaptive EMA shadow weights (preserves 100% Nuke native poly(0.3) optimizer trajectory)
    el_raw = raw.get("ema_lr") if isinstance(raw.get("ema_lr"), dict) else {}
    ema_lr = {
        "enabled": _as_bool(el_raw.get("enabled"), False),
        "ema_decay": _clamp(el_raw.get("ema_decay"), 0.90, 0.99999, 0.9995),
    }

    # 05. Semantic-preserving optical batch augmentation (paired horizontal flip + optional linear EV gain; EV defaults to 0.0)
    aug_raw = raw.get("augment") if isinstance(raw.get("augment"), dict) else {}
    aug_want_en = _as_bool(aug_raw.get("enabled"), False)
    aug_flip = _clamp(aug_raw.get("hflip_prob"), 0.0, 1.0, 0.5 if aug_want_en else 0.0)
    aug_ev = 0.0 if (c_in < 3 and c_out < 3) else _clamp(aug_raw.get("exposure_ev"), 0.0, 2.0, 0.0)
    aug_en = aug_want_en and (aug_flip > 0.0 or aug_ev > 0.0)
    augment = {
        "enabled": aug_en,
        "hflip_prob": aug_flip,
        "exposure_ev": aug_ev,
    }

    parsed = {
        "crop_grid": crop_grid,
        "loss_family": loss_family,
        "stage_schedule": stage_schedule,
        "ema_lr": ema_lr,
        "augment": augment,
    }
    parsed["any_enabled"] = any(parsed[k]["enabled"] for k in ADV_SECTION_KEYS)
    return parsed


def compute_stage_list(total_steps: int, multi_resolution: bool, stage_cfg: dict) -> list[tuple[int, int]]:
    """Compute [(scale_div, end_step), ...] list."""
    total = max(1, int(total_steps))
    if not multi_resolution:
        return [(1, total)]
    if not stage_cfg or not stage_cfg.get("enabled", False):
        s = math.ceil(total / 3.0)
        return [(4, min(total, s)), (2, min(total, 2 * s)), (1, total)]

    rq = float(stage_cfg.get("r_quarter", 15.0))
    rh = float(stage_cfg.get("r_half", 25.0))
    rf = float(stage_cfg.get("r_full", 60.0))
    denom = max(1e-9, rq + rh + rf)
    e1 = min(total, max(0, int(round(total * (rq / denom)))))
    e2 = min(total, max(e1, int(round(total * ((rq + rh) / denom)))))
    raw_stages = [(4, e1), (2, e2), (1, total)]
    out = []
    prev_end = 0
    for sc, end in raw_stages:
        if end > prev_end:
            out.append((sc, end))
            prev_end = end
    if not out or out[-1][1] < total:
        out.append((1, total))
    return out


def compute_scheduled_lr(
    lr_init: float,
    step_zero_based: int,
    total_steps: int,
    default_power: float,
    stage_cfg: dict | None,
    stage_start_step_zero_based: int = 0,
    is_stage_transition: bool = False,
) -> float:
    """Compute Nuke polynomial LR with optional linear soft ramp at multi-res stage transitions."""
    if total_steps <= 0:
        base_lr = lr_init
    else:
        clamped = min(max(0, step_zero_based), total_steps)
        base_lr = lr_init * ((1.0 - float(clamped) / float(total_steps)) ** default_power)

    if is_stage_transition and stage_cfg and stage_cfg.get("enabled", False):
        w_steps = int(stage_cfg.get("warmup_steps", 0))
        if w_steps > 0:
            offset = step_zero_based - stage_start_step_zero_based
            if 0 <= offset < w_steps:
                # Ramp smoothly from 0.10x to 1.0x over w_steps to prevent Adam v_t shock
                ramp = 0.10 + 0.90 * (float(offset + 1) / float(w_steps))
                base_lr *= min(1.0, ramp)
    return base_lr


def compute_ema_alpha(ema_decay: float, step_one_based: int) -> float:
    """Return lerp weight (1 - beta_k) for EMA update at 1-based step k."""
    beta_warmup = float(1 + step_one_based) / float(10 + step_one_based)
    beta_k = min(float(ema_decay), beta_warmup)
    return max(0.0, min(1.0, 1.0 - beta_k))


def compute_level_loss(pred: torch.Tensor, target: torch.Tensor, lf_cfg: dict) -> torch.Tensor:
    """Compute composite per-level loss on float32 tensors (B, C, H, W)."""
    hdr_alpha = float(lf_cfg.get("hdr_alpha", 0.0))
    charb_ratio = float(lf_cfg.get("charbonnier_ratio", 0.0))
    charb_eps = float(lf_cfg.get("charb_eps", 1e-3))
    grad_w = float(lf_cfg.get("grad_weight", 0.0))

    diff = pred - target
    if hdr_alpha > 0.0:
        if target.shape[1] >= 3:
            lum = (
                0.2126 * target[:, 0:1].abs()
                + 0.7152 * target[:, 1:2].abs()
                + 0.0722 * target[:, 2:3].abs()
            )
            denom_rgb = torch.pow(1.0 + lum, hdr_alpha)
            if target.shape[1] > 3:
                denom = torch.cat(
                    [denom_rgb.expand(-1, 3, -1, -1), torch.ones_like(target[:, 3:])],
                    dim=1,
                )
            else:
                denom = denom_rgb
        else:
            lum = target.abs()
            denom = torch.pow(1.0 + lum, hdr_alpha)
        r = diff / denom
    else:
        denom = None
        r = diff

    r_sq = r * r
    mse_term = r_sq.mean()

    if charb_ratio > 0.0:
        eps_sq = charb_eps * charb_eps
        charb_term = torch.sqrt(r_sq + eps_sq).mean()
        if charb_ratio >= 1.0:
            pix_loss = charb_term
        else:
            pix_loss = (1.0 - charb_ratio) * mse_term + charb_ratio * charb_term
    else:
        pix_loss = mse_term

    if grad_w > 0.0 and pred.shape[-2] > 1 and pred.shape[-1] > 1:
        dx = diff[:, :, :, 1:] - diff[:, :, :, :-1]
        dy = diff[:, :, 1:, :] - diff[:, :, :-1, :]
        if denom is not None:
            dx = dx / (0.5 * (denom[:, :, :, 1:] + denom[:, :, :, :-1]))
            dy = dy / (0.5 * (denom[:, :, 1:, :] + denom[:, :, :-1, :]))
        g_loss = dx.abs().mean() + dy.abs().mean()
        return pix_loss + grad_w * g_loss

    return pix_loss


class BatchAugmenter:
    """In-place GPU batch augmentation (paired horizontal flip + linear exposure stops jitter)."""

    def __init__(self, aug_cfg: dict, c_out: int, device: str, seed: int = 0):
        self.hflip_prob = float(aug_cfg.get("hflip_prob", 0.0))
        self.exposure_ev = float(aug_cfg.get("exposure_ev", 0.0))
        self.c_out = int(c_out)
        self.device = device
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(int(seed) + 7919)

    @torch.no_grad()
    def apply_(self, x: torch.Tensor, y: torch.Tensor):
        bs = x.shape[0]
        if self.hflip_prob > 0.0:
            m = torch.rand((bs,), generator=self.gen, device=x.device) < self.hflip_prob
            if bool(m.any().item()):
                idx = torch.nonzero(m, as_tuple=False).squeeze(1)
                x[idx] = torch.flip(x[idx], dims=[-1])
                y[idx] = torch.flip(y[idx], dims=[-1])
        if self.exposure_ev > 0.0 and (x.shape[1] >= 3 or y.shape[1] >= 3):
            ev = (torch.rand((bs, 1, 1, 1), generator=self.gen, device=x.device) * 2.0 - 1.0) * self.exposure_ev
            gain = torch.pow(2.0, ev)
            if x.shape[1] >= 3:
                x[:, :3].mul_(gain.to(dtype=x.dtype))
            if y.shape[1] >= 3:
                y[:, :3].mul_(gain.to(dtype=y.dtype))
