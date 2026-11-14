# -*- coding: utf-8 -*-
"""Empirical GPU Speed Benchmark Database Reader & Stage-Aware ETA Estimator.

Reads the standalone open-source SQLite database `data/gpu_benchmarks.db`:
  1. Exact empirical lookup + fitted Roofline curves (`tau_launch_ms + tau_pix_ms * B^b * Area^a`)
     for empirically tested GPUs (RTX 5090, RTX 5090 D, RTX 4090, A100-PCIE-40GB).
  2. Hardware-anchored priors (`launch_time_mult_vs_4090`, `pix_time_mult_vs_4090`) regressed from
     tested GPUs for 40+ untested NVIDIA GPU models.
  3. VRAM-tier heuristic fallback relative to tested GPUs for completely unlisted hardware.
"""
import json
import os
import sqlite3
import threading
from typing import Any

from . import db

GPU_BENCH_DB_PATH = os.path.join(db.ROOT, "data", "gpu_benchmarks.db")

_LOCK = threading.RLock()
_CACHE_SIG: tuple[int, int] | None = None
_CACHE_DATA: dict[str, Any] | None = None


def _fallback_curves() -> dict[str, dict[str, Any]]:
    return {
        "rtx_4090": {
            "Small": {
                "tau_launch_ms": 1.65, "tau_pix_ms": 1.18, "batch_exp": 0.90, "area_exp": 0.98,
                "adv_loss_mult": 1.065, "adv_ema_mult": 1.015, "adv_aug_mult": 1.015,
                "ref_engine_mult": 2.15, "warmup_sec": 7.5,
                "sps_1_4_b8_c256": 450.0, "sps_1_2_b8_c256": 265.0, "sps_1_1_b8_c256": 105.0,
            },
            "Medium": {
                "tau_launch_ms": 2.50, "tau_pix_ms": 2.22, "batch_exp": 0.92, "area_exp": 1.00,
                "adv_loss_mult": 1.065, "adv_ema_mult": 1.015, "adv_aug_mult": 1.015,
                "ref_engine_mult": 2.15, "warmup_sec": 7.5,
                "sps_1_4_b8_c256": 295.0, "sps_1_2_b8_c256": 158.0, "sps_1_1_b8_c256": 62.0,
            },
            "Large": {
                "tau_launch_ms": 5.40, "tau_pix_ms": 7.20, "batch_exp": 0.94, "area_exp": 1.00,
                "adv_loss_mult": 1.065, "adv_ema_mult": 1.015, "adv_aug_mult": 1.015,
                "ref_engine_mult": 2.15, "warmup_sec": 11.0,
                "sps_1_4_b8_c256": 134.0, "sps_1_2_b8_c256": 70.0, "sps_1_1_b8_c256": 26.0,
            },
        }
    }


def _load_db_cached() -> dict[str, Any]:
    global _CACHE_SIG, _CACHE_DATA
    try:
        st = os.stat(GPU_BENCH_DB_PATH)
        sig = (st.st_mtime_ns, st.st_size)
    except OSError:
        sig = None

    with _LOCK:
        if sig is not None and _CACHE_DATA is not None and _CACHE_SIG == sig:
            return _CACHE_DATA

    if sig is None:
        fallback = {
            "meta": {"schema_version": "0", "baseline_untested_anchor": "rtx_4090"},
            "tested_gpus": {},
            "curves": _fallback_curves(),
            "buckets": {},
            "priors": [],
        }
        with _LOCK:
            _CACHE_SIG = None
            _CACHE_DATA = fallback
        return fallback

    conn = sqlite3.connect(f"file:{GPU_BENCH_DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        meta = {r["key"]: r["value"] for r in conn.execute("SELECT key, value FROM meta").fetchall()}

        tested_gpus: dict[str, dict[str, Any]] = {}
        for r in conn.execute("SELECT * FROM tested_gpus ORDER BY order_priority ASC").fetchall():
            d = dict(r)
            d["match_patterns"] = json.loads(d.pop("match_patterns_json") or "[]")
            tested_gpus[d["gpu_key"]] = d

        curves: dict[str, dict[str, dict[str, Any]]] = {}
        for r in conn.execute("SELECT * FROM gpu_roofline_curves").fetchall():
            d = dict(r)
            curves.setdefault(d["gpu_key"], {})[d["model_size"]] = d

        buckets: dict[str, dict[str, float]] = {}
        for r in conn.execute("SELECT * FROM gpu_empirical_buckets").fetchall():
            gk = str(r["gpu_key"])
            bkey = f"{r['model_size']}|{int(r['crop_size'])}|{int(r['batch_size'])}|{int(r['adv_loss'])}|{int(r['scale'])}"
            buckets.setdefault(gk, {})[bkey] = float(r["median_sps"])

        priors: list[dict[str, Any]] = []
        for r in conn.execute("SELECT * FROM untested_gpu_priors ORDER BY order_priority ASC").fetchall():
            d = dict(r)
            d["match_patterns"] = json.loads(d.pop("match_patterns_json") or "[]")
            priors.append(d)

        if "rtx_4090" not in curves:
            curves.update(_fallback_curves())

        loaded = {
            "meta": meta,
            "tested_gpus": tested_gpus,
            "curves": curves,
            "buckets": buckets,
            "priors": priors,
        }
    finally:
        conn.close()

    with _LOCK:
        _CACHE_SIG = sig
        _CACHE_DATA = loaded
    return loaded


def _scale_anchor_curves(
    anchor_curves: dict[str, dict[str, Any]],
    launch_mult: float,
    pix_mult: float,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for ms, c in anchor_curves.items():
        nc = dict(c)
        nc["tau_launch_ms"] = round(float(c["tau_launch_ms"]) * launch_mult, 4)
        nc["tau_pix_ms"] = round(float(c["tau_pix_ms"]) * pix_mult, 4)
        out[ms] = nc
    return out


def resolve_gpu_profile(raw_gpu_name: str, vram_mb: int = 0) -> dict[str, Any]:
    """Match a GPU device name against empirically tested GPUs first, then hardware priors, then VRAM fallback."""
    data = _load_db_cached()
    tested = data["tested_gpus"]
    curves_map = data["curves"]
    buckets_map = data["buckets"]
    priors = data["priors"]
    anchor_key = str(data["meta"].get("baseline_untested_anchor") or "rtx_4090")
    anchor_curves = curves_map.get(anchor_key) or next(iter(curves_map.values()))

    norm = str(raw_gpu_name or "").strip().lower()
    is_laptop = any(w in norm for w in ("laptop", "mobile", "max-q"))

    # 1. Check specific untested variants that share digits with desktop tested flagships (e.g. Laptop / 4090 D / A100 SXM 80GB)
    if norm:
        for pr in priors:
            if int(pr.get("order_priority", 999)) < 45 or is_laptop or "80gb" in norm or "sxm" in norm:
                if any(pat in norm for pat in pr["match_patterns"]):
                    l_mult = float(pr["launch_time_mult_vs_4090"])
                    p_mult = float(pr["pix_time_mult_vs_4090"])
                    return {
                        "is_tested": False,
                        "estimation_mode": "prior_from_tested",
                        "matched_key": pr["prior_key"],
                        "matched_name": pr["display_name"],
                        "raw_name": raw_gpu_name or pr["display_name"],
                        "vram_gb": int(pr["vram_gb"]),
                        "curves": _scale_anchor_curves(anchor_curves, l_mult, p_mult),
                        "buckets": {},
                    }

        # 2. Match empirically tested GPUs (ordered by priority: RTX 5090 D -> RTX 5090 -> RTX 4090 -> A100-PCIE-40GB)
        for gk, gspec in sorted(tested.items(), key=lambda kv: int(kv[1].get("order_priority", 100))):
            if is_laptop and gk in ("rtx_5090", "rtx_5090_d", "rtx_4090"):
                continue
            if any(pat in norm for pat in gspec["match_patterns"]):
                return {
                    "is_tested": True,
                    "estimation_mode": "empirical_tested",
                    "matched_key": gk,
                    "matched_name": gspec["short_name"],
                    "canonical_name": gspec["canonical_name"],
                    "raw_name": raw_gpu_name or gspec["canonical_name"],
                    "vram_gb": int(gspec["vram_gb"]),
                    "tested_runs": int(gspec["tested_runs"]),
                    "tested_epochs": int(gspec["tested_epochs"]),
                    "curves": curves_map.get(gk, anchor_curves),
                    "buckets": buckets_map.get(gk, {}),
                }

        # 3. Match remaining hardware-anchored untested GPU priors
        for pr in priors:
            if any(pat in norm for pat in pr["match_patterns"]):
                l_mult = float(pr["launch_time_mult_vs_4090"])
                p_mult = float(pr["pix_time_mult_vs_4090"])
                return {
                    "is_tested": False,
                    "estimation_mode": "prior_from_tested",
                    "matched_key": pr["prior_key"],
                    "matched_name": pr["display_name"],
                    "raw_name": raw_gpu_name or pr["display_name"],
                    "vram_gb": int(pr["vram_gb"]),
                    "curves": _scale_anchor_curves(anchor_curves, l_mult, p_mult),
                    "buckets": {},
                }

    # 4. Unknown GPU model: estimate conservatively from VRAM tier relative to tested RTX 4090
    vram_gb = round(float(vram_mb) / 1024.0) if vram_mb and vram_mb > 0 else 16
    if vram_gb >= 32:
        l_mult, p_mult = 1.15, 1.05
    elif vram_gb >= 22:
        l_mult, p_mult = 1.15, 1.35
    elif vram_gb >= 15:
        l_mult, p_mult = 1.18, 1.75
    elif vram_gb >= 11:
        l_mult, p_mult = 1.25, 2.35
    else:
        l_mult, p_mult = 1.35, 3.20

    return {
        "is_tested": False,
        "estimation_mode": "vram_fallback_from_tested",
        "matched_key": "unknown_gpu",
        "matched_name": raw_gpu_name or f"GPU ({vram_gb} GB)",
        "raw_name": raw_gpu_name or "Unknown GPU",
        "vram_gb": vram_gb,
        "curves": _scale_anchor_curves(anchor_curves, l_mult, p_mult),
        "buckets": {},
    }


def eval_profile_stage_sps(profile: dict[str, Any], cfg: dict[str, Any] | None) -> tuple[float, float, float, float]:
    """Compute (sps_1_4, sps_1_2, sps_1_1, warmup_sec) for a resolved GPU profile and training config."""
    cfg = cfg or {}
    mo = cfg.get("model") or {}
    tr = cfg.get("train") or {}
    opts = cfg.get("optimizations") or {}
    adv = cfg.get("advanced") or {}

    msize = str(mo.get("model_size") or "Medium")
    if msize not in ("Small", "Medium", "Large"):
        msize = "Medium"
    bs = max(1, int(tr.get("batch_size") or 8))
    crop = max(32, int(tr.get("crop_size") or 256))

    aloss = 1 if bool((adv.get("loss_family") or {}).get("enabled", False)) else 0
    aema = bool((adv.get("ema_lr") or {}).get("enabled", False))
    aaug = bool((adv.get("augment") or adv.get("augmentation") or {}).get("enabled", False))
    is_ref_engine = str(opts.get("engine") or "fast") == "reference"

    curves = profile.get("curves") or {}
    c = curves.get(msize) or curves.get("Medium") or next(iter(curves.values()))
    buckets = profile.get("buckets") or {}

    tl = float(c.get("tau_launch_ms", 2.5))
    tp = float(c.get("tau_pix_ms", 2.2))
    bexp = float(c.get("batch_exp", 0.90))
    aexp = float(c.get("area_exp", 0.98))
    lmult = float(c.get("adv_loss_mult", 1.065)) if aloss else 1.0
    emult = float(c.get("adv_ema_mult", 1.015)) if aema else 1.0
    gmult = float(c.get("adv_aug_mult", 1.015)) if aaug else 1.0
    rmult = float(c.get("ref_engine_mult", 2.15)) if is_ref_engine else 1.0
    warmup_sec = float(c.get("warmup_sec", 7.5))

    out_sps: list[float] = []
    for sc in (4, 2, 1):
        spatial = ((float(crop) / float(sc)) / 256.0) ** (2.0 * aexp)
        t_ms = (tl + tp * (float(bs) ** bexp) * spatial) * lmult
        curve_sps = 1000.0 / max(0.2, t_ms)

        # Blend 75% exact empirical median + 25% smooth curve if this exact bucket was measured on this tested GPU
        bkey = f"{msize}|{crop}|{bs}|{aloss}|{sc}"
        b_hit = buckets.get(bkey)
        if b_hit is not None and b_hit > 0:
            sps = 0.75 * float(b_hit) + 0.25 * curve_sps
        else:
            sps = curve_sps

        sps = sps / (emult * gmult * rmult)
        out_sps.append(max(1.0, round(sps, 2)))

    return (out_sps[0], out_sps[1], out_sps[2], warmup_sec)


def predict_stage_sps(
    cfg: dict[str, Any] | None,
    gpu_name: str = "",
    vram_mb: int = 0,
) -> tuple[float, float, float, float, dict[str, Any]]:
    prof = resolve_gpu_profile(gpu_name, vram_mb=vram_mb)
    b4, b2, b1, wsec = eval_profile_stage_sps(prof, cfg)
    return b4, b2, b1, wsec, prof


def get_frontend_speed_payload(gpus_meta: list[tuple[str, str, int]]) -> dict[str, Any]:
    """Build compact JSON payload for `/api/v1/gpu-benchmarks` used by Studio ETA & Help Speed Catalog."""
    data = _load_db_cached()
    detected = []
    for gid, raw_label, vram_mb in (gpus_meta or []):
        # raw_label from scheduler._detect_gpus_meta may look like "GPU 0: NVIDIA GeForce RTX 4090 (24 GB)"
        clean_name = str(raw_label or "")
        if ":" in clean_name:
            clean_name = clean_name.split(":", 1)[1].strip()
        prof = resolve_gpu_profile(clean_name, vram_mb=int(vram_mb or 0))
        detected.append({
            "gpu_id": str(gid),
            "label": raw_label,
            "raw_name": clean_name,
            "vram_mb": int(vram_mb or 0),
            "is_tested": bool(prof["is_tested"]),
            "estimation_mode": prof["estimation_mode"],
            "matched_key": prof["matched_key"],
            "matched_name": prof["matched_name"],
            "curves": prof["curves"],
            "buckets": prof["buckets"],
        })

    if not detected:
        fb = resolve_gpu_profile("", vram_mb=0)
        detected.append({
            "gpu_id": "0",
            "label": fb["matched_name"],
            "raw_name": fb["raw_name"],
            "vram_mb": 0,
            "is_tested": False,
            "estimation_mode": fb["estimation_mode"],
            "matched_key": fb["matched_key"],
            "matched_name": fb["matched_name"],
            "curves": fb["curves"],
            "buckets": {},
        })

    tested_catalog = []
    for gk, gspec in sorted(data["tested_gpus"].items(), key=lambda kv: int(kv[1].get("order_priority", 100))):
        gc = data["curves"].get(gk, {})
        models_summary = {}
        for ms in ("Small", "Medium", "Large"):
            c = gc.get(ms)
            if c:
                models_summary[ms] = {
                    "sps_1_4": c["sps_1_4_b8_c256"],
                    "sps_1_2": c["sps_1_2_b8_c256"],
                    "sps_1_1": c["sps_1_1_b8_c256"],
                    "nuke_1_4": c.get("nuke_sps_1_4"),
                    "nuke_1_2": c.get("nuke_sps_1_2"),
                    "nuke_1_1": c.get("nuke_sps_1_1"),
                }
        tested_catalog.append({
            "gpu_key": gk,
            "canonical_name": gspec["canonical_name"],
            "short_name": gspec["short_name"],
            "architecture": gspec["architecture"],
            "vram_gb": gspec["vram_gb"],
            "tested_cards": gspec["tested_cards"],
            "tested_runs": gspec["tested_runs"],
            "tested_epochs": gspec["tested_epochs"],
            "tested_gpu_hours": gspec["tested_gpu_hours"],
            "models": models_summary,
        })

    return {
        "meta": data["meta"],
        "untested_priors_count": len(data["priors"]),
        "detected_gpus": detected,
        "tested_catalog": tested_catalog,
    }
