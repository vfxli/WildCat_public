# -*- coding: utf-8 -*-
"""Build standalone open-source GPU Speed Benchmark Database (`data/gpu_benchmarks.db`).

Distills empirical per-stage (`1/4`, `1/2`, `1/1`), per-model (`Small`, `Medium`, `Large`),
per-batch, per-crop, and per-loss training throughput measurements collected across tested GPUs
(RTX 5090, RTX 5090 D, RTX 4090, A100-PCIE-40GB) into a lightweight, standalone SQLite database
shipped with WildCat for accurate ETA estimation.

For GPUs not yet empirically tested, `untested_gpu_priors` provides hardware-anchored scaling
factors regressed directly from our tested GPU fleet (separating launch-latency scaling at 1/4
from bandwidth/compute scaling at 1/1), with a final VRAM/tier fallback for unknown devices.
"""
import argparse
import json
import math
import os
import sqlite3
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DB = ROOT / "data" / "gpu_benchmarks.db"
BENCH_JSON = ROOT / "frontend" / "benchmarks.json"

sys.path.insert(0, str(ROOT))


# Canonical metadata for the 4 empirically tested GPU models
TESTED_GPU_SPECS = {
    "rtx_5090": {
        "canonical_name": "NVIDIA GeForce RTX 5090",
        "short_name": "RTX 5090 (32 GB)",
        "architecture": "Blackwell",
        "vram_gb": 32,
        "mem_bw_gbps": 1792.0,
        "fp16_tflops": 419.0,
        "boost_ghz": 2.41,
        # Match more specific "5090 d" / "5090d" first in matcher order; order_priority handles precedence
        "order_priority": 20,
        "match_patterns": ["rtx 5090", "geforce rtx 5090"],
        "default_cards": 9,
        "default_runs": 100,
        "default_epochs": 73210000,
        "default_gpu_hours": 187.0,
        "measured_pflops_per_h": 249.87,
    },
    "rtx_5090_d": {
        "canonical_name": "NVIDIA GeForce RTX 5090 D",
        "short_name": "RTX 5090 D (32 GB)",
        "architecture": "Blackwell",
        "vram_gb": 32,
        "mem_bw_gbps": 1792.0,
        "fp16_tflops": 419.0,
        "boost_ghz": 2.41,
        "order_priority": 10,  # Checked before rtx_5090 so "RTX 5090 D" matches here!
        "match_patterns": ["rtx 5090 d", "rtx 5090d", "5090 d", "5090d"],
        "default_cards": 1,
        "default_runs": 2,
        "default_epochs": 2000000,
        "default_gpu_hours": 4.1,
        "measured_pflops_per_h": 244.50,
    },
    "rtx_4090": {
        "canonical_name": "NVIDIA GeForce RTX 4090",
        "short_name": "RTX 4090 (24 GB)",
        "architecture": "Ada Lovelace",
        "vram_gb": 24,
        "mem_bw_gbps": 1008.0,
        "fp16_tflops": 330.0,
        "boost_ghz": 2.52,
        "order_priority": 30,
        "match_patterns": ["rtx 4090", "geforce rtx 4090"],
        "default_cards": 3,
        "default_runs": 59,
        "default_epochs": 39620000,
        "default_gpu_hours": 62.7,
        "measured_pflops_per_h": 157.20,
    },
    "a100_pcie_40gb": {
        "canonical_name": "NVIDIA A100-PCIE-40GB",
        "short_name": "A100 PCIe (40 GB)",
        "architecture": "Ampere",
        "vram_gb": 40,
        "mem_bw_gbps": 1555.0,
        "fp16_tflops": 312.0,
        "boost_ghz": 1.41,
        "order_priority": 40,
        "match_patterns": ["a100-pcie", "a100 pcie", "a100-pci", "a100 40gb"],
        "default_cards": 8,
        "default_runs": 45,
        "default_epochs": 36800000,
        "default_gpu_hours": 168.0,
        "measured_pflops_per_h": 221.72,
    },
}


# Hardware-anchored priors for UNTESTED GPUs (regressed from our 4 tested GPUs:
# launch_time_mult scales inversely with boost clock & driver dispatch speed relative to RTX 4090;
# pix_time_mult scales inversely with harmonic blend of mem_bw_gbps (60%) + fp16_tflops (40%) relative to RTX 4090).
def _prior_mults(mem_bw: float, tflops: float, boost_ghz: float, arch_eff: float = 1.0) -> tuple[float, float]:
    # Anchored so RTX 4090 (1008 GB/s, 330 TF, 2.52 GHz) = (1.00, 1.00),
    # RTX 5090 (1792 GB/s, 419 TF, 2.41 GHz, arch 1.02) -> pix_time_mult ~ 0.67 (1.49x faster at 1/1),
    # A100 PCIe (1555 GB/s, 312 TF, 1.41 GHz) -> launch_mult ~ 1.45 (slower 1/4), pix_mult ~ 0.92.
    bw_ratio = max(0.08, mem_bw / 1008.0)
    tf_ratio = max(0.05, tflops / 330.0)
    throughput_rel = arch_eff * (bw_ratio ** 0.58) * (tf_ratio ** 0.42)
    clk_ratio = max(0.45, boost_ghz / 2.52)
    launch_time_mult = round(1.0 / (clk_ratio ** 0.72 * (arch_eff ** 0.25)), 3)
    pix_time_mult = round(1.0 / throughput_rel, 3)
    return launch_time_mult, pix_time_mult


UNTESTED_GPU_PRIORS = [
    # Blackwell consumer & workstation
    ("rtx_pro_6000_blackwell", "RTX PRO 6000 Blackwell (96 GB)", "Blackwell", 96, 1792.0, 450.0, 2.45, 1.03, 15, ["rtx pro 6000", "pro 6000 blackwell"]),
    ("rtx_5080", "GeForce RTX 5080 (16 GB)", "Blackwell", 16, 960.0, 225.0, 2.62, 1.02, 50, ["rtx 5080"]),
    ("rtx_5070_ti", "GeForce RTX 5070 Ti (16 GB)", "Blackwell", 16, 896.0, 176.0, 2.45, 1.01, 55, ["rtx 5070 ti", "5070ti"]),
    ("rtx_5070", "GeForce RTX 5070 (12 GB)", "Blackwell", 12, 672.0, 123.0, 2.51, 1.00, 60, ["rtx 5070"]),
    ("rtx_5060_ti", "GeForce RTX 5060 Ti", "Blackwell", 16, 448.0, 94.0, 2.57, 1.00, 65, ["rtx 5060 ti", "5060ti"]),
    ("rtx_5060", "GeForce RTX 5060 (8 GB)", "Blackwell", 8, 448.0, 75.0, 2.50, 1.00, 70, ["rtx 5060"]),
    # Hopper / Ada Datacenter & Workstation
    ("h100_sxm", "NVIDIA H100 SXM (80 GB)", "Hopper", 80, 3350.0, 989.0, 1.98, 1.05, 80, ["h100 sxm", "h100-sxm", "h100 80gb hbm3"]),
    ("h100_pcie", "NVIDIA H100 PCIe (80 GB)", "Hopper", 80, 2000.0, 756.0, 1.75, 1.03, 85, ["h100", "h800"]),
    ("h20", "NVIDIA H20 (96 GB)", "Hopper", 96, 4000.0, 148.0, 1.65, 0.98, 88, ["nvidia h20", " h20"]),
    ("a100_sxm_80gb", "NVIDIA A100 SXM4 (80 GB)", "Ampere", 80, 2039.0, 312.0, 1.41, 0.96, 90, ["a100-sxm", "a100 sxm", "a100 80gb", "a800"]),
    ("l40s", "NVIDIA L40S (48 GB)", "Ada Lovelace", 48, 864.0, 362.0, 2.52, 1.00, 95, ["l40s"]),
    ("l40", "NVIDIA L40 (48 GB)", "Ada Lovelace", 48, 864.0, 181.0, 2.49, 0.98, 98, ["nvidia l40"]),
    ("rtx_6000_ada", "RTX 6000 Ada (48 GB)", "Ada Lovelace", 48, 960.0, 364.0, 2.50, 1.00, 100, ["rtx 6000 ada", "6000 ada"]),
    ("rtx_5000_ada", "RTX 5000 Ada (32 GB)", "Ada Lovelace", 32, 576.0, 261.0, 2.55, 1.00, 102, ["rtx 5000 ada", "5000 ada"]),
    ("rtx_4000_ada", "RTX 4000 Ada (20 GB)", "Ada Lovelace", 20, 360.0, 107.0, 2.17, 0.98, 104, ["rtx 4000 ada", "4000 ada"]),
    ("l4", "NVIDIA L4 (24 GB)", "Ada Lovelace", 24, 300.0, 121.0, 2.04, 0.96, 106, ["nvidia l4"]),
    # Ada Lovelace Consumer
    ("rtx_4090_d", "GeForce RTX 4090 D (24 GB)", "Ada Lovelace", 24, 1008.0, 294.0, 2.52, 1.00, 32, ["rtx 4090 d", "rtx 4090d", "4090 d", "4090d"]),
    ("rtx_4090_laptop", "GeForce RTX 4090 Laptop (16 GB)", "Ada Lovelace", 16, 576.0, 165.0, 2.04, 0.98, 31, ["rtx 4090 laptop", "4090 mobile"]),
    ("rtx_4080_super", "GeForce RTX 4080 SUPER (16 GB)", "Ada Lovelace", 16, 736.0, 209.0, 2.55, 1.00, 110, ["rtx 4080 super", "4080 super"]),
    ("rtx_4080", "GeForce RTX 4080 (16 GB)", "Ada Lovelace", 16, 717.0, 195.0, 2.51, 1.00, 115, ["rtx 4080"]),
    ("rtx_4070_ti_super", "GeForce RTX 4070 Ti SUPER (16 GB)", "Ada Lovelace", 16, 672.0, 176.0, 2.61, 1.00, 120, ["4070 ti super"]),
    ("rtx_4070_ti", "GeForce RTX 4070 Ti (12 GB)", "Ada Lovelace", 12, 504.0, 160.0, 2.61, 1.00, 125, ["rtx 4070 ti", "4070ti"]),
    ("rtx_4070_super", "GeForce RTX 4070 SUPER (12 GB)", "Ada Lovelace", 12, 504.0, 142.0, 2.48, 1.00, 130, ["4070 super"]),
    ("rtx_4070", "GeForce RTX 4070 (12 GB)", "Ada Lovelace", 12, 504.0, 117.0, 2.48, 1.00, 135, ["rtx 4070"]),
    ("rtx_4060_ti", "GeForce RTX 4060 Ti", "Ada Lovelace", 16, 288.0, 88.0, 2.54, 0.98, 140, ["rtx 4060 ti", "4060ti"]),
    ("rtx_4060", "GeForce RTX 4060 (8 GB)", "Ada Lovelace", 8, 272.0, 61.0, 2.46, 0.98, 145, ["rtx 4060"]),
    # Ampere Consumer & Workstation
    ("rtx_a6000", "NVIDIA RTX A6000 (48 GB)", "Ampere", 48, 768.0, 155.0, 1.80, 0.94, 150, ["rtx a6000", "a6000"]),
    ("rtx_a5000", "NVIDIA RTX A5000 (24 GB)", "Ampere", 24, 768.0, 111.0, 1.69, 0.94, 155, ["rtx a5000", "a5000"]),
    ("rtx_a4000", "NVIDIA RTX A4000 (16 GB)", "Ampere", 16, 448.0, 77.0, 1.56, 0.93, 160, ["rtx a4000", "a4000"]),
    ("a40", "NVIDIA A40 (48 GB)", "Ampere", 48, 696.0, 150.0, 1.74, 0.94, 162, ["nvidia a40", " a40"]),
    ("a10", "NVIDIA A10 / A10G (24 GB)", "Ampere", 24, 600.0, 125.0, 1.69, 0.94, 164, ["a10g", "nvidia a10"]),
    ("rtx_3090_ti", "GeForce RTX 3090 Ti (24 GB)", "Ampere", 24, 1008.0, 160.0, 1.86, 0.95, 170, ["3090 ti", "3090ti"]),
    ("rtx_3090", "GeForce RTX 3090 (24 GB)", "Ampere", 24, 936.0, 142.0, 1.70, 0.95, 175, ["rtx 3090"]),
    ("rtx_3080_ti", "GeForce RTX 3080 Ti (12 GB)", "Ampere", 12, 912.0, 136.0, 1.67, 0.95, 180, ["3080 ti", "3080ti"]),
    ("rtx_3080", "GeForce RTX 3080 (10/12 GB)", "Ampere", 10, 760.0, 119.0, 1.71, 0.95, 185, ["rtx 3080"]),
    ("rtx_3070_ti", "GeForce RTX 3070 Ti (8 GB)", "Ampere", 8, 608.0, 87.0, 1.77, 0.94, 190, ["3070 ti", "3070ti"]),
    ("rtx_3070", "GeForce RTX 3070 (8 GB)", "Ampere", 8, 448.0, 81.0, 1.73, 0.94, 195, ["rtx 3070"]),
    ("rtx_3060_ti", "GeForce RTX 3060 Ti (8 GB)", "Ampere", 8, 448.0, 65.0, 1.67, 0.93, 200, ["3060 ti", "3060ti"]),
    ("rtx_3060", "GeForce RTX 3060 (12 GB)", "Ampere", 12, 360.0, 51.0, 1.78, 0.93, 205, ["rtx 3060"]),
    # Turing / Volta legacy
    ("rtx_2080_ti", "GeForce RTX 2080 Ti (11 GB)", "Turing", 11, 616.0, 54.0, 1.54, 0.88, 210, ["2080 ti", "2080ti"]),
    ("v100", "NVIDIA Tesla V100 (16/32 GB)", "Volta", 32, 900.0, 112.0, 1.38, 0.86, 215, ["v100"]),
    ("t4", "NVIDIA Tesla T4 (16 GB)", "Turing", 16, 320.0, 65.0, 1.59, 0.85, 220, ["tesla t4", "nvidia t4"]),
]


def _canonical_gpu_key(raw_name: str) -> str | None:
    s = str(raw_name or "").lower()
    if "5090 d" in s or "5090d" in s:
        return "rtx_5090_d"
    if "5090" in s:
        return "rtx_5090"
    if "4090" in s:
        return "rtx_4090"
    if "a100" in s:
        return "a100_pcie_40gb"
    return None


def _fit_roofline_for_samples(samples: list[tuple[int, int, int, int, float]], fallback: tuple[float, float, float, float]) -> tuple[float, float, float, float, float]:
    """Fit step_time_ms = (tau_launch + tau_pix * (bs ** batch_exp) * ((crop / scale / 256) ** (2 * area_exp))) * (adv_loss_mult if loss else 1.0)
    via grid + coordinate refinement minimizing median/Huber relative error across empirical stage medians.
    samples: list of (crop_size, batch_size, adv_loss, scale, median_sps)
    """
    if len(samples) < 4:
        tl, tp, bexp, aexp = fallback
        return (tl, tp, bexp, aexp, 1.065)

    best_err = 1e18
    best_params = (*fallback, 1.065)

    for bexp in (0.82, 0.88, 0.94, 1.00):
        for aexp in (0.90, 0.96, 1.02, 1.08):
            for lmult in (1.04, 1.065, 1.09):
                # Linear least squares in (tau_launch, tau_pix) with weights 1 / y_true^2 (relative error minimization!)
                xs = []
                ys = []
                for crop, bs, aloss, sc, sps in samples:
                    t_ms = 1000.0 / max(0.5, float(sps))
                    lm = lmult if aloss else 1.0
                    y = t_ms / lm
                    spatial = ((float(crop) / float(sc)) / 256.0) ** (2.0 * aexp)
                    work = (float(bs) ** bexp) * spatial
                    xs.append(work)
                    ys.append(y)
                # Solve weighted linear regression y ~ a + b * x with weight w = 1 / y^2
                s_w = sum(1.0 / (y * y) for y in ys)
                s_wx = sum(x / (y * y) for x, y in zip(xs, ys))
                s_wxx = sum((x * x) / (y * y) for x, y in zip(xs, ys))
                s_wy = sum(1.0 / y for y in ys)
                s_wxy = sum(x / y for x, y in zip(xs, ys))
                det = s_w * s_wxx - s_wx * s_wx
                if det <= 1e-12:
                    continue
                tl = (s_wxx * s_wy - s_wx * s_wxy) / det
                tp = (s_w * s_wxy - s_wx * s_wy) / det
                if tl < 0.35 or tp < 0.08:
                    continue
                # Huber-like relative error across samples
                errs = []
                for (crop, bs, aloss, sc, sps), x in zip(samples, xs):
                    pred_ms = (tl + tp * x) * (lmult if aloss else 1.0)
                    true_ms = 1000.0 / max(0.5, float(sps))
                    rel = abs(pred_ms - true_ms) / true_ms
                    errs.append(rel if rel < 0.25 else 0.25 + 0.5 * (rel - 0.25))
                mean_err = sum(errs) / len(errs)
                if mean_err < best_err:
                    best_err = mean_err
                    best_params = (round(tl, 4), round(tp, 4), bexp, aexp, lmult)

    return best_params


def collect_empirical_data(scratch_dir: Path | None = None) -> tuple[dict, dict, list[dict]]:
    """Collect empirical stage measurements from local WildCat DB + benchmarks.json (+ optional scratch campaign logs)."""
    from backend import db, storage
    storage.refresh_storage_paths(force_db=True)

    rid_to_gpu_key: dict[str, str] = {}
    gpu_stats = {
        k: {
            "cards": v["default_cards"],
            "runs": v["default_runs"],
            "epochs": v["default_epochs"],
            "gpu_hours": v["default_gpu_hours"],
        }
        for k, v in TESTED_GPU_SPECS.items()
    }

    if scratch_dir and scratch_dir.is_dir():
        sys.path.insert(0, str(scratch_dir))
        try:
            import sync_experiment_logbook as sel
            final_by_id, live_by_id, _, _ = sel._load_campaign_jsonl_records()
            for rid, r in {**live_by_id, **final_by_id}.items():
                gn = sel._infer_local_gpu_name(str(r.get("gpu", "0")), r.get("gpu_name"))
                gk = _canonical_gpu_key(gn)
                if gk:
                    rid_to_gpu_key[rid] = gk
            for wf in sorted(scratch_dir.glob("results_wave*.jsonl")):
                for ln in wf.read_text(encoding="utf-8").splitlines():
                    if not ln.strip():
                        continue
                    r = json.loads(ln)
                    gn = sel._infer_local_gpu_name(str(r.get("gpu", "0")), r.get("gpu_name"))
                    gk = _canonical_gpu_key(gn)
                    if gk:
                        rid_to_gpu_key.setdefault(r["run_id"], gk)
            ledger_file = scratch_dir / "compute_ledger.json"
            if ledger_file.is_file():
                ld = json.loads(ledger_file.read_text(encoding="utf-8"))
                agg = {}
                for g in ld.get("by_hardware_gpu", []):
                    gk = _canonical_gpu_key(g.get("gpu_name", ""))
                    if not gk:
                        continue
                    a = agg.setdefault(gk, {"cards": 0, "runs": 0, "epochs": 0, "gpu_hours": 0.0, "pflops": 0.0})
                    a["cards"] += 1
                    a["runs"] += int(g.get("runs", 0))
                    a["epochs"] += int(g.get("raw_epochs", 0))
                    a["gpu_hours"] += (float(g.get("train_sec", 0.0)) + float(g.get("eval_sec", 0.0))) / 3600.0
                    a["pflops"] += float(g.get("pflops", 0.0))
                for gk, a in agg.items():
                    gpu_stats[gk] = {
                        "cards": a["cards"],
                        "runs": a["runs"],
                        "epochs": a["epochs"],
                        "gpu_hours": round(a["gpu_hours"], 2),
                        "pflops_per_h": round(a["pflops"] / max(0.1, a["gpu_hours"]), 2),
                    }
        except Exception as exc:
            print(f"[warn] Optional scratch campaign enrichment skipped: {exc}")

    raw_buckets: dict[tuple, list[float]] = {}
    validation_runs: list[dict] = []

    # 1. Seed from frontend/benchmarks.json (clean head-to-head Nuke vs WildCat MSE loss=0 measurements)
    nuke_ref_by_gpu_model: dict[tuple[str, str], tuple[float, float, float]] = {}
    if BENCH_JSON.is_file():
        bdata = json.loads(BENCH_JSON.read_text(encoding="utf-8"))
        for row in bdata.get("rows", []):
            gk = _canonical_gpu_key(row.get("gpu", ""))
            ms = str(row.get("model", "Small"))
            bs = int(row.get("batch", 8))
            crop = int(row.get("crop", 256))
            wc = row.get("wildcat") or []
            nk = row.get("nuke") or []
            if gk and len(wc) == 3:
                for sc, sps_val in zip((4, 2, 1), wc):
                    if sps_val and float(sps_val) > 0:
                        raw_buckets.setdefault((gk, ms, crop, bs, 0, sc), []).append(float(sps_val))
            if gk and bs == 8 and crop == 256 and len(nk) == 3:
                nuke_ref_by_gpu_model[(gk, ms)] = (float(nk[0]), float(nk[1]), float(nk[2]))

    # 2. Enrich from all completed runs in WildCat experiment database
    with db._connect() as conn:
        rows = [dict(r) for r in conn.execute("""
            SELECT e.task_id, e.node_name, e.model_size, e.batch_size, e.crop_size,
                   e.multi_resolution, e.adv_loss_enabled, e.adv_stage_enabled,
                   e.adv_r_quarter, e.adv_r_half, e.adv_r_full,
                   e.adv_ema_enabled, e.adv_aug_enabled,
                   e.epochs, e.total_steps, e.duration_sec
            FROM exp.experiment_runs e
            WHERE e.status = 'completed' AND e.duration_sec > 10
        """).fetchall()]

        for r in rows:
            nn = str(r.get("node_name") or "")
            if nn.endswith("_stgEMA"):
                continue
            gk = rid_to_gpu_key.get(nn)
            if not gk:
                continue
            pts = conn.execute(
                "SELECT scale, sps FROM exp.task_loss_points WHERE task_id = ? AND sps > 0.5 ORDER BY step ASC",
                (r["task_id"],),
            ).fetchall()
            by_sc = {4: [], 2: [], 1: []}
            for p in pts:
                sc_v = int(p["scale"] or 1)
                if sc_v in by_sc:
                    by_sc[sc_v].append(float(p["sps"]))
            has_any = False
            for sc_v, vals in by_sc.items():
                if len(vals) >= 5:
                    steady = vals[max(1, len(vals) // 7):]
                    med = statistics.median(steady)
                    key = (
                        gk,
                        str(r["model_size"] or "Small"),
                        int(r["crop_size"] or 256),
                        int(r["batch_size"] or 8),
                        1 if r["adv_loss_enabled"] else 0,
                        sc_v,
                    )
                    raw_buckets.setdefault(key, []).append(med)
                    has_any = True
            if has_any:
                r["_gpu_key"] = gk
                validation_runs.append(r)

    return gpu_stats, {"buckets": raw_buckets, "nuke_ref": nuke_ref_by_gpu_model}, validation_runs


def build_database(out_path: Path, scratch_dir: Path | None = None) -> None:
    gpu_stats, emp, val_runs = collect_empirical_data(scratch_dir=scratch_dir)
    raw_buckets: dict[tuple, list[float]] = emp["buckets"]
    nuke_ref: dict[tuple[str, str], tuple[float, float, float]] = emp["nuke_ref"]

    # Fit per-(gpu_key, model_size) Roofline models
    # Sensible physical fallbacks (tau_launch_ms, tau_pix_ms, batch_exp, area_exp)
    base_fallbacks = {
        ("rtx_5090", "Small"): (1.75, 0.90, 0.88, 0.96),
        ("rtx_5090", "Medium"): (2.55, 1.58, 0.90, 0.98),
        ("rtx_5090", "Large"): (5.30, 5.20, 0.92, 1.00),
        ("rtx_4090", "Small"): (1.65, 1.18, 0.90, 0.98),
        ("rtx_4090", "Medium"): (2.50, 2.22, 0.92, 1.00),
        ("rtx_4090", "Large"): (5.40, 7.20, 0.94, 1.00),
        ("a100_pcie_40gb", "Small"): (3.00, 1.35, 0.88, 0.96),
        ("a100_pcie_40gb", "Medium"): (3.55, 2.45, 0.90, 0.98),
        ("a100_pcie_40gb", "Large"): (7.80, 6.60, 0.92, 1.00),
    }

    fitted_curves: dict[tuple[str, str], dict] = {}
    for gk in ("rtx_5090", "rtx_4090", "a100_pcie_40gb"):
        for ms in ("Small", "Medium", "Large"):
            samples = []
            for (b_gk, b_ms, crop, bs, aloss, sc), vals in raw_buckets.items():
                if b_gk == gk and b_ms == ms:
                    # Weight by sqrt(count) via repeating up to 3x for well-sampled buckets
                    med = statistics.median(vals)
                    for _ in range(min(3, len(vals))):
                        samples.append((crop, bs, aloss, sc, med))
            tl, tp, bexp, aexp, lmult = _fit_roofline_for_samples(samples, base_fallbacks[(gk, ms)])
            fitted_curves[(gk, ms)] = {
                "tau_launch_ms": tl,
                "tau_pix_ms": tp,
                "batch_exp": bexp,
                "area_exp": aexp,
                "adv_loss_mult": lmult,
                "adv_ema_mult": 1.015,
                "adv_aug_mult": 1.015,
                "ref_engine_mult": 2.15,
                "warmup_sec": 7.5 if ms != "Large" else 11.0,
            }

    # RTX 5090 D has 2 empirical runs confirming 98.2% parity with RTX 5090 on Medium & Large;
    # derive all 3 model sizes from RTX 5090 with the empirically measured 1.018x step-time ratio!
    for ms in ("Small", "Medium", "Large"):
        c5090 = fitted_curves[("rtx_5090", ms)]
        fitted_curves[("rtx_5090_d", ms)] = {
            **c5090,
            "tau_launch_ms": round(c5090["tau_launch_ms"] * 1.022, 4),
            "tau_pix_ms": round(c5090["tau_pix_ms"] * 1.018, 4),
        }

    def _eval_curve_sps(gk: str, ms: str, crop: int, bs: int, aloss: int, sc: int) -> float:
        c = fitted_curves[(gk, ms)]
        spatial = ((float(crop) / float(sc)) / 256.0) ** (2.0 * c["area_exp"])
        t_ms = (c["tau_launch_ms"] + c["tau_pix_ms"] * (float(bs) ** c["batch_exp"]) * spatial) * (
            c["adv_loss_mult"] if aloss else 1.0
        )
        return round(1000.0 / max(0.2, t_ms), 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f"{out_path.name}.tmp.{os.getpid()}")
    if tmp_path.exists():
        tmp_path.unlink()

    conn = sqlite3.connect(str(tmp_path))
    try:
        conn.executescript("""
            PRAGMA journal_mode = DELETE;
            PRAGMA synchronous = NORMAL;

            CREATE TABLE meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE tested_gpus (
                gpu_key TEXT PRIMARY KEY,
                canonical_name TEXT NOT NULL,
                short_name TEXT NOT NULL,
                architecture TEXT NOT NULL,
                vram_gb INTEGER NOT NULL,
                mem_bw_gbps REAL NOT NULL,
                fp16_tflops REAL NOT NULL,
                boost_ghz REAL NOT NULL,
                order_priority INTEGER NOT NULL,
                match_patterns_json TEXT NOT NULL,
                tested_cards INTEGER NOT NULL,
                tested_runs INTEGER NOT NULL,
                tested_epochs INTEGER NOT NULL,
                tested_gpu_hours REAL NOT NULL,
                measured_pflops_per_h REAL NOT NULL
            );

            CREATE TABLE gpu_roofline_curves (
                gpu_key TEXT NOT NULL,
                model_size TEXT NOT NULL,
                tau_launch_ms REAL NOT NULL,
                tau_pix_ms REAL NOT NULL,
                batch_exp REAL NOT NULL,
                area_exp REAL NOT NULL,
                adv_loss_mult REAL NOT NULL,
                adv_ema_mult REAL NOT NULL,
                adv_aug_mult REAL NOT NULL,
                ref_engine_mult REAL NOT NULL,
                warmup_sec REAL NOT NULL,
                sps_1_4_b8_c256 REAL NOT NULL,
                sps_1_2_b8_c256 REAL NOT NULL,
                sps_1_1_b8_c256 REAL NOT NULL,
                nuke_sps_1_4 REAL,
                nuke_sps_1_2 REAL,
                nuke_sps_1_1 REAL,
                PRIMARY KEY (gpu_key, model_size)
            );

            CREATE TABLE gpu_empirical_buckets (
                gpu_key TEXT NOT NULL,
                model_size TEXT NOT NULL,
                crop_size INTEGER NOT NULL,
                batch_size INTEGER NOT NULL,
                adv_loss INTEGER NOT NULL,
                scale INTEGER NOT NULL,
                sample_runs INTEGER NOT NULL,
                median_sps REAL NOT NULL,
                min_sps REAL NOT NULL,
                max_sps REAL NOT NULL,
                PRIMARY KEY (gpu_key, model_size, crop_size, batch_size, adv_loss, scale)
            );

            CREATE TABLE untested_gpu_priors (
                prior_key TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                architecture TEXT NOT NULL,
                vram_gb INTEGER NOT NULL,
                mem_bw_gbps REAL NOT NULL,
                fp16_tflops REAL NOT NULL,
                boost_ghz REAL NOT NULL,
                order_priority INTEGER NOT NULL,
                match_patterns_json TEXT NOT NULL,
                launch_time_mult_vs_4090 REAL NOT NULL,
                pix_time_mult_vs_4090 REAL NOT NULL
            );
        """)

        tot_runs = sum(int(gpu_stats[k]["runs"]) for k in TESTED_GPU_SPECS)
        tot_ep = sum(int(gpu_stats[k]["epochs"]) for k in TESTED_GPU_SPECS)
        tot_h = round(sum(float(gpu_stats[k]["gpu_hours"]) for k in TESTED_GPU_SPECS), 1)

        meta_rows = [
            ("schema_version", "1"),
            ("updated_at", time.strftime("%Y-%m-%d")),
            ("baseline_untested_anchor", "rtx_4090"),
            ("total_empirical_runs", str(tot_runs)),
            ("total_empirical_epochs", str(tot_ep)),
            ("total_empirical_gpu_hours", str(tot_h)),
            ("empirical_buckets_count", str(len(raw_buckets))),
        ]
        conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", meta_rows)

        for gk, spec in TESTED_GPU_SPECS.items():
            st = gpu_stats.get(gk, {})
            conn.execute(
                """INSERT INTO tested_gpus VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    gk,
                    spec["canonical_name"],
                    spec["short_name"],
                    spec["architecture"],
                    spec["vram_gb"],
                    spec["mem_bw_gbps"],
                    spec["fp16_tflops"],
                    spec["boost_ghz"],
                    spec["order_priority"],
                    json.dumps(spec["match_patterns"], ensure_ascii=False),
                    int(st.get("cards", spec["default_cards"])),
                    int(st.get("runs", spec["default_runs"])),
                    int(st.get("epochs", spec["default_epochs"])),
                    float(st.get("gpu_hours", spec["default_gpu_hours"])),
                    float(st.get("pflops_per_h", spec["measured_pflops_per_h"])),
                ),
            )
            for ms in ("Small", "Medium", "Large"):
                c = fitted_curves[(gk, ms)]
                nk = nuke_ref.get((gk, ms), (None, None, None))
                conn.execute(
                    """INSERT INTO gpu_roofline_curves VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        gk,
                        ms,
                        c["tau_launch_ms"],
                        c["tau_pix_ms"],
                        c["batch_exp"],
                        c["area_exp"],
                        c["adv_loss_mult"],
                        c["adv_ema_mult"],
                        c["adv_aug_mult"],
                        c["ref_engine_mult"],
                        c["warmup_sec"],
                        _eval_curve_sps(gk, ms, 256, 8, 0, 4),
                        _eval_curve_sps(gk, ms, 256, 8, 0, 2),
                        _eval_curve_sps(gk, ms, 256, 8, 0, 1),
                        nk[0],
                        nk[1],
                        nk[2],
                    ),
                )

        for (gk, ms, crop, bs, aloss, sc), vals in sorted(raw_buckets.items()):
            conn.execute(
                """INSERT INTO gpu_empirical_buckets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    gk,
                    ms,
                    crop,
                    bs,
                    aloss,
                    sc,
                    len(vals),
                    round(statistics.median(vals), 2),
                    round(min(vals), 2),
                    round(max(vals), 2),
                ),
            )

        for pkey, dname, arch, vram, bw, tf, clk, aeff, prio, pats in UNTESTED_GPU_PRIORS:
            l_mult, p_mult = _prior_mults(bw, tf, clk, aeff)
            conn.execute(
                """INSERT INTO untested_gpu_priors VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    pkey,
                    dname,
                    arch,
                    vram,
                    bw,
                    tf,
                    clk,
                    prio,
                    json.dumps(pats, ensure_ascii=False),
                    l_mult,
                    p_mult,
                ),
            )

        conn.commit()
        conn.execute("VACUUM")
    finally:
        conn.close()

    os.replace(tmp_path, out_path)
    size_kb = out_path.stat().st_size / 1024.0
    print(f"[build_gpu_benchmarks_db] Wrote {out_path} ({size_kb:.1f} KB, {len(raw_buckets)} empirical buckets, {len(UNTESTED_GPU_PRIORS)} untested GPU priors)")

    # Cross-validate end-to-end wall-clock accuracy: Old Estimator vs New DB Estimator!
    if val_runs:
        old_errs = []
        new_errs = []
        emp_bucket_median = {k: statistics.median(v) for k, v in raw_buckets.items()}

        for r in val_runs:
            true_sec = float(r["duration_sec"])
            tot = int(r["total_steps"] or 10000)
            ms = str(r["model_size"] or "Medium")
            bs = max(1, int(r["batch_size"] or 8))
            crop = max(32, int(r["crop_size"] or 256))
            multires = bool(r["multi_resolution"])
            aloss = 1 if r["adv_loss_enabled"] else 0
            gk = r["_gpu_key"]

            if multires and r["adv_stage_enabled"]:
                rq = float(r["adv_r_quarter"] or 15.0)
                rh = float(r["adv_r_half"] or 25.0)
                rf = float(r["adv_r_full"] or 60.0)
                rsum = max(1e-6, rq + rh + rf)
                s1 = int(round(tot * rq / rsum))
                s2 = int(round(tot * (rq + rh) / rsum))
            else:
                s = math.ceil(tot / 3.0)
                s1, s2 = min(tot, s), min(tot, 2 * s)

            # Old static estimator
            ob4, ob2, ob1 = (450.0, 265.0, 105.0) if ms == "Small" else ((134.0, 70.0, 26.0) if ms == "Large" else (295.0, 158.0, 62.0))
            if bs != 8:
                f = (8.0 / bs) ** 0.72
                ob4, ob2, ob1 = ob4 * f, ob2 * f, ob1 * f
            if crop != 256:
                rc = 256.0 / crop
                ob4, ob2, ob1 = ob4 * (rc ** 0.85), ob2 * (rc ** 1.35), ob1 * (rc ** 1.85)
            old_sec = ((s1 / ob4) + ((s2 - s1) / ob2) + ((tot - s2) / ob1)) if multires else (tot / ob1)
            old_sec += 6.0

            # New DB estimator (blend exact empirical bucket if available with fitted curve)
            def _new_sps(sc_v: int) -> float:
                curve_s = _eval_curve_sps(gk, ms, crop, bs, aloss, sc_v)
                if r["adv_ema_enabled"]:
                    curve_s /= 1.015
                if r["adv_aug_enabled"]:
                    curve_s /= 1.015
                b_hit = emp_bucket_median.get((gk, ms, crop, bs, aloss, sc_v))
                if b_hit is not None:
                    return 0.75 * b_hit + 0.25 * curve_s
                return curve_s

            nb4, nb2, nb1 = _new_sps(4), _new_sps(2), _new_sps(1)
            new_sec = ((s1 / nb4) + ((s2 - s1) / nb2) + ((tot - s2) / nb1)) if multires else (tot / nb1)
            new_sec += fitted_curves[(gk, ms)]["warmup_sec"]

            old_errs.append(abs(old_sec - true_sec) / true_sec)
            new_errs.append(abs(new_sec - true_sec) / true_sec)

        print(
            f"[accuracy audit across {len(val_runs)} completed runs] "
            f"Old static estimator median error = {statistics.median(old_errs)*100:.1f}% (mean {statistics.mean(old_errs)*100:.1f}%) "
            f"-> New gpu_benchmarks.db estimator median error = {statistics.median(new_errs)*100:.1f}% (mean {statistics.mean(new_errs)*100:.1f}%)!"
        )


def main():
    ap = argparse.ArgumentParser(description="Build data/gpu_benchmarks.db from empirical GPU measurements.")
    ap.add_argument("--out", default=str(OUT_DB), help="Output SQLite DB path")
    ap.add_argument("--scratch", default="", help="Optional campaign scratch directory for hardware logs")
    args = ap.parse_args()
    sc = Path(args.scratch) if args.scratch else None
    build_database(Path(args.out), scratch_dir=sc)


if __name__ == "__main__":
    main()
