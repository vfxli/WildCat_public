# -*- coding: utf-8 -*-
"""Numeric SQLite Analytics, Group Registry, Incremental Sync & Lifecycle Archiver."""
import csv
import glob
import io
import json
import math
import os
import shutil
import threading
import time
from urllib.parse import quote

from wildcat import advanced as _adv

from . import db, i18n, storage, telemetry

UNGROUPED_KEY = "__ungrouped__"

_SYNC_LOCK = threading.Lock()
_LAST_SYNC_TS: dict[str, float] = {}


def _num_float(v) -> float | None:
    if v is None:
        return None
    try:
        fv = float(v)
        return fv if math.isfinite(fv) else None
    except (TypeError, ValueError):
        return None


def _num_int(v, default: int | None = 0) -> int | None:
    if v is None:
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _infer_model_size(mo: dict) -> str:
    msize = str(mo.get("model_size") or "").strip()
    if msize in ("Small", "Medium", "Large"):
        return msize
    nr = int(mo.get("n_resblocks") or 1)
    ks = int(mo.get("k_size") or 5)
    if nr >= 3:
        return "Large"
    if ks <= 3:
        return "Small"
    return "Medium"


def _resolve_dataset_meta(cfg: dict, user_id: int) -> dict:
    exp = (cfg or {}).get("experiment") or {}
    paths = (cfg or {}).get("paths") or {}
    src_p = str(paths.get("source") or "").strip()

    for cand_id in (exp.get("dataset_id"), exp.get("eval_asset_id")):
        if cand_id:
            row = db.query_one("SELECT * FROM assets WHERE id = ?", (int(cand_id),))
            if row:
                return {
                    "dataset_id": int(row["id"]),
                    "dataset_name": str(row.get("name") or ""),
                    "dataset_frames": int(row.get("frame_count") or 0),
                    "width": int(row.get("width") or 0),
                    "height": int(row.get("height") or 0),
                }
    if src_p:
        row = db.query_one(
            "SELECT * FROM assets WHERE source_path = ? ORDER BY user_id = ? DESC, id ASC LIMIT 1",
            (src_p, int(user_id)),
        )
        if row:
            return {
                "dataset_id": int(row["id"]),
                "dataset_name": str(row.get("name") or ""),
                "dataset_frames": int(row.get("frame_count") or 0),
                "width": int(row.get("width") or 0),
                "height": int(row.get("height") or 0),
            }
        s_files = storage.expand_sequence_files(src_p)
        return {
            "dataset_id": None,
            "dataset_name": os.path.basename(os.path.dirname(src_p.rstrip("/\\"))) or src_p,
            "dataset_frames": len(s_files),
            "width": 0,
            "height": 0,
        }
    return {
        "dataset_id": None,
        "dataset_name": "",
        "dataset_frames": 0,
        "width": 0,
        "height": 0,
    }


def extract_numeric_task_fields(cfg: dict, user_id: int) -> dict:
    """Extract strictly typed numeric hyperparameters & organizational fields from task config."""
    cfg = cfg or {}
    mo = cfg.get("model") or {}
    tr = cfg.get("train") or {}
    exp = cfg.get("experiment") or {}
    adv = _adv.parse_advanced_config(cfg)
    ds = _resolve_dataset_meta(cfg, user_id)

    msize = _infer_model_size(mo)
    in_ch = max(1, int(mo.get("in_channels") or 3))
    out_ch = max(1, int(mo.get("out_channels") or in_ch))
    epochs = max(0, int(tr.get("epochs") or 0))
    bs_raw = tr.get("batch_size")
    batch_size = int(bs_raw) if (bs_raw is not None and str(bs_raw).lower() != "auto") else 8
    crop_size = max(16, int(tr.get("crop_size") or 256))
    multi_res = 1 if bool(tr.get("multi_resolution", True)) else 0
    prec = str(tr.get("precision") or "adaptive").lower()
    seed = int(tr.get("seed") or 0)

    cg_en = bool(adv["crop_grid"]["enabled"])
    crop_grid = int(adv["crop_grid"]["full_res_grid"]) if cg_en else 4

    lf_en = 1 if bool(adv["loss_family"]["enabled"]) else 0
    charb_ratio = float(adv["loss_family"]["charbonnier_ratio"]) if lf_en else 0.0
    grad_weight = float(adv["loss_family"]["grad_weight"]) if lf_en else 0.0
    hdr_alpha = float(adv["loss_family"]["hdr_alpha"]) if lf_en else 0.0

    st_en = 1 if bool(adv["stage_schedule"]["enabled"]) else 0
    r_q = float(adv["stage_schedule"]["r_quarter"]) if st_en else 33.333
    r_h = float(adv["stage_schedule"]["r_half"]) if st_en else 33.333
    r_f = float(adv["stage_schedule"]["r_full"]) if st_en else 33.334
    warmup_st = int(adv["stage_schedule"]["warmup_steps"]) if st_en else 0

    ema_en = 1 if bool(adv["ema_lr"]["enabled"]) else 0
    ema_decay = float(adv["ema_lr"]["ema_decay"]) if ema_en else 0.0

    aug_en = 1 if bool(adv["augment"]["enabled"]) else 0
    hflip_prob = float(adv["augment"]["hflip_prob"]) if aug_en else 0.0
    exposure_ev = float(adv["augment"]["exposure_ev"]) if aug_en else 0.0

    init_w = str(tr.get("initial_weights") or "").strip()
    bs_auto = tr.get("batch_size_auto")
    non_default = (
        msize != "Medium"
        or not multi_res
        or prec != "adaptive"
        or (bs_auto is False or (bs_auto is None and str(bs_raw or "auto").lower() != "auto"))
        or crop_size != 256
        or bool(init_w and init_w.lower() != "none")
        or cg_en
        or bool(lf_en)
        or bool(st_en)
        or bool(ema_en)
        or bool(aug_en)
    )

    bid = str(exp.get("batch_id") or "").strip()
    if bid == UNGROUPED_KEY:
        bid = ""
    bname = str(exp.get("batch_name") or "").strip() if bid else ""

    return {
        "dataset_id": ds["dataset_id"],
        "dataset_name": ds["dataset_name"],
        "dataset_frames": int(ds["dataset_frames"]),
        "width": int(ds["width"]),
        "height": int(ds["height"]),
        "model_size": msize,
        "in_channels": in_ch,
        "out_channels": out_ch,
        "epochs": epochs,
        "batch_size": batch_size,
        "crop_size": crop_size,
        "multi_resolution": multi_res,
        "precision_mode": prec,
        "seed": seed,
        "is_nuke_native": 0 if non_default else 1,
        "adv_crop_grid": crop_grid,
        "adv_loss_enabled": lf_en,
        "adv_charb_ratio": round(charb_ratio, 6),
        "adv_grad_weight": round(grad_weight, 6),
        "adv_hdr_alpha": round(hdr_alpha, 6),
        "adv_stage_enabled": st_en,
        "adv_r_quarter": round(r_q, 4),
        "adv_r_half": round(r_h, 4),
        "adv_r_full": round(r_f, 4),
        "adv_warmup_steps": warmup_st,
        "adv_ema_enabled": ema_en,
        "adv_ema_decay": round(ema_decay, 7),
        "adv_aug_enabled": aug_en,
        "adv_hflip_prob": round(hflip_prob, 4),
        "adv_exposure_ev": round(exposure_ev, 4),
        "batch_id": bid,
        "batch_name": bname,
        "sort_order": int(exp.get("sort_order") or 0),
        "exp_label": str(exp.get("label") or "").strip(),
        "exp_note": str(exp.get("note") or "").strip(),
    }


def sanitize_public_run_config(cfg: dict, node_name: str = "", dataset_name: str = "") -> str:
    """Return a clean JSON string containing only open-source-safe hyperparameters & experiment metadata."""
    cfg = cfg if isinstance(cfg, dict) else {}
    mo = dict(cfg.get("model") or {})
    tr_raw = dict(cfg.get("train") or {})
    tr = {}
    for k in (
        "epochs",
        "total_steps",
        "batch_size",
        "batch_size_auto",
        "crop_size",
        "multi_resolution",
        "precision",
        "seed",
        "lr",
        "val_every",
        "save_every",
    ):
        if k in tr_raw:
            tr[k] = tr_raw[k]
    tr["initial_weights"] = "None"

    exp_raw = dict(cfg.get("experiment") or {})
    exp_clean = {
        k: exp_raw[k]
        for k in ("batch_id", "batch_name", "sort_order", "label", "note")
        if k in exp_raw
    }
    ds_slug = (dataset_name or "benchmark").strip("/\\") or "benchmark"
    clean_obj = {
        "paths": {
            "source": f"dataset/{ds_slug}/source",
            "target": f"dataset/{ds_slug}/target",
            "run_dir": f"run/{node_name}" if node_name else "run/exp",
        },
        "model": mo,
        "train": tr,
        "advanced": dict(cfg.get("advanced") or {}),
        "experiment": exp_clean,
    }
    return json.dumps(clean_obj, ensure_ascii=False)


def _upsert_experiment_run_conn(conn, task_id: int) -> None:
    """Mirror sanitized benchmark/experiment run header & KPIs into `exp.experiment_runs` (`experiments.db`)."""
    t = conn.execute("SELECT * FROM main.tasks WHERE id = ?", (int(task_id),)).fetchone()
    if not t:
        return
    td = dict(t)
    try:
        raw_cfg = json.loads(td.get("config_json") or "{}")
    except Exception:
        raw_cfg = {}
    clean_cfg_json = sanitize_public_run_config(
        raw_cfg,
        node_name=str(td.get("node_name") or ""),
        dataset_name=str(td.get("dataset_name") or ""),
    )
    now = time.time()
    conn.execute(
        """
        INSERT INTO exp.experiment_runs (
            task_id, user_id, node_name, status,
            batch_id, batch_name, sort_order, exp_label, exp_note,
            dataset_name, dataset_frames, width, height,
            model_size, in_channels, out_channels,
            epochs, last_step, total_steps, last_loss,
            batch_size, crop_size, multi_resolution, precision_mode, seed, is_nuke_native,
            adv_crop_grid, adv_loss_enabled, adv_charb_ratio, adv_grad_weight, adv_hdr_alpha,
            adv_stage_enabled, adv_r_quarter, adv_r_half, adv_r_full, adv_warmup_steps,
            adv_ema_enabled, adv_ema_decay, adv_aug_enabled, adv_hflip_prob, adv_exposure_ev,
            duration_sec, avg_sps, best_val_mse, best_val_step, last_val_mse,
            best_psnr_srgb, best_psnr_aces, best_edge_l1, best_seq_hfi, best_seq_delta,
            config_json, created_at, started_at, finished_at, updated_at
        ) VALUES (
            ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?
        )
        ON CONFLICT(task_id) DO UPDATE SET
            user_id = excluded.user_id,
            node_name = excluded.node_name,
            status = excluded.status,
            batch_id = excluded.batch_id,
            batch_name = excluded.batch_name,
            sort_order = excluded.sort_order,
            exp_label = excluded.exp_label,
            exp_note = excluded.exp_note,
            dataset_name = excluded.dataset_name,
            dataset_frames = excluded.dataset_frames,
            width = excluded.width,
            height = excluded.height,
            model_size = excluded.model_size,
            in_channels = excluded.in_channels,
            out_channels = excluded.out_channels,
            epochs = excluded.epochs,
            last_step = excluded.last_step,
            total_steps = excluded.total_steps,
            last_loss = excluded.last_loss,
            batch_size = excluded.batch_size,
            crop_size = excluded.crop_size,
            multi_resolution = excluded.multi_resolution,
            precision_mode = excluded.precision_mode,
            seed = excluded.seed,
            is_nuke_native = excluded.is_nuke_native,
            adv_crop_grid = excluded.adv_crop_grid,
            adv_loss_enabled = excluded.adv_loss_enabled,
            adv_charb_ratio = excluded.adv_charb_ratio,
            adv_grad_weight = excluded.adv_grad_weight,
            adv_hdr_alpha = excluded.adv_hdr_alpha,
            adv_stage_enabled = excluded.adv_stage_enabled,
            adv_r_quarter = excluded.adv_r_quarter,
            adv_r_half = excluded.adv_r_half,
            adv_r_full = excluded.adv_r_full,
            adv_warmup_steps = excluded.adv_warmup_steps,
            adv_ema_enabled = excluded.adv_ema_enabled,
            adv_ema_decay = excluded.adv_ema_decay,
            adv_aug_enabled = excluded.adv_aug_enabled,
            adv_hflip_prob = excluded.adv_hflip_prob,
            adv_exposure_ev = excluded.adv_exposure_ev,
            duration_sec = excluded.duration_sec,
            avg_sps = excluded.avg_sps,
            best_val_mse = excluded.best_val_mse,
            best_val_step = excluded.best_val_step,
            last_val_mse = excluded.last_val_mse,
            best_psnr_srgb = excluded.best_psnr_srgb,
            best_psnr_aces = excluded.best_psnr_aces,
            best_edge_l1 = excluded.best_edge_l1,
            best_seq_hfi = excluded.best_seq_hfi,
            best_seq_delta = excluded.best_seq_delta,
            config_json = excluded.config_json,
            started_at = excluded.started_at,
            finished_at = excluded.finished_at,
            updated_at = excluded.updated_at
        """,
        (
            int(td["id"]),
            int(td.get("user_id") or 1),
            str(td.get("node_name") or f"task_{td['id']}"),
            str(td.get("status") or "completed"),
            str(td.get("batch_id") or ""),
            str(td.get("batch_name") or ""),
            int(td.get("sort_order") or 0),
            str(td.get("exp_label") or ""),
            str(td.get("exp_note") or ""),
            str(td.get("dataset_name") or ""),
            int(td.get("dataset_frames") or 0),
            int(td.get("width") or 0),
            int(td.get("height") or 0),
            str(td.get("model_size") or ""),
            int(td.get("in_channels") or 3),
            int(td.get("out_channels") or 3),
            int(td.get("epochs") or 0),
            int(td.get("last_step") or 0),
            int(td.get("total_steps") or 0),
            td.get("last_loss"),
            int(td.get("batch_size") or 0),
            int(td.get("crop_size") or 256),
            int(td.get("multi_resolution") if td.get("multi_resolution") is not None else 1),
            str(td.get("precision_mode") or "adaptive"),
            int(td.get("seed") or 0),
            int(td.get("is_nuke_native") if td.get("is_nuke_native") is not None else 1),
            int(td.get("adv_crop_grid") or 4),
            int(td.get("adv_loss_enabled") or 0),
            float(td.get("adv_charb_ratio") or 0.0),
            float(td.get("adv_grad_weight") or 0.0),
            float(td.get("adv_hdr_alpha") or 0.0),
            int(td.get("adv_stage_enabled") or 0),
            float(td.get("adv_r_quarter") if td.get("adv_r_quarter") is not None else 33.333),
            float(td.get("adv_r_half") if td.get("adv_r_half") is not None else 33.333),
            float(td.get("adv_r_full") if td.get("adv_r_full") is not None else 33.334),
            int(td.get("adv_warmup_steps") or 0),
            int(td.get("adv_ema_enabled") or 0),
            float(td.get("adv_ema_decay") or 0.0),
            int(td.get("adv_aug_enabled") or 0),
            float(td.get("adv_hflip_prob") or 0.0),
            float(td.get("adv_exposure_ev") or 0.0),
            td.get("duration_sec"),
            td.get("avg_sps"),
            td.get("best_val_mse"),
            td.get("best_val_step"),
            td.get("last_val_mse"),
            td.get("best_psnr_srgb"),
            td.get("best_psnr_aces"),
            td.get("best_edge_l1"),
            td.get("best_seq_hfi"),
            td.get("best_seq_delta"),
            clean_cfg_json,
            float(td.get("created_at") or now),
            td.get("started_at"),
            td.get("finished_at"),
            now,
        ),
    )


def sync_task_hyperparams_to_db(task_id: int, cfg: dict | None = None) -> None:
    """Persist typed numeric hyperparameter & group columns onto `tasks` and `exp.experiment_runs`."""
    row = db.query_one("SELECT id, user_id, config_json, started_at, finished_at FROM tasks WHERE id = ?", (int(task_id),))
    if not row:
        return
    if cfg is None:
        try:
            cfg = json.loads(row.get("config_json") or "{}")
        except Exception:
            cfg = {}
    nf = extract_numeric_task_fields(cfg, int(row["user_id"]))
    dur = None
    if row.get("started_at") and row.get("finished_at") and float(row["finished_at"]) >= float(row["started_at"]):
        dur = round(float(row["finished_at"]) - float(row["started_at"]), 2)

    now_ts = time.time()
    with db.transaction() as conn:
        conn.execute(
            """
            UPDATE tasks SET
                dataset_id = ?, dataset_name = ?, dataset_frames = ?, width = ?, height = ?,
                model_size = ?, in_channels = ?, out_channels = ?,
                epochs = ?, batch_size = ?, crop_size = ?, multi_resolution = ?,
                precision_mode = ?, seed = ?, is_nuke_native = ?,
                adv_crop_grid = ?,
                adv_loss_enabled = ?, adv_charb_ratio = ?, adv_grad_weight = ?, adv_hdr_alpha = ?,
                adv_stage_enabled = ?, adv_r_quarter = ?, adv_r_half = ?, adv_r_full = ?, adv_warmup_steps = ?,
                adv_ema_enabled = ?, adv_ema_decay = ?,
                adv_aug_enabled = ?, adv_hflip_prob = ?, adv_exposure_ev = ?,
                batch_id = ?, batch_name = ?, sort_order = ?, exp_label = ?, exp_note = ?,
                duration_sec = COALESCE(?, duration_sec),
                analytics_synced_at = CASE WHEN COALESCE(analytics_synced_at, 0) > 0 THEN analytics_synced_at ELSE ? END
            WHERE id = ?
            """,
            (
                nf["dataset_id"], nf["dataset_name"], nf["dataset_frames"], nf["width"], nf["height"],
                nf["model_size"], nf["in_channels"], nf["out_channels"],
                nf["epochs"], nf["batch_size"], nf["crop_size"], nf["multi_resolution"],
                nf["precision_mode"], nf["seed"], nf["is_nuke_native"],
                nf["adv_crop_grid"],
                nf["adv_loss_enabled"], nf["adv_charb_ratio"], nf["adv_grad_weight"], nf["adv_hdr_alpha"],
                nf["adv_stage_enabled"], nf["adv_r_quarter"], nf["adv_r_half"], nf["adv_r_full"], nf["adv_warmup_steps"],
                nf["adv_ema_enabled"], nf["adv_ema_decay"],
                nf["adv_aug_enabled"], nf["adv_hflip_prob"], nf["adv_exposure_ev"],
                nf["batch_id"], nf["batch_name"], nf["sort_order"], nf["exp_label"], nf["exp_note"],
                dur,
                now_ts,
                int(task_id),
            ),
        )
        if nf["batch_id"]:
            now = time.time()
            existing_grp = conn.execute(
                "SELECT id, batch_name FROM exp.experiment_groups WHERE user_id = ? AND batch_id = ?",
                (int(row["user_id"]), nf["batch_id"]),
            ).fetchone()
            if not existing_grp:
                min_ord_row = conn.execute(
                    "SELECT MIN(sort_order) FROM exp.experiment_groups WHERE user_id = ?",
                    (int(row["user_id"]),),
                ).fetchone()
                next_top = (int(min_ord_row[0]) - 1) if (min_ord_row and min_ord_row[0] is not None) else 1
                conn.execute(
                    """
                    INSERT INTO exp.experiment_groups (user_id, batch_id, batch_name, sort_order, note, created_at, updated_at)
                    VALUES (?, ?, ?, ?, '', ?, ?)
                    """,
                    (int(row["user_id"]), nf["batch_id"], nf["batch_name"] or nf["batch_id"], next_top, now, now),
                )
        _upsert_experiment_run_conn(conn, int(task_id))


def migrate_legacy_groups_for_user(user_id: int) -> None:
    """Idempotently migrate legacy `lab_groups_u<uid>` setting JSON into `experiment_groups` SQLite table."""
    uid = int(user_id)
    legacy_key = f"lab_groups_u{uid}"
    raw = db.get_setting(legacy_key, "")
    if not raw:
        return
    existing_cnt = db.query_one("SELECT COUNT(*) AS c FROM experiment_groups WHERE user_id = ?", (uid,))
    if existing_cnt and int(existing_cnt["c"]) > 0:
        return
    try:
        parsed = json.loads(raw)
    except Exception:
        return
    if not isinstance(parsed, dict):
        return

    grps = parsed.get("groups") if isinstance(parsed.get("groups"), dict) else {}
    ord_list = parsed.get("order") if isinstance(parsed.get("order"), list) else []
    now = time.time()

    ordered_keys = []
    seen = set()
    for k in ord_list:
        sk = str(k or "").strip()
        if sk and (sk == UNGROUPED_KEY or sk in grps) and sk not in seen:
            seen.add(sk)
            ordered_keys.append(sk)
    for k in grps:
        sk = str(k or "").strip()
        if sk and sk not in seen:
            seen.add(sk)
            ordered_keys.append(sk)

    with db.transaction() as conn:
        for idx, sk in enumerate(ordered_keys, start=1):
            if sk == UNGROUPED_KEY:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO experiment_groups
                        (user_id, batch_id, batch_name, sort_order, note, created_at, updated_at)
                    VALUES (?, ?, '__UNGROUPED__', ?, '', ?, ?)
                    """,
                    (uid, UNGROUPED_KEY, idx, now, now),
                )
            else:
                info = grps.get(sk) or {}
                bname = str(info.get("batch_name") or sk).strip() or sk
                cts = float(info.get("created_at") or now)
                conn.execute(
                    """
                    INSERT OR IGNORE INTO experiment_groups
                        (user_id, batch_id, batch_name, sort_order, note, created_at, updated_at)
                    VALUES (?, ?, ?, ?, '', ?, ?)
                    """,
                    (uid, sk, bname, idx, cts, now),
                )


def sync_task_log_to_db(task_row: dict, force: bool = False, max_chart_points: int = 1200) -> dict | None:
    """Ingest training Loss & Validation MSE numeric curves and summary KPIs from `train.log` into SQLite."""
    if not task_row:
        return None
    tid = int(task_row["id"])
    uid = int(task_row["user_id"])
    rdir_abs = storage.abs_from_root(task_row.get("run_dir") or "")
    log_path = os.path.join(rdir_abs, "train.log") if rdir_abs else ""
    if not log_path or not os.path.isfile(log_path):
        return None

    try:
        log_mtime = round(os.path.getmtime(log_path), 3)
    except OSError:
        return None

    is_terminal = task_row.get("status") in ("completed", "stopped", "failed", "cancelled")
    prev = db.query_one("SELECT synced_step, log_mtime, is_final FROM task_metrics WHERE task_id = ?", (tid,))
    if not force and prev and abs(float(prev["log_mtime"]) - log_mtime) < 0.002 and (prev["is_final"] or not is_terminal):
        return prev

    parsed = telemetry.parse_train_log_full_cached(log_path, max_chart_points=max_chart_points)
    steps = parsed.get("steps") or []
    vals = parsed.get("vals") or []
    loss_by_step = parsed.get("loss_by_step") or {}

    loss_rows = []
    sps_vals = []
    for s in steps:
        st = _num_int(s.get("step"), None)
        tot = _num_float(s.get("total"))
        if st is None or tot is None:
            continue
        sc = _num_int(s.get("scale"), 1) or 1
        lr_v = _num_float(s.get("lr"))
        sps_v = _num_float(s.get("sps"))
        if sps_v is not None and sps_v > 0:
            sps_vals.append(sps_v)
        loss_rows.append((tid, st, sc, tot, lr_v, sps_v))

    val_rows = []
    best_val_mse = None
    best_val_step = None
    for v in vals:
        vst = _num_int(v.get("step"), None)
        vmse = _num_float(v.get("mse"))
        if vst is None or vmse is None:
            continue
        val_rows.append((tid, vst, vmse))
        if best_val_mse is None or vmse < best_val_mse:
            best_val_mse = vmse
            best_val_step = vst

    last_val_mse = val_rows[-1][2] if val_rows else None
    all_losses = [fv for fv in (_num_float(x) for x in loss_by_step.values()) if fv is not None]
    min_train_loss = min(all_losses) if all_losses else (min((r[3] for r in loss_rows), default=None))
    final_loss = loss_rows[-1][3] if loss_rows else _num_float(task_row.get("last_loss"))
    synced_step = max(
        loss_rows[-1][1] if loss_rows else 0,
        val_rows[-1][1] if val_rows else 0,
        int(task_row.get("last_step") or 0),
    )
    avg_sps = round(sum(sps_vals) / len(sps_vals), 2) if sps_vals else None
    max_sps = round(max(sps_vals), 2) if sps_vals else None
    dur = None
    if task_row.get("started_at") and task_row.get("finished_at") and float(task_row["finished_at"]) >= float(task_row["started_at"]):
        dur = round(float(task_row["finished_at"]) - float(task_row["started_at"]), 2)

    now = time.time()
    final_flag = 1 if is_terminal else 0

    with db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO task_metrics (
                task_id, user_id, synced_step, log_mtime, step_points, val_points,
                min_train_loss, final_loss, best_val_mse, best_val_step, last_val_mse,
                avg_sps, max_sps, is_final, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(task_id) DO UPDATE SET
                synced_step = excluded.synced_step,
                log_mtime = excluded.log_mtime,
                step_points = excluded.step_points,
                val_points = excluded.val_points,
                min_train_loss = excluded.min_train_loss,
                final_loss = excluded.final_loss,
                best_val_mse = excluded.best_val_mse,
                best_val_step = excluded.best_val_step,
                last_val_mse = excluded.last_val_mse,
                avg_sps = excluded.avg_sps,
                max_sps = excluded.max_sps,
                is_final = excluded.is_final,
                updated_at = excluded.updated_at
            """,
            (
                tid, uid, synced_step, log_mtime, len(loss_rows), len(val_rows),
                min_train_loss, final_loss, best_val_mse, best_val_step, last_val_mse,
                avg_sps, max_sps, final_flag, now,
            ),
        )
        conn.execute("DELETE FROM task_loss_points WHERE task_id = ?", (tid,))
        if loss_rows:
            conn.executemany(
                "INSERT INTO task_loss_points (task_id, step, scale, total_loss, lr, sps) VALUES (?, ?, ?, ?, ?, ?)",
                loss_rows,
            )
        conn.execute("DELETE FROM task_val_points WHERE task_id = ?", (tid,))
        if val_rows:
            conn.executemany(
                "INSERT INTO task_val_points (task_id, step, val_mse) VALUES (?, ?, ?)",
                val_rows,
            )
        conn.execute(
            """
            UPDATE tasks SET
                best_val_mse = ?,
                best_val_step = ?,
                last_val_mse = ?,
                avg_sps = COALESCE(?, avg_sps),
                duration_sec = COALESCE(?, duration_sec),
                last_loss = COALESCE(?, last_loss),
                analytics_synced_at = ?
            WHERE id = ?
            """,
            (best_val_mse, best_val_step, last_val_mse, avg_sps, dur, final_loss, now, tid),
        )
        _upsert_experiment_run_conn(conn, tid)

    return {
        "task_id": tid,
        "synced_step": synced_step,
        "best_val_mse": best_val_mse,
        "best_val_step": best_val_step,
        "last_val_mse": last_val_mse,
        "final_loss": final_loss,
    }


def sync_eval_result_to_db(
    task_id: int,
    user_id: int,
    eval_key: str,
    data: dict,
    file_mtime: float = 0.0,
) -> int | None:
    """Persist supervised fidelity & full-sequence temporal flicker metrics as typed numeric SQLite rows."""
    if not data or not eval_key:
        return None
    tid = int(task_id)
    uid = int(user_id)
    sup = data.get("supervised") or {}
    seq = data.get("sequence") or {}
    per_frame = seq.get("per_frame") if isinstance(seq.get("per_frame"), list) else []
    now = time.time()

    ckpt_mode = str(data.get("ckpt_mode") or "best")
    ckpt_step = int(data.get("ckpt_step") or 0)
    ckpt_file = str(data.get("ckpt_file") or "")
    eval_asset_id = _num_int(data.get("eval_asset_id"), None)
    eval_asset_name = str(data.get("eval_asset_name") or "")
    device_name = str(data.get("device") or "")

    sup_frames = int(sup.get("frame_count") or 0)
    mse = _num_float(sup.get("mse"))
    mae = _num_float(sup.get("mae"))
    psnr_aces = _num_float(sup.get("psnr_aces"))
    psnr_srgb = _num_float(sup.get("psnr_srgb"))
    edge_l1 = _num_float(sup.get("edge_l1"))
    p99_err = _num_float(sup.get("p99_err"))
    bg_drift_mae = _num_float(sup.get("bg_drift_mae"))

    seq_frames = int(seq.get("frame_count") or 0)
    is_paired_seq = 1 if bool(seq.get("is_paired")) else 0
    seq_mse = _num_float(seq.get("mse"))
    seq_mae = _num_float(seq.get("mae"))
    seq_psnr_aces = _num_float(seq.get("psnr_aces"))
    seq_psnr_srgb = _num_float(seq.get("psnr_srgb"))
    seq_edge_l1 = _num_float(seq.get("edge_l1"))
    seq_p99_err = _num_float(seq.get("p99_err"))
    seq_bg_drift_mae = _num_float(seq.get("bg_drift_mae"))

    edit_l1 = _num_float(seq.get("edit_l1"))
    flicker_delta = _num_float(seq.get("flicker_delta"))
    flicker_delta_max = _num_float(seq.get("flicker_delta_max"))
    flicker_hf = _num_float(seq.get("flicker_hf"))
    flicker_static = _num_float(seq.get("flicker_static"))
    peak_flicker_idx = _num_int(seq.get("peak_flicker_idx"), None) if seq else None
    peak_flicker_val = _num_float(seq.get("peak_flicker_val"))
    elapsed_sec = _num_float(data.get("elapsed_sec"))
    evaluated_at = str(data.get("evaluated_at") or "")

    with db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO experiment_evals (
                task_id, user_id, eval_key, ckpt_mode, ckpt_step, ckpt_file,
                eval_asset_id, eval_asset_name, device_name,
                sup_frames, mse, mae, psnr_aces, psnr_srgb, edge_l1, p99_err, bg_drift_mae,
                seq_frames, is_paired_seq, seq_mse, seq_mae, seq_psnr_aces, seq_psnr_srgb,
                seq_edge_l1, seq_p99_err, seq_bg_drift_mae,
                edit_l1, flicker_delta, flicker_delta_max, flicker_hf, flicker_static,
                peak_flicker_idx, peak_flicker_val, elapsed_sec, evaluated_at,
                file_mtime, created_at, updated_at
            ) VALUES (
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?
            )
            ON CONFLICT(task_id, eval_key) DO UPDATE SET
                ckpt_mode = excluded.ckpt_mode,
                ckpt_step = excluded.ckpt_step,
                ckpt_file = excluded.ckpt_file,
                eval_asset_id = excluded.eval_asset_id,
                eval_asset_name = excluded.eval_asset_name,
                device_name = excluded.device_name,
                sup_frames = excluded.sup_frames,
                mse = excluded.mse,
                mae = excluded.mae,
                psnr_aces = excluded.psnr_aces,
                psnr_srgb = excluded.psnr_srgb,
                edge_l1 = excluded.edge_l1,
                p99_err = excluded.p99_err,
                bg_drift_mae = excluded.bg_drift_mae,
                seq_frames = excluded.seq_frames,
                is_paired_seq = excluded.is_paired_seq,
                seq_mse = excluded.seq_mse,
                seq_mae = excluded.seq_mae,
                seq_psnr_aces = excluded.seq_psnr_aces,
                seq_psnr_srgb = excluded.seq_psnr_srgb,
                seq_edge_l1 = excluded.seq_edge_l1,
                seq_p99_err = excluded.seq_p99_err,
                seq_bg_drift_mae = excluded.seq_bg_drift_mae,
                edit_l1 = excluded.edit_l1,
                flicker_delta = excluded.flicker_delta,
                flicker_delta_max = excluded.flicker_delta_max,
                flicker_hf = excluded.flicker_hf,
                flicker_static = excluded.flicker_static,
                peak_flicker_idx = excluded.peak_flicker_idx,
                peak_flicker_val = excluded.peak_flicker_val,
                elapsed_sec = excluded.elapsed_sec,
                evaluated_at = excluded.evaluated_at,
                file_mtime = excluded.file_mtime,
                updated_at = excluded.updated_at
            """,
            (
                tid, uid, eval_key, ckpt_mode, ckpt_step, ckpt_file,
                eval_asset_id, eval_asset_name, device_name,
                sup_frames, mse, mae, psnr_aces, psnr_srgb, edge_l1, p99_err, bg_drift_mae,
                seq_frames, is_paired_seq, seq_mse, seq_mae, seq_psnr_aces, seq_psnr_srgb,
                seq_edge_l1, seq_p99_err, seq_bg_drift_mae,
                edit_l1, flicker_delta, flicker_delta_max, flicker_hf, flicker_static,
                peak_flicker_idx, peak_flicker_val, elapsed_sec, evaluated_at,
                float(file_mtime or now), now, now,
            ),
        )
        ev_row = conn.execute(
            "SELECT id FROM experiment_evals WHERE task_id = ? AND eval_key = ?",
            (tid, eval_key),
        ).fetchone()
        eval_id = int(ev_row["id"])
        conn.execute("DELETE FROM experiment_eval_frames WHERE eval_id = ?", (eval_id,))
        if per_frame:
            frame_rows = []
            for idx_fallback, fr in enumerate(per_frame):
                f_idx = int(fr.get("idx", idx_fallback))
                f_name = str(fr.get("name") or f"frame_{f_idx:04d}")
                f_file = str(fr.get("file") or f"frame_{f_idx:04d}.jpg")
                frame_rows.append(
                    (
                        eval_id,
                        f_idx,
                        f_name,
                        f_file,
                        _num_float(fr.get("edit_l1")),
                        _num_float(fr.get("flicker_delta")),
                        _num_float(fr.get("flicker_hf")),
                        _num_float(fr.get("flicker_static")),
                        _num_float(fr.get("mse")),
                        _num_float(fr.get("psnr_srgb")),
                        _num_float(fr.get("psnr_aces")),
                    )
                )
            conn.executemany(
                """
                INSERT INTO experiment_eval_frames (
                    eval_id, frame_idx, frame_name, file_name,
                    edit_l1, flicker_delta, flicker_hf, flicker_static,
                    mse, psnr_srgb, psnr_aces
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                frame_rows,
            )

        # Roll up best KPIs onto `tasks` for O(1) list display & SQL sorting
        agg = conn.execute(
            """
            SELECT MAX(psnr_srgb) AS b_psnr_srgb,
                   MAX(psnr_aces) AS b_psnr_aces,
                   MIN(edge_l1) AS b_edge_l1,
                   MIN(flicker_hf) AS b_hfi,
                   MIN(flicker_delta) AS b_delta
            FROM experiment_evals
            WHERE task_id = ?
            """,
            (tid,),
        ).fetchone()
        if agg:
            conn.execute(
                """
                UPDATE tasks SET
                    best_psnr_srgb = ?,
                    best_psnr_aces = ?,
                    best_edge_l1 = ?,
                    best_seq_hfi = ?,
                    best_seq_delta = ?,
                    analytics_synced_at = ?
                WHERE id = ?
                """,
                (
                    agg["b_psnr_srgb"],
                    agg["b_psnr_aces"],
                    agg["b_edge_l1"],
                    agg["b_hfi"],
                    agg["b_delta"],
                    now,
                    tid,
                ),
            )
        _upsert_experiment_run_conn(conn, tid)
    return eval_id


def sync_task_evals_from_disk(task_row: dict) -> int:
    """Scan `run/<node>/eval/*/metrics.json` and ingest any new or updated evaluation files into SQLite."""
    if not task_row or task_row.get("files_pruned"):
        return 0
    tid = int(task_row["id"])
    uid = int(task_row["user_id"])
    rdir_abs = storage.abs_from_root(task_row.get("run_dir") or "")
    eval_root = os.path.join(rdir_abs, "eval") if rdir_abs else ""
    if not eval_root or not os.path.isdir(eval_root):
        return 0

    existing = {
        r["eval_key"]: float(r["file_mtime"] or 0.0)
        for r in db.query_all("SELECT eval_key, file_mtime FROM experiment_evals WHERE task_id = ?", (tid,))
    }
    synced = 0
    for entry in sorted(os.listdir(eval_root)):
        m_path = os.path.join(eval_root, entry, "metrics.json")
        if not os.path.isfile(m_path):
            continue
        try:
            mt = round(os.path.getmtime(m_path), 3)
        except OSError:
            continue
        if entry in existing and abs(existing[entry] - mt) < 0.002:
            continue
        try:
            with open(m_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception:
            continue
        if sync_eval_result_to_db(tid, uid, entry, payload, file_mtime=mt) is not None:
            synced += 1
    return synced


def sync_pending_tasks_to_db(user_id: int | None = None, force: bool = False) -> dict:
    """Fast incremental sync of unindexed tasks, logs, groups, and evaluation results into SQLite."""
    scope_key = f"u_{user_id}" if user_id is not None else "__all__"
    now = time.time()
    if not force and (now - _LAST_SYNC_TS.get(scope_key, 0.0)) < 1.5:
        return {"skipped_throttled": True}

    with _SYNC_LOCK:
        now = time.time()
        if not force and (now - _LAST_SYNC_TS.get(scope_key, 0.0)) < 1.5:
            return {"skipped_throttled": True}
        _LAST_SYNC_TS[scope_key] = now

        if user_id is not None:
            migrate_legacy_groups_for_user(int(user_id))
            rows = db.query_all(
                """
                SELECT t.*, tm.is_final AS tm_is_final, tm.log_mtime AS tm_log_mtime,
                       er.task_id AS er_synced
                FROM tasks t
                LEFT JOIN exp.task_metrics tm ON tm.task_id = t.id
                LEFT JOIN exp.experiment_runs er ON er.task_id = t.id
                WHERE t.user_id = ?
                ORDER BY t.id ASC
                """,
                (int(user_id),),
            )
        else:
            for u in db.query_all("SELECT id FROM users"):
                migrate_legacy_groups_for_user(int(u["id"]))
            rows = db.query_all(
                """
                SELECT t.*, tm.is_final AS tm_is_final, tm.log_mtime AS tm_log_mtime,
                       er.task_id AS er_synced
                FROM tasks t
                LEFT JOIN exp.task_metrics tm ON tm.task_id = t.id
                LEFT JOIN exp.experiment_runs er ON er.task_id = t.id
                ORDER BY t.id ASC
                """
            )

        synced_hp = 0
        synced_logs = 0
        synced_evals = 0
        for r in rows:
            need_hp = (
                force
                or not r.get("model_size")
                or float(r.get("analytics_synced_at") or 0.0) <= 0.0
                or not r.get("er_synced")
            )
            if need_hp:
                sync_task_hyperparams_to_db(int(r["id"]))
                synced_hp += 1
            if not r.get("files_pruned"):
                is_term = r.get("status") in ("completed", "stopped", "failed", "cancelled")
                if force or not r.get("tm_is_final") or not is_term:
                    if sync_task_log_to_db(r, force=force) is not None:
                        synced_logs += 1
                synced_evals += sync_task_evals_from_disk(r)
            elif float(r.get("analytics_synced_at") or 0.0) <= 0.0:
                db.execute("UPDATE tasks SET analytics_synced_at = ? WHERE id = ?", (now, int(r["id"])))

        return {
            "synced_hyperparams": synced_hp,
            "synced_logs": synced_logs,
            "synced_evals": synced_evals,
        }


def load_task_curves_from_db(task_id: int) -> dict | None:
    """Load convergence summary + downsampled numeric Loss & Val MSE points directly from SQLite."""
    tid = int(task_id)
    summary = db.query_one("SELECT * FROM task_metrics WHERE task_id = ?", (tid,))
    if not summary:
        return None
    loss_rows = db.query_all(
        "SELECT step, scale, total_loss, lr, sps FROM task_loss_points WHERE task_id = ? ORDER BY step ASC",
        (tid,),
    )
    val_rows = db.query_all(
        "SELECT step, val_mse FROM task_val_points WHERE task_id = ? ORDER BY step ASC",
        (tid,),
    )
    steps = [
        {
            "step": int(r["step"]),
            "scale": int(r["scale"] or 1),
            "total": float(r["total_loss"]),
            "lr": r["lr"],
            "sps": r["sps"],
        }
        for r in loss_rows
    ]
    vals = [
        {
            "step": int(r["step"]),
            "mse": float(r["val_mse"]),
        }
        for r in val_rows
    ]
    return {
        "summary": summary,
        "steps": steps,
        "vals": vals,
    }


def load_eval_from_db(task_id: int, node_name: str, run_dir_rel: str, eval_key: str) -> dict | None:
    """Load supervised & temporal sequence evaluation record directly from numeric SQLite tables."""
    if not eval_key:
        return None
    ev = db.query_one(
        "SELECT * FROM experiment_evals WHERE task_id = ? AND eval_key = ?",
        (int(task_id), str(eval_key)),
    )
    if not ev:
        return None

    sup_out = None
    if int(ev.get("sup_frames") or 0) > 0 or ev.get("psnr_srgb") is not None:
        sup_out = {
            "frame_count": int(ev.get("sup_frames") or 0),
            "mse": ev.get("mse"),
            "mae": ev.get("mae"),
            "psnr_aces": ev.get("psnr_aces"),
            "psnr_srgb": ev.get("psnr_srgb"),
            "edge_l1": ev.get("edge_l1"),
            "p99_err": ev.get("p99_err"),
            "bg_drift_mae": ev.get("bg_drift_mae"),
        }

    seq_out = None
    if int(ev.get("seq_frames") or 0) > 0:
        f_rows = db.query_all(
            "SELECT * FROM experiment_eval_frames WHERE eval_id = ? ORDER BY frame_idx ASC",
            (int(ev["id"]),),
        )
        rdir_abs = storage.abs_from_root(run_dir_rel or "")
        out_dir_abs = os.path.join(rdir_abs, "eval", eval_key, "out") if rdir_abs else ""
        frames_on_disk = bool(out_dir_abs and os.path.isdir(out_dir_abs))
        base_url = f"/f/run/{quote(node_name)}/eval/{quote(eval_key)}"
        is_paired = bool(ev.get("is_paired_seq"))

        per_frame = []
        for fr in f_rows:
            idx = int(fr["frame_idx"])
            fn = fr.get("file_name") or f"frame_{idx:04d}.jpg"
            item = {
                "idx": idx,
                "name": fr.get("frame_name") or fn,
                "file": fn,
                "edit_l1": fr.get("edit_l1"),
                "flicker_delta": fr.get("flicker_delta"),
                "flicker_hf": fr.get("flicker_hf"),
                "flicker_static": fr.get("flicker_static"),
            }
            if fr.get("mse") is not None:
                item["mse"] = fr["mse"]
                item["psnr_srgb"] = fr["psnr_srgb"]
                item["psnr_aces"] = fr["psnr_aces"]
            if frames_on_disk:
                item["out_url"] = f"{base_url}/out/{quote(fn)}"
                item["flicker_url"] = f"{base_url}/flicker/{quote(fn)}"
                if is_paired:
                    item["err_url"] = f"{base_url}/err/{quote(fn)}"
            per_frame.append(item)

        seq_out = {
            "frame_count": int(ev["seq_frames"]),
            "is_paired": is_paired,
            "frames_on_disk": frames_on_disk,
            "edit_l1": ev.get("edit_l1"),
            "flicker_delta": ev.get("flicker_delta"),
            "flicker_delta_max": ev.get("flicker_delta_max"),
            "flicker_hf": ev.get("flicker_hf"),
            "flicker_static": ev.get("flicker_static"),
            "peak_flicker_idx": int(ev["peak_flicker_idx"]) if ev.get("peak_flicker_idx") is not None else 0,
            "peak_flicker_val": ev.get("peak_flicker_val"),
            "per_frame": per_frame,
        }
        if is_paired and ev.get("seq_psnr_srgb") is not None:
            seq_out.update(
                {
                    "mse": ev.get("seq_mse"),
                    "mae": ev.get("seq_mae"),
                    "psnr_aces": ev.get("seq_psnr_aces"),
                    "psnr_srgb": ev.get("seq_psnr_srgb"),
                    "edge_l1": ev.get("seq_edge_l1"),
                    "p99_err": ev.get("seq_p99_err"),
                    "bg_drift_mae": ev.get("seq_bg_drift_mae"),
                }
            )

    return {
        "eval_key": ev["eval_key"],
        "eval_asset_id": ev.get("eval_asset_id"),
        "eval_asset_name": ev.get("eval_asset_name") or "",
        "ckpt_mode": ev.get("ckpt_mode") or "best",
        "ckpt_step": int(ev.get("ckpt_step") or 0),
        "ckpt_file": ev.get("ckpt_file") or "",
        "device": ev.get("device_name") or "",
        "elapsed_sec": ev.get("elapsed_sec"),
        "evaluated_at": ev.get("evaluated_at") or "",
        "from_sqlite": True,
        "supervised": sup_out,
        "sequence": seq_out,
    }


def archive_tasks_heavy_files(
    user_id: int,
    is_admin: bool,
    task_ids: list[int],
    keep_final_model: bool = True,
) -> dict:
    """Sync 100% of task curves & eval metrics into SQLite first, then clean heavy disk caches to free space."""
    freed_bytes = 0
    archived_count = 0
    seen = set()

    for raw_tid in task_ids or []:
        tid = int(raw_tid)
        if tid in seen:
            continue
        seen.add(tid)
        row = db.query_one("SELECT * FROM tasks WHERE id = ?", (tid,))
        if not row:
            continue
        if not is_admin and int(row["user_id"]) != int(user_id):
            raise PermissionError(i18n.t("server.taskEditForbidden"))
        if row["status"] in ("running", "queued"):
            continue

        # Guarantee all numeric records are safely persisted in SQLite BEFORE removing any disk cache
        sync_task_hyperparams_to_db(tid)
        sync_task_log_to_db(row, force=True)
        sync_task_evals_from_disk(row)

        rdir_abs = storage.abs_from_root(row.get("run_dir") or "")
        if rdir_abs and os.path.isdir(rdir_abs):
            before_b = storage.dir_total_bytes(row["run_dir"])
            for sub in ("preview_sup", "contact_sup"):
                sp = os.path.join(rdir_abs, sub)
                if os.path.isdir(sp):
                    shutil.rmtree(sp, ignore_errors=True)
            eval_root = os.path.join(rdir_abs, "eval")
            if os.path.isdir(eval_root):
                for ek in os.listdir(eval_root):
                    for img_sub in ("out", "flicker", "err"):
                        ip = os.path.join(eval_root, ek, img_sub)
                        if os.path.isdir(ip):
                            shutil.rmtree(ip, ignore_errors=True)
            for pat in ("Training_*.png", ".bundle_*.zip"):
                for fp in glob.glob(os.path.join(rdir_abs, pat)):
                    try:
                        os.remove(fp)
                    except OSError:
                        pass
            ckpt_dir = os.path.join(rdir_abs, "ckpt")
            if keep_final_model:
                # Keep only latest/best exported model or latest.pt if no exported file exists; drop step_*.pt optimizer states
                exported_list = glob.glob(os.path.join(rdir_abs, "Training_*.[0-9]*.pt"))
                has_exported = bool(exported_list)
                if os.path.isdir(ckpt_dir):
                    if has_exported:
                        shutil.rmtree(ckpt_dir, ignore_errors=True)
                    else:
                        for fp in glob.glob(os.path.join(ckpt_dir, "step_*.pt")):
                            try:
                                os.remove(fp)
                            except OSError:
                                pass
            else:
                if os.path.isdir(ckpt_dir):
                    shutil.rmtree(ckpt_dir, ignore_errors=True)
                model_pats = ["Training_*.pt"]
                for pat in model_pats:
                    for fp in glob.glob(os.path.join(rdir_abs, pat)):
                        try:
                            os.remove(fp)
                        except OSError:
                            pass
            after_b = storage.dir_total_bytes(row["run_dir"])
            freed_bytes += max(0, before_b - after_b)

        db.execute(
            "UPDATE tasks SET files_pruned = 1, analytics_synced_at = ? WHERE id = ?",
            (time.time(), tid),
        )
        archived_count += 1

    return {
        "archived_count": archived_count,
        "freed_mb": round(freed_bytes / (1024**2), 1),
    }


def get_db_analytics_summary(user_id: int | None = None, all_users: bool = False) -> dict:
    """Query SQLite directly for live counts of managed numeric records."""
    sync_pending_tasks_to_db(user_id=None if all_users else user_id)
    if all_users or user_id is None:
        t_cnt = db.query_one("SELECT COUNT(*) AS c FROM tasks")["c"]
        g_cnt = db.query_one("SELECT COUNT(*) AS c FROM experiment_groups WHERE batch_id != ?", (UNGROUPED_KEY,))["c"]
        e_cnt = db.query_one("SELECT COUNT(*) AS c FROM experiment_evals")["c"]
        lp_cnt = db.query_one("SELECT COUNT(*) AS c FROM task_loss_points")["c"]
        vp_cnt = db.query_one("SELECT COUNT(*) AS c FROM task_val_points")["c"]
        fp_cnt = db.query_one("SELECT COUNT(*) AS c FROM experiment_eval_frames")["c"]
    else:
        uid = int(user_id)
        t_cnt = db.query_one("SELECT COUNT(*) AS c FROM tasks WHERE user_id = ?", (uid,))["c"]
        g_cnt = db.query_one(
            "SELECT COUNT(*) AS c FROM experiment_groups WHERE user_id = ? AND batch_id != ?",
            (uid, UNGROUPED_KEY),
        )["c"]
        e_cnt = db.query_one("SELECT COUNT(*) AS c FROM experiment_evals WHERE user_id = ?", (uid,))["c"]
        lp_cnt = db.query_one(
            "SELECT COUNT(*) AS c FROM task_loss_points lp JOIN tasks t ON t.id = lp.task_id WHERE t.user_id = ?",
            (uid,),
        )["c"]
        vp_cnt = db.query_one(
            "SELECT COUNT(*) AS c FROM task_val_points vp JOIN tasks t ON t.id = vp.task_id WHERE t.user_id = ?",
            (uid,),
        )["c"]
        fp_cnt = db.query_one(
            "SELECT COUNT(*) AS c FROM experiment_eval_frames ef JOIN experiment_evals ee ON ee.id = ef.eval_id WHERE ee.user_id = ?",
            (uid,),
        )["c"]

    core_sz = os.path.getsize(db.DB_PATH) if os.path.isfile(db.DB_PATH) else 0
    exp_sz = os.path.getsize(db.EXP_DB_PATH) if os.path.isfile(db.EXP_DB_PATH) else 0
    r_cnt = (db.query_one("SELECT COUNT(*) AS c FROM exp.experiment_runs") or {}).get("c", 0)
    return {
        "tasks": int(t_cnt or 0),
        "experiment_runs": int(r_cnt or 0),
        "groups": int(g_cnt or 0),
        "evals": int(e_cnt or 0),
        "curve_points": int((lp_cnt or 0) + (vp_cnt or 0)),
        "eval_frame_points": int(fp_cnt or 0),
        "core_db_kb": round(core_sz / 1024.0, 1),
        "experiments_db_mb": round(exp_sz / (1024.0 * 1024.0), 2),
    }


def export_tasks_analytics(user_id: int, all_users: bool = False, fmt: str = "csv") -> tuple[bytes, str, str]:
    """Export all SQLite numeric task hyperparameters, convergence stats, and evaluation KPIs as CSV or JSON."""
    sync_pending_tasks_to_db(user_id=None if all_users else user_id, force=True)
    where_clause = "" if all_users else "WHERE t.user_id = ?"
    params = () if all_users else (int(user_id),)

    rows = db.query_all(
        f"""
        SELECT
            t.id AS task_id,
            t.username,
            t.node_name,
            t.status,
            t.batch_id,
            t.batch_name,
            t.sort_order,
            t.exp_label,
            t.exp_note,
            t.dataset_id,
            t.dataset_name,
            t.dataset_frames,
            t.width,
            t.height,
            t.model_size,
            t.in_channels,
            t.out_channels,
            t.epochs,
            t.last_step,
            t.total_steps,
            t.batch_size,
            t.crop_size,
            t.multi_resolution,
            t.precision_mode,
            t.seed,
            t.is_nuke_native,
            t.adv_crop_grid,
            t.adv_loss_enabled,
            t.adv_charb_ratio,
            t.adv_grad_weight,
            t.adv_hdr_alpha,
            t.adv_stage_enabled,
            t.adv_r_quarter,
            t.adv_r_half,
            t.adv_r_full,
            t.adv_warmup_steps,
            t.adv_ema_enabled,
            t.adv_ema_decay,
            t.adv_aug_enabled,
            t.adv_hflip_prob,
            t.adv_exposure_ev,
            t.duration_sec,
            t.avg_sps,
            tm.min_train_loss,
            tm.final_loss,
            t.best_val_mse,
            t.best_val_step,
            t.last_val_mse,
            ee.ckpt_mode AS eval_ckpt_mode,
            ee.ckpt_step AS eval_ckpt_step,
            ee.eval_asset_name,
            ee.psnr_srgb,
            ee.psnr_aces,
            ee.mse AS eval_mse,
            ee.mae AS eval_mae,
            ee.edge_l1,
            ee.p99_err,
            ee.bg_drift_mae,
            ee.seq_frames,
            ee.edit_l1,
            ee.flicker_delta,
            ee.flicker_delta_max,
            ee.flicker_hf,
            ee.flicker_static,
            ee.peak_flicker_idx,
            ee.peak_flicker_val,
            t.files_pruned,
            t.created_at,
            t.finished_at
        FROM tasks t
        LEFT JOIN task_metrics tm ON tm.task_id = t.id
        LEFT JOIN experiment_evals ee ON ee.id = (
            SELECT e2.id FROM experiment_evals e2
            WHERE e2.task_id = t.id
            ORDER BY (e2.psnr_srgb IS NOT NULL) DESC, e2.psnr_srgb DESC, e2.updated_at DESC
            LIMIT 1
        )
        {where_clause}
        ORDER BY t.user_id ASC, t.batch_id ASC, t.sort_order ASC, t.id DESC
        """,
        params,
    )

    stamp = time.strftime("%Y%m%d_%H%M%S")
    if str(fmt or "csv").lower() == "json":
        payload = json.dumps({"exported_at": stamp, "count": len(rows), "records": rows}, ensure_ascii=False, indent=2)
        return payload.encode("utf-8"), f"wildcat_analytics_{stamp}.json", "application/json"

    buf = io.StringIO()
    if rows:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = [
            "task_id", "username", "node_name", "status", "batch_name", "dataset_name",
            "model_size", "epochs", "total_steps", "best_val_mse", "psnr_srgb", "flicker_hf",
        ]
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    # UTF-8 with BOM so Excel opens numeric & CJK columns cleanly without garbling
    csv_bytes = ("\ufeff" + buf.getvalue()).encode("utf-8")
    return csv_bytes, f"wildcat_analytics_{stamp}.csv", "text/csv; charset=utf-8"
