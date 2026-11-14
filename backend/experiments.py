# -*- coding: utf-8 -*-
"""Experiment Lab Backend: Batch Queue Submission, Multi-Run Curve/Metric Comparison & Streaming Sequence Evaluation."""
import collections
import glob
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import traceback
import uuid
from urllib.parse import quote

from wildcat import advanced as _adv

from . import analytics_db, db, i18n, scheduler, service, storage, telemetry


def summarize_config_diff(cfg: dict) -> dict:
    """Extract concise technical parameter tags comparing `cfg` against Nuke CopyCat standard defaults."""
    cfg = cfg or {}
    mo = cfg.get("model") or {}
    tr = cfg.get("train") or {}
    adv = _adv.parse_advanced_config(cfg)

    tags = []
    non_default = False

    # Model size
    msize = str(mo.get("model_size") or "").strip()
    if not msize:
        nr = int(mo.get("n_resblocks") or 1)
        ks = int(mo.get("k_size") or 5)
        if nr >= 3:
            msize = "Large"
        elif ks <= 3:
            msize = "Small"
        else:
            msize = "Medium"
    if msize != "Medium":
        non_default = True
        tags.append(msize)

    # Multi-Resolution (Nuke default: True)
    multi_res = bool(tr.get("multi_resolution", True))
    if not multi_res:
        non_default = True
        tags.append("MultiRes=Off")

    # Precision (Nuke default: adaptive)
    prec = str(tr.get("precision") or "adaptive").lower()
    if prec != "adaptive":
        non_default = True
        tags.append(prec.upper())

    # Batch size (Nuke default: Auto -> 8)
    bs_auto = tr.get("batch_size_auto")
    bs_raw = tr.get("batch_size", "auto")
    if bs_auto is False or (bs_auto is None and str(bs_raw).lower() != "auto"):
        non_default = True
        tags.append(f"Batch={bs_raw}")

    # Crop size (Nuke default: 256)
    crop = int(tr.get("crop_size") or 256)
    if crop != 256:
        non_default = True
        tags.append(f"Crop={crop}")

    # Epochs (Nuke default: 10000)
    ep = int(tr.get("epochs") or 0)
    st_tot = int(tr.get("total_steps") or 0)
    if ep > 0 and ep != 10000:
        tags.append(f"{ep}ep")
    elif ep == 0 and st_tot > 0:
        tags.append(f"{st_tot}st")

    # Initial weights
    init_w = str(tr.get("initial_weights") or "").strip()
    if init_w and init_w.lower() != "none":
        non_default = True
        tags.append(f"Init={os.path.splitext(os.path.basename(init_w))[0]}")

    # 5 Advanced Hyperparameter modules (matching wildcat/advanced.py)
    if adv["crop_grid"]["enabled"]:
        non_default = True
        gpx = int(adv["crop_grid"]["full_res_grid"])
        tags.append(f"Grid={gpx}px")

    if adv["loss_family"]["enabled"]:
        non_default = True
        cr = float(adv["loss_family"]["charbonnier_ratio"])
        gw = float(adv["loss_family"]["grad_weight"])
        ha = float(adv["loss_family"]["hdr_alpha"])
        lparts = []
        if cr > 0:
            lparts.append(f"Charb({cr:g})")
        if gw > 0:
            lparts.append(f"{gw:g}∇")
        if ha > 0:
            lparts.append(f"HDR({ha:g})")
        tags.append("+".join(lparts) if lparts else "CompLoss")

    if adv["stage_schedule"]["enabled"]:
        non_default = True
        rq = int(round(adv["stage_schedule"]["r_quarter"]))
        rh = int(round(adv["stage_schedule"]["r_half"]))
        rf = int(round(adv["stage_schedule"]["r_full"]))
        tags.append(f"Stage({rq}/{rh}/{rf})")

    if adv["ema_lr"]["enabled"]:
        non_default = True
        tags.append(f"EMA({adv['ema_lr']['ema_decay']:g})")

    if adv["augment"]["enabled"]:
        non_default = True
        ap = []
        if adv["augment"]["hflip_prob"] > 0:
            ap.append(f"Flip({adv['augment']['hflip_prob']:g})")
        if adv["augment"]["exposure_ev"] > 0:
            ap.append(f"EV±{adv['augment']['exposure_ev']:g}")
        tags.append("Aug(" + "/".join(ap) + ")" if ap else "Aug")

    stop_opt_ep = int(tr.get("stop_after_optimal_epochs") or 0)
    if stop_opt_ep > 0:
        tags.append(f"OptStop(+{stop_opt_ep}ep)")

    exp_meta = cfg.get("experiment") or {}
    return {
        "is_nuke_native": not non_default,
        "model_size": msize,
        "tags": tags,
        "label": str(exp_meta.get("label") or "").strip(),
        "note": str(exp_meta.get("note") or "").strip(),
        "batch_id": str(exp_meta.get("batch_id") or "").strip(),
        "batch_name": str(exp_meta.get("batch_name") or "").strip(),
        "sort_order": int(exp_meta.get("sort_order") or 0),
    }


UNGROUPED_KEY = "__ungrouped__"


def _atomic_write_json(abs_path: str, payload: dict) -> None:
    if not abs_path:
        return
    parent = os.path.dirname(abs_path)
    if not parent or not os.path.isdir(parent):
        return
    tmp_path = f"{abs_path}.tmp.{os.getpid()}.{threading.get_ident()}"
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        os.replace(tmp_path, abs_path)
    except OSError:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass


def _persist_task_config(task_row: dict, cfg: dict) -> None:
    cfg_json = json.dumps(cfg, ensure_ascii=False)
    exp = (cfg or {}).get("experiment") or {}
    bid = str(exp.get("batch_id") or "").strip()
    if bid == UNGROUPED_KEY:
        bid = ""
    bname = str(exp.get("batch_name") or "").strip() if bid else ""
    sord = int(exp.get("sort_order") or 0)
    lbl = str(exp.get("label") or "").strip()
    note = str(exp.get("note") or "").strip()
    db.execute(
        """
        UPDATE tasks
        SET config_json = ?, batch_id = ?, batch_name = ?, sort_order = ?, exp_label = ?, exp_note = ?
        WHERE id = ?
        """,
        (cfg_json, bid, bname, sord, lbl, note, int(task_row["id"])),
    )
    cfg_rel = str(task_row.get("config_path") or "").strip()
    if cfg_rel:
        cfg_abs = storage.abs_from_root(cfg_rel)
        if os.path.isfile(cfg_abs):
            _atomic_write_json(cfg_abs, cfg)
    rdir_rel = str(task_row.get("run_dir") or "").strip()
    node_name = str(task_row.get("node_name") or "").strip()
    if rdir_rel and node_name:
        run_cfg_abs = os.path.join(storage.abs_from_root(rdir_rel), f"{node_name}.json")
        if os.path.isfile(run_cfg_abs):
            _atomic_write_json(run_cfg_abs, cfg)


def _normalize_group_sort_orders(conn, user_id: int) -> list[dict]:
    rows = [
        dict(r)
        for r in conn.execute(
            "SELECT * FROM experiment_groups WHERE user_id = ? ORDER BY sort_order ASC, id ASC",
            (int(user_id),),
        ).fetchall()
    ]
    for idx, r in enumerate(rows, start=1):
        if int(r["sort_order"]) != idx:
            conn.execute(
                "UPDATE experiment_groups SET sort_order = ? WHERE id = ?",
                (idx, int(r["id"])),
            )
            r["sort_order"] = idx
    return rows


def get_user_group_meta(user_id: int) -> dict:
    """Return ordered experiment group registry from SQLite `experiment_groups`, reconciled with `tasks`."""
    uid = int(user_id)
    analytics_db.migrate_legacy_groups_for_user(uid)

    task_rows = db.query_all(
        "SELECT id, batch_id, batch_name, config_json, created_at FROM tasks WHERE user_id = ? ORDER BY id DESC",
        (uid,),
    )
    discovered = []
    discovered_info = {}
    has_ungrouped = False

    for r in task_rows:
        bid = str(r.get("batch_id") or "").strip()
        bname = str(r.get("batch_name") or "").strip()
        if not bid:
            try:
                cfg = json.loads(r.get("config_json") or "{}")
            except Exception:
                cfg = {}
            exp = cfg.get("experiment") or {}
            bid = str(exp.get("batch_id") or "").strip()
            bname = str(exp.get("batch_name") or "").strip()
            if bid and bid != UNGROUPED_KEY:
                db.execute(
                    "UPDATE tasks SET batch_id = ?, batch_name = ?, sort_order = ? WHERE id = ?",
                    (bid, bname or bid, int(exp.get("sort_order") or 0), int(r["id"])),
                )
        if not bid or bid == UNGROUPED_KEY:
            has_ungrouped = True
            continue
        if bid not in discovered_info:
            discovered.append(bid)
            discovered_info[bid] = (bname or bid, float(r.get("created_at") or time.time()))

    now = time.time()
    with db.transaction() as conn:
        existing_rows = [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM experiment_groups WHERE user_id = ? ORDER BY sort_order ASC, id ASC",
                (uid,),
            ).fetchall()
        ]
        by_bid = {r["batch_id"]: r for r in existing_rows}
        min_sort = min((int(r["sort_order"]) for r in existing_rows), default=1)

        # Prepend any newly discovered task groups in newest-first order
        new_bids = [b for b in discovered if b not in by_bid]
        for offset, bid in enumerate(reversed(new_bids), start=1):
            bname, cts = discovered_info[bid]
            conn.execute(
                """
                INSERT INTO experiment_groups (user_id, batch_id, batch_name, sort_order, note, created_at, updated_at)
                VALUES (?, ?, ?, ?, '', ?, ?)
                """,
                (uid, bid, bname, min_sort - offset, cts, now),
            )

        # Update placeholder group names if task config carries a human-readable batch_name
        for bid, (bname, _) in discovered_info.items():
            if bid in by_bid and by_bid[bid]["batch_name"] == bid and bname != bid:
                conn.execute(
                    "UPDATE experiment_groups SET batch_name = ?, updated_at = ? WHERE id = ?",
                    (bname, now, int(by_bid[bid]["id"])),
                )

        if has_ungrouped and UNGROUPED_KEY not in by_bid:
            max_sort = max((int(r["sort_order"]) for r in existing_rows), default=0)
            conn.execute(
                """
                INSERT OR IGNORE INTO experiment_groups
                    (user_id, batch_id, batch_name, sort_order, note, created_at, updated_at)
                VALUES (?, ?, '__UNGROUPED__', ?, '', ?, ?)
                """,
                (uid, UNGROUPED_KEY, max_sort + 100, now, now),
            )

        final_rows = _normalize_group_sort_orders(conn, uid)

    ordered_groups = [
        {
            "batch_id": r["batch_id"],
            "batch_name": r["batch_name"],
            "sort_order": int(r["sort_order"]),
            "created_at": float(r["created_at"]),
        }
        for r in final_rows
        if r["batch_id"] != UNGROUPED_KEY
    ]
    order_keys = [r["batch_id"] for r in final_rows]
    return {
        "groups": ordered_groups,
        "order": order_keys,
    }


def create_experiment_group(
    user_id: int,
    is_admin: bool,
    batch_name: str,
    task_ids: list[int] | None = None,
) -> dict:
    uid = int(user_id)
    clean_name = str(batch_name or "").strip()
    if not clean_name:
        raise ValueError(i18n.t("server.expGroupNameRequired"))
    analytics_db.migrate_legacy_groups_for_user(uid)
    batch_id = f"grp_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    now = time.time()

    with db.transaction() as conn:
        min_row = conn.execute(
            "SELECT MIN(sort_order) FROM experiment_groups WHERE user_id = ?",
            (uid,),
        ).fetchone()
        top_sort = (int(min_row[0]) - 1) if (min_row and min_row[0] is not None) else 0
        conn.execute(
            """
            INSERT INTO experiment_groups (user_id, batch_id, batch_name, sort_order, note, created_at, updated_at)
            VALUES (?, ?, ?, ?, '', ?, ?)
            """,
            (uid, batch_id, clean_name, top_sort, now, now),
        )
        _normalize_group_sort_orders(conn, uid)

    moved = 0
    if task_ids:
        mv = move_tasks_to_group(
            user_id=uid,
            is_admin=is_admin,
            task_ids=task_ids,
            target_batch_id=batch_id,
            target_batch_name=clean_name,
        )
        moved = int(mv.get("moved_count") or 0)

    return {
        "batch_id": batch_id,
        "batch_name": clean_name,
        "moved_count": moved,
        "lab_groups": get_user_group_meta(uid),
    }


def rename_experiment_group(
    user_id: int,
    is_admin: bool,
    batch_id: str,
    batch_name: str,
) -> dict:
    uid = int(user_id)
    bid = str(batch_id or "").strip()
    clean_name = str(batch_name or "").strip()
    if not bid or bid == UNGROUPED_KEY:
        raise ValueError(i18n.t("server.expGroupNotFound"))
    if not clean_name:
        raise ValueError(i18n.t("server.expGroupNameRequired"))

    analytics_db.migrate_legacy_groups_for_user(uid)
    now = time.time()
    with db.transaction() as conn:
        existing = conn.execute(
            "SELECT id FROM experiment_groups WHERE user_id = ? AND batch_id = ?",
            (uid, bid),
        ).fetchone()
        if existing:
            conn.execute(
                "UPDATE experiment_groups SET batch_name = ?, updated_at = ? WHERE id = ?",
                (clean_name, now, int(existing["id"])),
            )
        else:
            conn.execute(
                """
                INSERT INTO experiment_groups (user_id, batch_id, batch_name, sort_order, note, created_at, updated_at)
                VALUES (?, ?, ?, 0, '', ?, ?)
                """,
                (uid, bid, clean_name, now, now),
            )
            _normalize_group_sort_orders(conn, uid)

    rows = db.query_all("SELECT * FROM tasks WHERE user_id = ? ORDER BY id ASC", (uid,))
    updated = 0
    for r in rows:
        try:
            cfg = json.loads(r.get("config_json") or "{}")
        except Exception:
            cfg = {}
        exp = cfg.get("experiment") or {}
        if str(r.get("batch_id") or exp.get("batch_id") or "").strip() == bid:
            exp["batch_id"] = bid
            exp["batch_name"] = clean_name
            cfg["experiment"] = exp
            _persist_task_config(r, cfg)
            updated += 1

    return {
        "batch_id": bid,
        "batch_name": clean_name,
        "updated_tasks": updated,
        "lab_groups": get_user_group_meta(uid),
    }


def move_tasks_to_group(
    user_id: int,
    is_admin: bool,
    task_ids: list[int],
    target_batch_id: str,
    target_batch_name: str = "",
) -> dict:
    uid = int(user_id)
    clean_target = str(target_batch_id or "").strip()
    if clean_target == "__new__":
        return create_experiment_group(
            user_id=uid,
            is_admin=is_admin,
            batch_name=target_batch_name,
            task_ids=task_ids,
        )

    analytics_db.migrate_legacy_groups_for_user(uid)
    loaded_rows = []
    seen = set()
    for raw_tid in task_ids or []:
        tid = int(raw_tid)
        if tid in seen:
            continue
        seen.add(tid)
        row = db.query_one("SELECT * FROM tasks WHERE id = ?", (tid,))
        if not row:
            continue
        if not is_admin and int(row["user_id"]) != uid:
            raise PermissionError(i18n.t("server.taskEditForbidden"))
        loaded_rows.append(row)

    is_ungroup = clean_target in ("", UNGROUPED_KEY)
    resolved_name = ""
    base_sort = 0
    now = time.time()

    if not is_ungroup:
        grp_row = db.query_one(
            "SELECT * FROM experiment_groups WHERE user_id = ? AND batch_id = ?",
            (uid, clean_target),
        )
        resolved_name = (
            str(target_batch_name or "").strip()
            or str((grp_row or {}).get("batch_name") or "").strip()
            or clean_target
        )
        with db.transaction() as conn:
            if not grp_row:
                min_row = conn.execute(
                    "SELECT MIN(sort_order) FROM experiment_groups WHERE user_id = ?",
                    (uid,),
                ).fetchone()
                top_sort = (int(min_row[0]) - 1) if (min_row and min_row[0] is not None) else 0
                conn.execute(
                    """
                    INSERT INTO experiment_groups (user_id, batch_id, batch_name, sort_order, note, created_at, updated_at)
                    VALUES (?, ?, ?, ?, '', ?, ?)
                    """,
                    (uid, clean_target, resolved_name, top_sort, now, now),
                )
                _normalize_group_sort_orders(conn, uid)
            elif target_batch_name and str(target_batch_name).strip():
                conn.execute(
                    "UPDATE experiment_groups SET batch_name = ?, updated_at = ? WHERE id = ?",
                    (resolved_name, now, int(grp_row["id"])),
                )

        existing_rows = db.query_all(
            "SELECT id, batch_id, sort_order, config_json FROM tasks WHERE user_id = ?",
            (uid,),
        )
        moving_set = {int(r["id"]) for r in loaded_rows}
        existing_in_grp = 0
        for er in existing_rows:
            if int(er["id"]) in moving_set:
                continue
            ebid = str(er.get("batch_id") or "").strip()
            esord = int(er.get("sort_order") or 0)
            if not ebid:
                try:
                    ecfg = json.loads(er.get("config_json") or "{}")
                except Exception:
                    ecfg = {}
                eexp = ecfg.get("experiment") or {}
                ebid = str(eexp.get("batch_id") or "").strip()
                esord = int(eexp.get("sort_order") or esord)
            if ebid == clean_target:
                existing_in_grp += 1
                base_sort = max(base_sort, esord, existing_in_grp)

    for idx, row in enumerate(loaded_rows):
        try:
            cfg = json.loads(row.get("config_json") or "{}")
        except Exception:
            cfg = {}
        exp = dict(cfg.get("experiment") or {})
        if is_ungroup:
            exp.pop("batch_id", None)
            exp.pop("batch_name", None)
            exp["sort_order"] = idx + 1
        else:
            exp["batch_id"] = clean_target
            exp["batch_name"] = resolved_name
            exp["sort_order"] = base_sort + idx + 1
        cfg["experiment"] = exp
        _persist_task_config(row, cfg)

    return {
        "batch_id": "" if is_ungroup else clean_target,
        "batch_name": "" if is_ungroup else resolved_name,
        "moved_count": len(loaded_rows),
        "lab_groups": get_user_group_meta(uid),
    }


def reorder_experiment_groups_or_tasks(
    user_id: int,
    is_admin: bool,
    group_order: list[str] | None = None,
    task_ids: list[int] | None = None,
) -> dict:
    uid = int(user_id)
    if group_order is not None:
        get_user_group_meta(uid)
        now = time.time()
        with db.transaction() as conn:
            rows = [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM experiment_groups WHERE user_id = ? ORDER BY sort_order ASC, id ASC",
                    (uid,),
                ).fetchall()
            ]
            by_bid = {r["batch_id"]: r for r in rows}
            if UNGROUPED_KEY in group_order and UNGROUPED_KEY not in by_bid:
                conn.execute(
                    """
                    INSERT INTO experiment_groups
                        (user_id, batch_id, batch_name, sort_order, note, created_at, updated_at)
                    VALUES (?, ?, '__UNGROUPED__', 9999, '', ?, ?)
                    """,
                    (uid, UNGROUPED_KEY, now, now),
                )
                rows = [
                    dict(r)
                    for r in conn.execute(
                        "SELECT * FROM experiment_groups WHERE user_id = ? ORDER BY sort_order ASC, id ASC",
                        (uid,),
                    ).fetchall()
                ]
                by_bid = {r["batch_id"]: r for r in rows}

            ordered_bids = []
            seen = set()
            for k in group_order:
                sk = str(k or "").strip()
                if sk in by_bid and sk not in seen:
                    seen.add(sk)
                    ordered_bids.append(sk)
            for r in rows:
                if r["batch_id"] not in seen:
                    seen.add(r["batch_id"])
                    ordered_bids.append(r["batch_id"])
            for idx, sk in enumerate(ordered_bids, start=1):
                conn.execute(
                    "UPDATE experiment_groups SET sort_order = ?, updated_at = ? WHERE user_id = ? AND batch_id = ?",
                    (idx, now, uid, sk),
                )

    reordered_tasks = 0
    if task_ids:
        seen_t = set()
        pos = 1
        for raw_tid in task_ids:
            tid = int(raw_tid)
            if tid in seen_t:
                continue
            seen_t.add(tid)
            row = db.query_one("SELECT * FROM tasks WHERE id = ?", (tid,))
            if not row:
                continue
            if not is_admin and int(row["user_id"]) != uid:
                raise PermissionError(i18n.t("server.taskEditForbidden"))
            try:
                cfg = json.loads(row.get("config_json") or "{}")
            except Exception:
                cfg = {}
            exp = dict(cfg.get("experiment") or {})
            exp["sort_order"] = pos
            cfg["experiment"] = exp
            _persist_task_config(row, cfg)
            pos += 1
            reordered_tasks += 1

    return {
        "reordered_tasks": reordered_tasks,
        "lab_groups": get_user_group_meta(uid),
    }


def dissolve_experiment_group(user_id: int, is_admin: bool, batch_id: str) -> dict:
    uid = int(user_id)
    bid = str(batch_id or "").strip()
    if not bid or bid == UNGROUPED_KEY:
        raise ValueError(i18n.t("server.expGroupNotFound"))

    analytics_db.migrate_legacy_groups_for_user(uid)
    with db.transaction() as conn:
        conn.execute(
            "DELETE FROM experiment_groups WHERE user_id = ? AND batch_id = ?",
            (uid, bid),
        )
        _normalize_group_sort_orders(conn, uid)

    rows = db.query_all("SELECT * FROM tasks WHERE user_id = ? ORDER BY id ASC", (uid,))
    ungrouped_count = 0
    for r in rows:
        try:
            cfg = json.loads(r.get("config_json") or "{}")
        except Exception:
            cfg = {}
        exp = dict(cfg.get("experiment") or {})
        if str(r.get("batch_id") or exp.get("batch_id") or "").strip() == bid:
            exp.pop("batch_id", None)
            exp.pop("batch_name", None)
            cfg["experiment"] = exp
            _persist_task_config(r, cfg)
            ungrouped_count += 1

    return {
        "dissolved_batch_id": bid,
        "ungrouped_tasks": ungrouped_count,
        "lab_groups": get_user_group_meta(uid),
    }


def _get_asset_seq_files(asset_row: dict | None) -> list[str]:
    """Return full validation sequence files ONLY when the dataset actually has >= 2 continuous eval frames."""
    if not asset_row:
        return []
    if asset_row.get("kind") == "eval_sequence":
        seq_rel = str(asset_row.get("source_path") or asset_row.get("rel_path") or "")
    else:
        seq_rel = str(asset_row.get("seq_path") or "")
    if not seq_rel:
        return []
    files = storage.expand_sequence_files(seq_rel)
    return files if len(files) >= 2 else []


def resolve_eval_asset(
    user_id: int,
    is_admin: bool,
    eval_asset_id: int | None,
    tasks_for_fallback: list[dict] | None = None,
) -> dict | None:
    if eval_asset_id:
        row = db.query_one("SELECT * FROM assets WHERE id = ?", (int(eval_asset_id),))
        if not row:
            raise ValueError(i18n.t("server.assetNotFound"))
        if not is_admin and row["user_id"] != user_id:
            raise PermissionError(i18n.t("server.assetAccessForbidden"))
        return row

    # Auto-detect dataset from selected tasks when eval_asset_id is omitted
    for t in tasks_for_fallback or []:
        cfg = (t or {}).get("config") or {}
        exp_meta = cfg.get("experiment") or {}
        for cand_id in (exp_meta.get("eval_asset_id"), exp_meta.get("dataset_id")):
            if cand_id:
                row = db.query_one("SELECT * FROM assets WHERE id = ?", (int(cand_id),))
                if row and (is_admin or row["user_id"] == user_id):
                    return row
        src_p = str((cfg.get("paths") or {}).get("source") or "").strip()
        if src_p:
            if is_admin:
                row = db.query_one(
                    "SELECT * FROM assets WHERE source_path = ? ORDER BY user_id = ? DESC, id ASC LIMIT 1",
                    (src_p, user_id),
                )
            else:
                row = db.query_one(
                    "SELECT * FROM assets WHERE user_id = ? AND source_path = ? ORDER BY id ASC LIMIT 1",
                    (user_id, src_p),
                )
            if row:
                return row
    return None


def compute_eval_key(ckpt_step: int, asset_row: dict | None) -> str:
    seq_files = _get_asset_seq_files(asset_row)
    if not asset_row or not seq_files:
        return f"step{int(ckpt_step)}_sup_only"
    aid = int(asset_row["id"])
    seq_rel = (
        str(asset_row.get("source_path") or "")
        if asset_row.get("kind") == "eval_sequence"
        else str(asset_row.get("seq_path") or "")
    )
    fc = len(seq_files)
    try:
        mt = int(os.path.getmtime(seq_files[0]))
    except OSError:
        mt = 0
    sig = hashlib.sha1(f"{seq_rel}|{fc}|{mt}".encode("utf-8")).hexdigest()[:8]
    return f"step{int(ckpt_step)}_seq{aid}_{sig}"


def resolve_eval_checkpoint(run_dir_rel: str, cfg: dict, ckpt_mode: str = "best") -> tuple[str | None, int]:
    """Return `(ckpt_abs_path, step)` for `best` (lowest val_mse) or `latest` saved checkpoint."""
    rdir_abs = storage.abs_from_root(run_dir_rel)
    if not os.path.isdir(rdir_abs):
        return None, 0
    parsed = telemetry.parse_train_log_full_cached(os.path.join(rdir_abs, "train.log"))
    inv = telemetry.list_task_models(run_dir_rel, cfg, parsed_log=parsed)
    items = inv.get("items") or []

    target_step = 0
    if items:
        if ckpt_mode == "best":
            best_items = [it for it in items if it.get("is_best")]
            target_step = int((best_items[0] if best_items else items[0])["step"])
        else:
            target_step = int(items[0]["step"])

    ckpt_dir = os.path.join(rdir_abs, "ckpt")
    if target_step > 0:
        cand_step = os.path.join(ckpt_dir, f"step_{target_step:06d}.pt")
        if os.path.isfile(cand_step):
            return cand_step, target_step
        # Check exported TorchScript Training_*.<step>.pt
        for p in glob.glob(os.path.join(rdir_abs, f"Training_*.{target_step}.pt")):
            if os.path.isfile(p):
                return p, target_step

    lat = os.path.join(ckpt_dir, "latest.pt")
    if os.path.isfile(lat):
        sp = telemetry.inspect_ckpt_cached(lat)
        return lat, int(sp.get("step") or target_step or 0)

    step_pts = sorted(glob.glob(os.path.join(ckpt_dir, "step_*.pt")))
    if step_pts:
        sp = telemetry.inspect_ckpt_cached(step_pts[-1])
        return step_pts[-1], int(sp.get("step") or 0)

    return None, 0


def load_cached_eval(
    run_dir_rel: str,
    node_name: str,
    eval_key: str,
    task_id: int | None = None,
    user_id: int | None = None,
) -> dict | None:
    if not eval_key:
        return None
    rdir_abs = storage.abs_from_root(run_dir_rel)
    eval_dir = os.path.join(rdir_abs, "eval", eval_key)
    m_path = os.path.join(eval_dir, "metrics.json")

    if task_id is not None:
        db_eval = analytics_db.load_eval_from_db(int(task_id), node_name, run_dir_rel, eval_key)
        if db_eval is not None and not os.path.isfile(m_path):
            return db_eval
        if db_eval is not None and os.path.isfile(m_path):
            try:
                mt = round(os.path.getmtime(m_path), 3)
                ev_meta = db.query_one(
                    "SELECT file_mtime FROM experiment_evals WHERE task_id = ? AND eval_key = ?",
                    (int(task_id), eval_key),
                )
                if ev_meta and abs(float(ev_meta["file_mtime"] or 0.0) - mt) < 0.002:
                    return db_eval
            except OSError:
                return db_eval

    if not os.path.isfile(m_path):
        return None
    try:
        mt = round(os.path.getmtime(m_path), 3)
        with open(m_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return None

    if task_id is not None and user_id is not None:
        analytics_db.sync_eval_result_to_db(int(task_id), int(user_id), eval_key, data, file_mtime=mt)
        db_eval = analytics_db.load_eval_from_db(int(task_id), node_name, run_dir_rel, eval_key)
        if db_eval is not None:
            return db_eval

    seq = data.get("sequence")
    if isinstance(seq, dict) and isinstance(seq.get("per_frame"), list):
        base_url = f"/f/run/{quote(node_name)}/eval/{quote(eval_key)}"
        has_err = bool(seq.get("is_paired"))
        out_dir_abs = os.path.join(eval_dir, "out")
        frames_on_disk = os.path.isdir(out_dir_abs)
        seq["frames_on_disk"] = frames_on_disk
        if frames_on_disk:
            for row in seq["per_frame"]:
                fn = row.get("file") or f"frame_{int(row.get('idx', 0)):04d}.jpg"
                row["out_url"] = f"{base_url}/out/{quote(fn)}"
                row["flicker_url"] = f"{base_url}/flicker/{quote(fn)}"
                if has_err:
                    row["err_url"] = f"{base_url}/err/{quote(fn)}"
    return data


def batch_submit_experiments(
    user_id: int,
    username: str,
    batch_name: str,
    eval_asset_id: int | None,
    items: list[dict],
    target_batch_id: str | None = None,
    is_admin: bool = False,
    client_ip: str = "",
) -> dict:
    if not items:
        raise ValueError(i18n.t("server.expEmptyBatch"))
    if len(items) > 64:
        raise ValueError(i18n.t("server.expBatchTooLarge"))

    uid = int(user_id)
    u_row = db.query_one(
        "SELECT role, max_tasks, submitted_task_count, max_epochs FROM users WHERE id = ?",
        (uid,),
    )
    restrict_on = str(db.get_setting("enable_temp_user_restrictions", "1") or "1").strip() != "0"
    is_privileged = bool(is_admin or (u_row and u_row.get("role") == "admin") or not restrict_on)
    if not is_privileged and u_row:
        max_t = int(u_row.get("max_tasks") if u_row.get("max_tasks") is not None else 2)
        used_t = int(u_row.get("submitted_task_count") or 0)
        max_ep = int(u_row.get("max_epochs") if u_row.get("max_epochs") is not None else 500000)
        existing_nodes = {
            r["node_name"]
            for r in db.query_all("SELECT node_name FROM tasks WHERE user_id = ?", (uid,))
        }
        new_nodes = set()
        for idx, item in enumerate(items):
            nname = storage.sanitize_name(str(item.get("node_name") or f"Exp_{idx + 1}").strip(), "WildCat1")
            if nname not in existing_nodes:
                new_nodes.add(nname)
            tr_c = ((item.get("config") or {}).get("train")) or {}
            req_ep = int(tr_c.get("epochs") or 0)
            if max_ep > 0 and req_ep > max_ep:
                raise PermissionError(
                    i18n.t("server.tempUserMaxEpochsExceeded", max_epochs=max_ep, requested=req_ep)
                )
        if used_t + len(new_nodes) > max_t:
            raise PermissionError(
                i18n.t("server.tempUserMaxTasksExceeded", max_tasks=max_t, used=used_t)
            )
    analytics_db.migrate_legacy_groups_for_user(uid)
    req_bid = str(target_batch_id or "").strip()
    base_sort = 0
    now = time.time()

    if req_bid and req_bid != UNGROUPED_KEY:
        batch_id = req_bid
        grp_row = db.query_one(
            "SELECT * FROM experiment_groups WHERE user_id = ? AND batch_id = ?",
            (uid, batch_id),
        )
        clean_bname = (
            (batch_name or "").strip()
            or str((grp_row or {}).get("batch_name") or "").strip()
            or time.strftime("Exp_%m%d_%H%M")
        )
        existing_rows = db.query_all(
            "SELECT batch_id, sort_order, config_json FROM tasks WHERE user_id = ?",
            (uid,),
        )
        existing_cnt = 0
        for er in existing_rows:
            ebid = str(er.get("batch_id") or "").strip()
            esord = int(er.get("sort_order") or 0)
            if not ebid:
                try:
                    ecfg = json.loads(er.get("config_json") or "{}")
                except Exception:
                    ecfg = {}
                eexp = ecfg.get("experiment") or {}
                ebid = str(eexp.get("batch_id") or "").strip()
                esord = int(eexp.get("sort_order") or esord)
            if ebid == batch_id:
                existing_cnt += 1
                base_sort = max(base_sort, esord, existing_cnt)
    else:
        batch_id = f"exp_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        clean_bname = (batch_name or "").strip() or time.strftime("Exp_%m%d_%H%M")

    with db.transaction() as conn:
        existing_grp = conn.execute(
            "SELECT id FROM experiment_groups WHERE user_id = ? AND batch_id = ?",
            (uid, batch_id),
        ).fetchone()
        if existing_grp:
            conn.execute(
                "UPDATE experiment_groups SET batch_name = ?, updated_at = ? WHERE id = ?",
                (clean_bname, now, int(existing_grp["id"])),
            )
        else:
            min_row = conn.execute(
                "SELECT MIN(sort_order) FROM experiment_groups WHERE user_id = ?",
                (uid,),
            ).fetchone()
            top_sort = (int(min_row[0]) - 1) if (min_row and min_row[0] is not None) else 0
            conn.execute(
                """
                INSERT INTO experiment_groups (user_id, batch_id, batch_name, sort_order, note, created_at, updated_at)
                VALUES (?, ?, ?, ?, '', ?, ?)
                """,
                (uid, batch_id, clean_bname, top_sort, now, now),
            )
            _normalize_group_sort_orders(conn, uid)

    submitted = []
    total = len(items)

    for idx, item in enumerate(items):
        cfg = json.loads(json.dumps(item.get("config") or {}))
        prev_exp = cfg.get("experiment") or {}
        ds_id = prev_exp.get("dataset_id")
        eff_eval_id = int(eval_asset_id) if eval_asset_id else (int(ds_id) if ds_id else None)
        cfg["experiment"] = {
            "batch_id": batch_id,
            "batch_name": clean_bname,
            "label": str(item.get("label") or "").strip(),
            "note": str(item.get("note") or "").strip(),
            "dataset_id": int(ds_id) if ds_id else eff_eval_id,
            "eval_asset_id": eff_eval_id,
            "index": idx + 1,
            "total": total,
            "sort_order": base_sort + idx + 1,
        }
        node_name = str(item.get("node_name") or f"Exp_{idx + 1}").strip()
        pref_gpu = str(item.get("preferred_gpu") or "auto").strip()
        prio = int(item.get("priority") or 0)
        t = scheduler.SCHEDULER.submit_task(
            user_id=uid,
            username=username,
            node_name=node_name,
            job_type="train",
            cfg=cfg,
            preferred_gpu=pref_gpu,
            priority=prio,
            is_admin=is_admin,
            client_ip=client_ip,
        )
        submitted.append(t)

    return {
        "batch_id": batch_id,
        "batch_name": clean_bname,
        "count": len(submitted),
        "tasks": submitted,
        "lab_groups": get_user_group_meta(uid),
    }


def compare_tasks(
    user_id: int,
    is_admin: bool,
    task_ids: list[int],
    eval_asset_id: int | None = None,
    ckpt_mode: str = "best",
) -> dict:
    analytics_db.sync_pending_tasks_to_db(user_id=None if is_admin else user_id)
    mode = "best" if ckpt_mode == "best" else "latest"
    loaded_tasks = []
    seen = set()
    for raw_tid in task_ids or []:
        tid = int(raw_tid)
        if tid in seen:
            continue
        seen.add(tid)
        t = service.get_task_for_user(user_id, is_admin, task_id=tid)
        if t:
            loaded_tasks.append(t)

    asset_row = resolve_eval_asset(user_id, is_admin, eval_asset_id, tasks_for_fallback=loaded_tasks)

    asset_info = None
    if asset_row:
        seq_files = _get_asset_seq_files(asset_row)
        has_seq = len(seq_files) >= 2
        seq_urls = telemetry.batch_to_jpg_urls(seq_files, quality=88) if has_seq else []
        pair_s_files = (
            []
            if asset_row.get("kind") == "eval_sequence"
            else storage.expand_sequence_files(asset_row.get("source_path") or "")
        )
        asset_info = {
            "id": asset_row["id"],
            "name": asset_row["name"],
            "kind": asset_row["kind"],
            "demo_key": asset_row.get("demo_key") or "",
            "paired_frame_count": len(pair_s_files),
            "frame_count": len(seq_files),
            "seq_frame_count": len(seq_files),
            "has_eval_sequence": has_seq,
            "width": asset_row.get("width"),
            "height": asset_row.get("height"),
            "in_channels": asset_row.get("in_channels") or 3,
            "out_channels": asset_row.get("out_channels") or 3,
            "is_paired": False,
            "source_frames": [
                {"idx": i, "name": os.path.basename(fp), "url": u}
                for i, (fp, u) in enumerate(zip(seq_files, seq_urls))
            ],
            "target_frames": [],
        }

    qstat = scheduler.SCHEDULER.queue_status()
    runs_out = []
    for t in loaded_tasks:
        scheduler.SCHEDULER.enrich_task_with_queue(t, qstat=qstat)
        cfg = t.get("config") or {}
        tid = int(t["id"])
        t_uid = int(t["user_id"])
        rdir_abs = storage.abs_from_root(t["run_dir"])
        log_abs = os.path.join(rdir_abs, "train.log")

        db_curves = None
        if t["status"] in ("completed", "stopped", "failed", "cancelled") or t.get("files_pruned") or not os.path.isfile(log_abs):
            if os.path.isfile(log_abs) and not t.get("files_pruned"):
                analytics_db.sync_task_log_to_db(t)
            db_curves = analytics_db.load_task_curves_from_db(tid)

        if db_curves is not None:
            steps = db_curves["steps"]
            vals = db_curves["vals"]
            sm = db_curves["summary"]
            best_val = sm.get("best_val_mse")
            last_val = sm.get("last_val_mse")
            last_loss = sm.get("final_loss") if sm.get("final_loss") is not None else t.get("last_loss")
        else:
            parsed = telemetry.parse_train_log_full_cached(log_abs, max_chart_points=1800)
            vals = parsed.get("vals") or []
            best_val = min((v["mse"] for v in vals if v.get("mse") is not None), default=None)
            last_val = vals[-1]["mse"] if vals else None
            steps = parsed.get("steps") or []
            last_loss = steps[-1].get("total") if steps else t.get("last_loss")

        ckpt_abs, ckpt_step = resolve_eval_checkpoint(t["run_dir"], cfg, ckpt_mode=mode)
        eval_key = ""
        cached_eval = None
        if ckpt_abs:
            eval_key = compute_eval_key(ckpt_step, asset_row)
            cached_eval = load_cached_eval(
                t["run_dir"], t["node_name"], eval_key, task_id=tid, user_id=t_uid
            )
        if cached_eval is None:
            # Fallback to SQLite evaluation record (e.g., when heavy .pt checkpoints were archived to free disk space!)
            if asset_row:
                fb_ev = db.query_one(
                    """
                    SELECT eval_key, ckpt_step FROM experiment_evals
                    WHERE task_id = ? AND (eval_asset_id = ? OR eval_asset_id IS NULL)
                    ORDER BY (eval_asset_id = ?) DESC, (ckpt_mode = ?) DESC, updated_at DESC
                    LIMIT 1
                    """,
                    (tid, int(asset_row["id"]), int(asset_row["id"]), mode),
                )
            else:
                fb_ev = db.query_one(
                    """
                    SELECT eval_key, ckpt_step FROM experiment_evals
                    WHERE task_id = ?
                    ORDER BY (ckpt_mode = ?) DESC, updated_at DESC
                    LIMIT 1
                    """,
                    (tid, mode),
                )
            if fb_ev:
                eval_key = fb_ev["eval_key"]
                ckpt_step = ckpt_step or int(fb_ev["ckpt_step"] or 0)
                cached_eval = analytics_db.load_eval_from_db(tid, t["node_name"], t["run_dir"], eval_key)

        dur_sec = t.get("duration_sec")
        if dur_sec is None and t.get("started_at") and t.get("finished_at") and t["finished_at"] >= t["started_at"]:
            dur_sec = round((t["finished_at"] - t["started_at"]), 1)

        runs_out.append(
            {
                "task_id": tid,
                "node_name": t["node_name"],
                "status": t["status"],
                "files_pruned": bool(t.get("files_pruned")),
                "last_step": t.get("last_step") or 0,
                "total_steps": t.get("total_steps") or 0,
                "duration_sec": dur_sec,
                "has_ckpt": bool(ckpt_abs),
                "ckpt_step": ckpt_step,
                "eval_key": eval_key,
                "param_diff": summarize_config_diff(cfg),
                "best_val_mse": best_val,
                "last_val_mse": last_val,
                "last_train_loss": last_loss,
                "steps": steps,
                "vals": vals,
                "eval": cached_eval,
            }
        )

    return {
        "ckpt_mode": mode,
        "eval_asset": asset_info,
        "runs": runs_out,
        "eval_runner": EVAL_RUNNER.status(),
    }


class SequenceEvalRunner:
    """Single-worker background subprocess runner for sequence evaluation (0 persistent FastAPI VRAM)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._queue = collections.deque()
        self._active = None  # dict describing current job
        self._proc = None
        self._last_error = ""
        self._completed_in_batch = 0
        self._batch_total = 0
        self._worker = None

    def _ensure_worker(self):
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._loop, daemon=True, name="WildCatEvalRunner")
            self._worker.start()

    def _pick_eval_device(self) -> str:
        try:
            q = scheduler.SCHEDULER.queue_status()
            gpus = [g for g in (q.get("gpus") or []) if g.get("enabled", True)]
            if not gpus:
                return "cpu"
            has_running = int(q.get("running_count") or 0) > 0
            idle = [g for g in gpus if not g.get("busy")]
            if idle:
                if has_running:
                    best = min(
                        idle,
                        key=lambda g: (
                            0 if g.get("is_secondary") else (2 if g.get("is_primary") else 1),
                            -int(g.get("vram_mb") or scheduler._GPU_VRAM_MB.get(str(g["gpu_id"]), 0)),
                            int(g["gpu_id"]) if str(g["gpu_id"]).isdigit() else 0,
                        ),
                    )
                else:
                    best = min(
                        idle,
                        key=lambda g: (
                            0 if g.get("is_primary") else (1 if g.get("is_secondary") else 2),
                            -int(g.get("vram_mb") or scheduler._GPU_VRAM_MB.get(str(g["gpu_id"]), 0)),
                            int(g["gpu_id"]) if str(g["gpu_id"]).isdigit() else 0,
                        ),
                    )
                return f"cuda:{best['gpu_id']}"
            best = min(
                gpus,
                key=lambda g: (
                    0 if g.get("is_secondary") else (2 if g.get("is_primary") else 1),
                    int(g.get("active_count") or 0),
                    -int(g.get("vram_mb") or scheduler._GPU_VRAM_MB.get(str(g["gpu_id"]), 0)),
                ),
            )
            return f"cuda:{best['gpu_id']}"
        except Exception:
            pass
        return "cuda:0"

    def enqueue(
        self,
        user_id: int,
        is_admin: bool,
        task_ids: list[int],
        eval_asset_id: int | None = None,
        ckpt_mode: str = "best",
        force: bool = False,
    ) -> dict:
        mode = "best" if ckpt_mode == "best" else "latest"
        loaded_tasks = []
        for raw_tid in task_ids or []:
            t = service.get_task_for_user(user_id, is_admin, task_id=int(raw_tid))
            if t:
                loaded_tasks.append(t)

        asset_row = resolve_eval_asset(user_id, is_admin, eval_asset_id, tasks_for_fallback=loaded_tasks)
        seq_files = _get_asset_seq_files(asset_row)
        has_seq = len(seq_files) >= 2

        jobs = []
        skipped_cached = 0
        for t in loaded_tasks:
            cfg = t.get("config") or {}
            ckpt_abs, ckpt_step = resolve_eval_checkpoint(t["run_dir"], cfg, ckpt_mode=mode)
            if not ckpt_abs:
                continue
            eval_key = compute_eval_key(ckpt_step, asset_row)
            if not force and load_cached_eval(
                t["run_dir"], t["node_name"], eval_key, task_id=int(t["id"]), user_id=int(t["user_id"])
            ) is not None:
                skipped_cached += 1
                continue

            eval_src = ""
            eval_tgt = ""
            asset_name = asset_row["name"] if asset_row else ""
            if asset_row and has_seq:
                seq_rel = (
                    asset_row.get("source_path")
                    if asset_row.get("kind") == "eval_sequence"
                    else asset_row.get("seq_path")
                ) or ""
                if seq_rel:
                    eval_src = storage.abs_from_root(seq_rel)

            rdir_abs = storage.abs_from_root(t["run_dir"])
            eval_out_dir = os.path.join(rdir_abs, "eval", eval_key)
            jobs.append(
                {
                    "task_id": t["id"],
                    "user_id": t["user_id"],
                    "node_name": t["node_name"],
                    "ckpt_path": ckpt_abs,
                    "ckpt_step": ckpt_step,
                    "ckpt_mode": mode,
                    "task_config": cfg,
                    "eval_key": eval_key,
                    "eval_asset_id": asset_row["id"] if asset_row else None,
                    "eval_asset_name": asset_name,
                    "eval_source_dir": eval_src,
                    "eval_target_dir": eval_tgt,
                    "eval_out_dir": eval_out_dir,
                }
            )

        with self._cond:
            self._ensure_worker()
            self._last_error = ""
            existing_keys = {
                (j["task_id"], j["eval_key"]) for j in self._queue
            }
            if self._active:
                existing_keys.add((self._active["task_id"], self._active["eval_key"]))
            added = 0
            for j in jobs:
                k = (j["task_id"], j["eval_key"])
                if k not in existing_keys:
                    self._queue.append(j)
                    existing_keys.add(k)
                    added += 1
            if added > 0:
                if self._active is None:
                    self._completed_in_batch = 0
                    self._batch_total = len(self._queue)
                else:
                    self._batch_total += added
                self._cond.notify_all()

        st = self.status()
        st["added"] = added
        st["skipped_cached"] = skipped_cached
        return st

    def cancel(self) -> dict:
        with self._cond:
            self._queue.clear()
            proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
        return self.status()

    def status(self) -> dict:
        with self._lock:
            active = dict(self._active) if self._active else None
            q_len = len(self._queue)
            done_tasks = self._completed_in_batch
            tot_tasks = self._batch_total
            last_err = self._last_error

        frame_done = 0
        frame_total = 0
        phase = ""
        if active:
            prog_p = os.path.join(active["eval_out_dir"], "progress.json")
            if os.path.isfile(prog_p):
                try:
                    with open(prog_p, "r", encoding="utf-8") as fh:
                        pr = json.load(fh)
                    frame_done = int(pr.get("done") or 0)
                    frame_total = int(pr.get("total") or 0)
                    phase = str(pr.get("phase") or "")
                except Exception:
                    pass

        return {
            "busy": bool(active or q_len > 0),
            "current_task_id": active["task_id"] if active else None,
            "current_node": active["node_name"] if active else "",
            "phase": phase,
            "queued_tasks": q_len,
            "task_done": done_tasks,
            "task_total": tot_tasks,
            "frame_done": frame_done,
            "frame_total": frame_total,
            "last_error": last_err,
        }

    def _loop(self):
        while True:
            with self._cond:
                while not self._queue:
                    self._active = None
                    self._cond.wait()
                job = self._queue.popleft()
                self._active = job

            try:
                os.makedirs(job["eval_out_dir"], exist_ok=True)
                spec = dict(job)
                spec["device"] = self._pick_eval_device()
                spec_path = os.path.join(job["eval_out_dir"], "spec.json")
                with open(spec_path, "w", encoding="utf-8") as fh:
                    json.dump(spec, fh, ensure_ascii=False, indent=2)

                eval_env = storage.apply_subprocess_storage_env(os.environ.copy())
                proc = subprocess.Popen(
                    [scheduler.PYTHON_BIN, "-m", "wildcat.evaluator", spec_path],
                    cwd=db.ROOT,
                    env=eval_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                with self._lock:
                    self._proc = proc
                out, _ = proc.communicate()
                with self._lock:
                    self._proc = None

                if proc.returncode != 0:
                    msg = (out or "").strip()[-800:] or f"Evaluator exited with code {proc.returncode}"
                    sys.stderr.write(f"[WildCat EvalRunner ERROR] task #{job['task_id']}:\n{out}\n")
                    sys.stderr.flush()
                    with self._lock:
                        self._last_error = f"{job['node_name']}: {msg}"
                else:
                    m_path = os.path.join(job["eval_out_dir"], "metrics.json")
                    if os.path.isfile(m_path):
                        try:
                            mt = round(os.path.getmtime(m_path), 3)
                            with open(m_path, "r", encoding="utf-8") as mfh:
                                m_data = json.load(mfh)
                            t_uid = job.get("user_id")
                            if not t_uid:
                                tr = db.query_one("SELECT user_id FROM tasks WHERE id = ?", (int(job["task_id"]),))
                                t_uid = tr["user_id"] if tr else 1
                            analytics_db.sync_eval_result_to_db(
                                int(job["task_id"]), int(t_uid), str(job["eval_key"]), m_data, file_mtime=mt
                            )
                        except Exception:
                            pass
            except Exception as exc:
                tb = traceback.format_exc()
                sys.stderr.write(f"[WildCat EvalRunner EXCEPTION]\n{tb}\n")
                sys.stderr.flush()
                with self._lock:
                    self._last_error = f"{job.get('node_name', '')}: {exc}"
            finally:
                with self._lock:
                    self._completed_in_batch += 1
                    if not self._queue:
                        self._active = None


EVAL_RUNNER = SequenceEvalRunner()
