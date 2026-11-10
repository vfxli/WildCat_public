# -*- coding: utf-8 -*-
"""Decoupled Business Service Layer for Users, Admin Settings, Assets, and Tasks/Queue."""
import glob
import json
import os
import socket
import subprocess
import time
import zipfile

from . import db, scheduler, storage, telemetry


def get_user_by_id(user_id: int) -> dict | None:
    return db.query_one(
        "SELECT id, username, role, api_key, is_active, note, created_at FROM users WHERE id = ?",
        (user_id,),
    )


def get_user_by_api_key(api_key: str) -> dict | None:
    if not api_key:
        return None
    u = db.query_one(
        "SELECT id, username, role, api_key, is_active, note, created_at FROM users WHERE api_key = ?",
        (api_key.strip(),),
    )
    if u and u["is_active"]:
        return u
    return None


def _ip_machine_name(client_ip: str = "") -> str:
    ip = (client_ip or "").strip()
    if ip in ("", "127.0.0.1", "::1", "localhost", "testclient"):
        cand = socket.gethostname() or "local_host"
    else:
        cand = f"host_{ip.replace('.', '_').replace(':', '_')}"
    mname = storage.sanitize_name(cand, "")
    return mname if len(mname) >= 2 else "workstation"


def resolve_machine_name(raw_machine: str = "", client_ip: str = "") -> str:
    cand = (raw_machine or "").strip()
    if cand:
        mname = storage.sanitize_name(cand, "")
        if len(mname) >= 2:
            return mname
    return _ip_machine_name(client_ip)


def get_or_create_machine_user(raw_machine: str = "", client_ip: str = "") -> dict:
    mname = resolve_machine_name(raw_machine, client_ip)
    row = db.query_one("SELECT * FROM users WHERE username = ?", (mname,))
    if row:
        if not row["is_active"]:
            raise PermissionError(f"机器 [{mname}] 已被管理员禁用")
        return get_user_by_id(row["id"])

    db.execute(
        """
        INSERT OR IGNORE INTO users (username, password_hash, role, api_key, is_active, note, created_at)
        VALUES (?, 'machine_identity', 'user', ?, 1, ?, ?)
        """,
        (mname, db.generate_api_key(), f"IP: {client_ip}" if client_ip else "", time.time()),
    )
    row = db.query_one("SELECT * FROM users WHERE username = ?", (mname,))
    if not row:
        raise RuntimeError(f"无法创建或读取机器账户: {mname}")
    if not row["is_active"]:
        raise PermissionError(f"机器 [{mname}] 已被管理员禁用")
    storage.user_root(mname)
    seed_demo_assets_for_user(row["id"])
    return get_user_by_id(row["id"])


def seed_demo_assets_for_user(user_id: int, force: bool = False):
    """Register the bundled demo datasets for a user. Off by default (clean production install);
    set WILDCAT_SEED_DEMO=1 to enable, or pass force=True when triggered explicitly by user."""
    if not force and os.environ.get("WILDCAT_SEED_DEMO", "0") != "1":
        return
    paired_demos = [
    ]
    now = time.time()
    for dname, s_rel, t_rel in paired_demos:
        if os.path.isdir(storage.abs_from_root(s_rel)) and os.path.isdir(storage.abs_from_root(t_rel)):
            st = storage.inspect_paired_dirs(s_rel, t_rel)
            if st["frame_count"] > 0:
                db.execute(
                    """
                    INSERT OR IGNORE INTO assets (
                        user_id, name, kind, rel_path, source_path, target_path,
                        frame_count, width, height, in_channels, out_channels,
                        avg_mae, max_mae, strategy, size_bytes, created_at
                    ) VALUES (?, ?, 'paired_dataset', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'standard', ?, ?)
                    """,
                    (
                        user_id,
                        dname,
                        s_rel,
                        s_rel,
                        t_rel,
                        st["frame_count"],
                        st["width"],
                        st["height"],
                        st["in_channels"],
                        st["out_channels"],
                        st["avg_mae"],
                        st["max_mae"],
                        st["size_bytes"],
                        now,
                    ),
                )


def create_user_account(
    username: str,
    password: str = "",
    role: str = "user",
    note: str = "",
    seed_demos: bool = True,
) -> dict:
    uname = storage.sanitize_name(username, "")
    if not uname or len(uname) < 2:
        raise ValueError("机器名至少需要 2 位字母/数字/下划线")
    existing = db.query_one("SELECT id FROM users WHERE username = ?", (uname,))
    if existing:
        raise ValueError(f"机器名 '{uname}' 已存在")

    uid = db.execute(
        """
        INSERT INTO users (username, password_hash, role, api_key, is_active, note, created_at)
        VALUES (?, 'machine_identity', 'user', ?, 1, ?, ?)
        """,
        (uname, db.generate_api_key(), note or "", time.time()),
    )
    storage.user_root(uname)
    if seed_demos:
        seed_demo_assets_for_user(uid)
    return get_user_by_id(uid)


def self_register_user(username: str, password: str, note: str = "") -> dict:
    allow = str(db.get_setting("allow_self_registration", "false")).lower() == "true"
    if not allow:
        raise PermissionError("Self-registration is currently disabled by administrator.")
    return create_user_account(username, password, role="user", note=note or "Self-registered")


def admin_update_user(
    user_id: int,
    password: str = None,
    role: str = None,
    is_active: bool = None,
    note: str = None,
) -> dict:
    u = db.query_one("SELECT * FROM users WHERE id = ?", (user_id,))
    if not u:
        raise ValueError("User not found.")
    if password:
        if len(password) < 4:
            raise ValueError("Password must be at least 4 characters.")
        db.execute("UPDATE users SET password_hash = ? WHERE id = ?", (db.hash_password(password), user_id))
    if role in ("user", "admin"):
        db.execute("UPDATE users SET role = ? WHERE id = ?", (role, user_id))
    if is_active is not None:
        db.execute("UPDATE users SET is_active = ? WHERE id = ?", (1 if is_active else 0, user_id))
    if note is not None:
        db.execute("UPDATE users SET note = ? WHERE id = ?", (note, user_id))
    return get_user_by_id(user_id)


def regenerate_user_api_key(user_id: int) -> str:
    new_key = db.generate_api_key()
    db.execute("UPDATE users SET api_key = ? WHERE id = ?", (new_key, user_id))
    return new_key


def admin_delete_user(user_id: int):
    u = get_user_by_id(user_id)
    if not u:
        return
    # Stop & delete all tasks belonging to this machine
    user_tasks = db.query_all("SELECT id FROM tasks WHERE user_id = ?", (user_id,))
    for t in user_tasks:
        scheduler.SCHEDULER.delete_task(t["id"])
    # Delete machine's uploaded assets directory
    user_assets = db.query_all("SELECT rel_path FROM assets WHERE user_id = ?", (user_id,))
    for a in user_assets:
        storage.delete_asset_files(a["rel_path"])
    db.execute("DELETE FROM users WHERE id = ?", (user_id,))


def list_all_users_with_stats() -> list:
    users = db.query_all(
        "SELECT id, username, role, api_key, is_active, note, created_at FROM users ORDER BY id ASC"
    )
    for u in users:
        uid = u["id"]
        u["asset_count"] = (
            db.query_one("SELECT COUNT(*) AS c FROM assets WHERE user_id = ?", (uid,)) or {}
        ).get("c", 0)
        u["task_count"] = (
            db.query_one("SELECT COUNT(*) AS c FROM tasks WHERE user_id = ?", (uid,)) or {}
        ).get("c", 0)
    return users


# ─────────────────────── Asset Management ───────────────────────


def list_assets(user_id: int, all_users: bool = False) -> list:
    if all_users:
        rows = db.query_all(
            """
            SELECT a.*, u.username
            FROM assets a
            JOIN users u ON a.user_id = u.id
            ORDER BY a.created_at DESC, a.id DESC
            """
        )
    else:
        rows = db.query_all(
            """
            SELECT a.*, u.username
            FROM assets a
            JOIN users u ON a.user_id = u.id
            WHERE a.user_id = ?
            ORDER BY a.created_at DESC, a.id DESC
            """,
            (user_id,),
        )
    for r in rows:
        sp = r.get("source_path") or r.get("rel_path") or ""
        tp = r.get("target_path") or r.get("rel_path") or ""
        s_files = storage.expand_sequence_files(sp)
        t_files = storage.expand_sequence_files(tp)
        r["frame_count"] = min(len(s_files), len(t_files)) if (s_files and t_files) else len(s_files)
        ext = os.path.splitext(s_files[0])[1].lower().lstrip(".") if s_files else "exr"
        r["file_format"] = ext.upper()
        r["colorspace"] = "ACEScg" if ext == "exr" else "sRGB"
        r["source_thumb_url"] = telemetry.to_jpg_url(s_files[0]) if s_files else None
        r["target_thumb_url"] = telemetry.to_jpg_url(t_files[0]) if t_files else None
        r["thumb_url"] = r["target_thumb_url"] or r["source_thumb_url"]
    return rows


def create_paired_dataset(
    user_id: int,
    username: str,
    dataset_name: str,
    source_tuples: list,
    target_tuples: list,
    max_error_pct: float = 35.0,
    strategy: str = "standard",
    enable_check: bool = True,
    input_colorspace: str = "ACEScg",
    working_colorspace: str = "ACEScg",
) -> dict:
    clean_name = storage.sanitize_name(dataset_name, "dataset")
    info = storage.validate_and_save_paired_dataset(
        username=username,
        dataset_name=clean_name,
        source_tuples=source_tuples,
        target_tuples=target_tuples,
        max_error_pct=max_error_pct,
        strategy=strategy,
        enable_check=enable_check,
        input_colorspace=input_colorspace,
        working_colorspace=working_colorspace,
    )
    now = time.time()
    existing = db.query_one(
        "SELECT id FROM assets WHERE user_id = ? AND name = ?", (user_id, clean_name)
    )
    if existing:
        aid = existing["id"]
        db.execute(
            """
            UPDATE assets
            SET kind = 'paired_dataset', rel_path = ?, source_path = ?, target_path = ?,
                frame_count = ?, width = ?, height = ?, in_channels = ?, out_channels = ?,
                avg_mae = ?, max_mae = ?, strategy = ?, size_bytes = ?, created_at = ?
            WHERE id = ?
            """,
            (
                info["rel_path"],
                info["source_path"],
                info["target_path"],
                info["frame_count"],
                info["width"],
                info["height"],
                info["in_channels"],
                info["out_channels"],
                info["avg_mae"],
                info["max_mae"],
                info["strategy"],
                info["size_bytes"],
                now,
                aid,
            ),
        )
    else:
        aid = db.execute(
            """
            INSERT INTO assets (
                user_id, name, kind, rel_path, source_path, target_path,
                frame_count, width, height, in_channels, out_channels,
                avg_mae, max_mae, strategy, size_bytes, created_at
            ) VALUES (?, ?, 'paired_dataset', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                clean_name,
                info["rel_path"],
                info["source_path"],
                info["target_path"],
                info["frame_count"],
                info["width"],
                info["height"],
                info["in_channels"],
                info["out_channels"],
                info["avg_mae"],
                info["max_mae"],
                info["strategy"],
                info["size_bytes"],
                now,
            ),
        )
    row = db.query_one("SELECT * FROM assets WHERE id = ?", (aid,))
    s_files = storage.expand_sequence_files(row["source_path"])
    t_files = storage.expand_sequence_files(row["target_path"])
    row["source_thumb_url"] = telemetry.to_jpg_url(s_files[0]) if s_files else None
    row["target_thumb_url"] = telemetry.to_jpg_url(t_files[0]) if t_files else None
    row["thumb_url"] = row["target_thumb_url"] or row["source_thumb_url"]
    return row


def delete_asset(user_id: int, is_admin: bool, asset_id: int):
    row = db.query_one("SELECT * FROM assets WHERE id = ?", (asset_id,))
    if not row:
        raise ValueError("Asset not found.")
    if not is_admin and row["user_id"] != user_id:
        raise PermissionError("You can only delete your own assets.")
    storage.delete_asset_files(row["rel_path"])
    db.execute("DELETE FROM assets WHERE id = ?", (asset_id,))


# ─────────────────────── Task & Queue Management ───────────────────────


def list_tasks(user_id: int, all_users: bool = False) -> list:
    qstat = scheduler.SCHEDULER.queue_status()
    qpos_map = {q["id"]: q["queue_position"] for q in qstat["queued"]}

    if all_users:
        rows = db.query_all("SELECT * FROM tasks ORDER BY id DESC")
    else:
        rows = db.query_all("SELECT * FROM tasks WHERE user_id = ? ORDER BY id DESC", (user_id,))

    for r in rows:
        r["queue_position"] = qpos_map.get(r["id"])
        try:
            r["config"] = json.loads(r.get("config_json") or "{}")
        except Exception:
            r["config"] = {}
        rdir_abs = storage.abs_from_root(r["run_dir"])
        has_nuke_models = bool(glob.glob(os.path.join(rdir_abs, "Training_*.[0-9]*.pt")))
        r["has_ckpt"] = os.path.isfile(os.path.join(rdir_abs, "ckpt", "latest.pt")) or has_nuke_models
        r["size_mb"] = round(storage.dir_total_bytes(r["run_dir"]) / (1024 * 1024), 1)
    return rows


def get_task_for_user(user_id: int, is_admin: bool, task_id: int = None, node_name: str = None) -> dict | None:
    if task_id is not None:
        row = db.query_one("SELECT * FROM tasks WHERE id = ?", (task_id,))
    else:
        row = db.query_one("SELECT * FROM tasks WHERE node_name = ?", (node_name,))
    if not row:
        return None
    if not is_admin and row["user_id"] != user_id:
        return None
    try:
        row["config"] = json.loads(row.get("config_json") or "{}")
    except Exception:
        row["config"] = {}
    return row


def delete_user_task(user_id: int, is_admin: bool, task_id: int):
    row = db.query_one("SELECT * FROM tasks WHERE id = ?", (task_id,))
    if not row:
        raise ValueError("Task not found.")
    if not is_admin and row["user_id"] != user_id:
        raise PermissionError("You can only delete your own tasks.")
    scheduler.SCHEDULER.delete_task(task_id)


def stop_user_task(user_id: int, is_admin: bool, task_id: int) -> dict:
    row = db.query_one("SELECT * FROM tasks WHERE id = ?", (task_id,))
    if not row:
        raise ValueError("Task not found.")
    if not is_admin and row["user_id"] != user_id:
        raise PermissionError("You can only stop your own tasks.")
    return scheduler.SCHEDULER.stop_task(task_id)


def export_task_cat(user_id: int, is_admin: bool, task_id: int, step: int = None) -> dict:
    row = get_task_for_user(user_id, is_admin, task_id=task_id)
    if not row:
        raise ValueError("Task not found or access denied.")
    rdir_abs = storage.abs_from_root(row["run_dir"])
    pfx = telemetry.resolve_nuke_prefix(rdir_abs)
    inv = telemetry.list_task_models(row["run_dir"], row.get("config"))
    target_step = int(step) if step is not None else (inv["items"][0]["step"] if inv["items"] else int(row.get("last_step") or 0))
    nuke_pt = os.path.join(rdir_abs, f"{pfx}.{target_step}.pt") if target_step > 0 else os.path.join(rdir_abs, f"{row['node_name']}.pt")
    cmd = [scheduler.PYTHON_BIN, "export_cat.py", row["config_path"]]
    if step is not None and os.path.isfile(os.path.join(rdir_abs, "ckpt", f"step_{int(step):06d}.pt")):
        cmd.append(str(int(step)))
    cmd.extend(["--out-pt", nuke_pt])
    res = subprocess.run(
        cmd,
        cwd=db.ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=90,
    )
    if res.returncode != 0:
        raise RuntimeError((res.stderr or "") + "\n" + (res.stdout or "") or "Export failed")
    nname = row["node_name"]
    return {
        "node_name": nname,
        "step": target_step,
        "nuke_prefix": pfx,
        "pt_path": storage.rel_to_root(nuke_pt),
        "pt_filename": os.path.basename(nuke_pt),
        "message": res.stdout.strip(),
    }


def ensure_nuke_model_file(
    user_id: int,
    is_admin: bool,
    task_id: int,
    ext: str,
    step: int = None,
) -> tuple[str, str]:
    """Ensure the Nuke-named model file (Training_YYMMDD_HHMMSS.<step>.cat/.pt) exists on disk
    for the requested step (or latest step when step is None) and return (abs_path, download_filename)."""
    row = get_task_for_user(user_id, is_admin, task_id=task_id)
    if not row:
        raise ValueError("任务不存在或无权访问。")
    rdir_abs = storage.abs_from_root(row["run_dir"])
    inv = telemetry.list_task_models(row["run_dir"], row.get("config"))
    if not inv["items"]:
        raise ValueError("当前任务尚未保存任何模型检查点，请等待到达 Checkpoint Interval 或先暂停任务保存进度。")

    if step is None:
        item = inv["items"][0]
    else:
        st_req = int(step)
        item = next((x for x in inv["items"] if x["step"] == st_req), None)
        if not item:
            raise ValueError(f"未找到 Step {st_req} 对应的模型版本。")

    st = int(item["step"])
    target_fn = item["pt_filename"]
    target_abs = os.path.join(rdir_abs, target_fn)
    if os.path.isfile(target_abs) and os.path.getsize(target_abs) > 0:
        return target_abs, target_fn

    export_task_cat(user_id, is_admin, task_id, step=st)
    if not os.path.isfile(target_abs) or os.path.getsize(target_abs) == 0:
        raise FileNotFoundError(f"模型文件导出后未找到: {target_fn}")
    return target_abs, target_fn


def build_all_models_zip(user_id: int, is_admin: bool, task_id: int, ext: str) -> tuple[str, str]:
    """Bundle all checkpoint step models of this task (strictly Nuke-named) into a fast ZIP_STORED archive."""
    row = get_task_for_user(user_id, is_admin, task_id=task_id)
    if not row:
        raise ValueError("任务不存在或无权访问。")
    rdir_abs = storage.abs_from_root(row["run_dir"])
    inv = telemetry.list_task_models(row["run_dir"], row.get("config"))
    if not inv["items"]:
        raise ValueError("当前任务尚未生成任何可下载的模型版本。")

    ordered_items = sorted(inv["items"], key=lambda x: x["step"])
    files_to_pack = []
    for it in ordered_items:
        f_abs, f_name = ensure_nuke_model_file(user_id, is_admin, task_id, ext, step=it["step"])
        files_to_pack.append((f_abs, f_name))

    pfx = inv["nuke_prefix"]
    latest_st = ordered_items[-1]["step"]
    bundle_name = f".bundle_{pfx}_{ext}_{len(files_to_pack)}_{latest_st}.zip"
    bundle_abs = os.path.join(rdir_abs, bundle_name)
    dl_name = f"{pfx}.zip"

    if not os.path.isfile(bundle_abs) or os.path.getsize(bundle_abs) == 0:
        for old_b in glob.glob(os.path.join(rdir_abs, f".bundle_{pfx}_{ext}_*.zip")):
            if old_b != bundle_abs:
                try:
                    os.remove(old_b)
                except OSError:
                    pass
        tmp_zip = f"{bundle_abs}.tmp.{os.getpid()}"
        with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_STORED) as zf:
            for f_abs, f_name in files_to_pack:
                zf.write(f_abs, arcname=f_name)
        os.replace(tmp_zip, bundle_abs)

    return bundle_abs, dl_name
