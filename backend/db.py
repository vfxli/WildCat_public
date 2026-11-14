# -*- coding: utf-8 -*-
"""SQLite Database Layer for Multi-User Auth, System Settings, User Assets, and Task Queue."""
import contextlib
import hashlib
import hmac
import json
import os
import secrets
import shutil
import sqlite3
import threading
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STORAGE_ROOT = os.environ.get("WC_STORAGE_ROOT") or os.path.join(ROOT, "storage")
DB_PATH = os.environ.get("WC_DB_PATH") or os.path.join(STORAGE_ROOT, "wildcat.db")
EXP_DB_PATH = os.environ.get("WC_EXP_DB_PATH") or os.path.join(STORAGE_ROOT, "experiments.db")
OPEN_SEED_EXP_DB_PATH = os.environ.get("WC_OPEN_SEED_EXP_DB") or os.path.join(ROOT, "data", "open_experiments.db")

_DB_LOCK = threading.RLock()


def _connect():
    os.makedirs(STORAGE_ROOT, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(EXP_DB_PATH)), exist_ok=True)
    default_storage = os.path.join(ROOT, "storage")
    allow_seed = os.environ.get("WC_SKIP_OPEN_SEED") != "1" and (
        os.path.abspath(STORAGE_ROOT) == os.path.abspath(default_storage)
        or os.environ.get("WC_ENABLE_OPEN_SEED") == "1"
    )
    if allow_seed and not os.path.exists(EXP_DB_PATH) and os.path.isfile(OPEN_SEED_EXP_DB_PATH):
        shutil.copy2(OPEN_SEED_EXP_DB_PATH, EXP_DB_PATH)
    conn = sqlite3.connect(DB_PATH, timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000;")
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("ATTACH DATABASE ? AS exp;", (EXP_DB_PATH,))
    conn.execute("PRAGMA exp.journal_mode=WAL;")
    return conn


# Precomputed PBKDF2-HMAC-SHA256 salted hash (no plaintext password stored in source code or DB)
_DEFAULT_ADMIN_HASH = ""  # no built-in password: set WC_PASSWORD or use the one generated on first start


MAX_PASSWORD_CHARS = 128
ADMIN_TOKEN_TTL_SEC = 24 * 3600


def hash_password(password: str, salt: str = None) -> str:
    if not salt:
        salt = secrets.token_hex(8)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000)
    return f"{salt}${dk.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    pwd = str(password or "")
    if len(pwd) > MAX_PASSWORD_CHARS or not stored_hash or "$" not in stored_hash:
        # Constant-work dummy hash prevents timing oracle & multi-MB PBKDF2 CPU exhaustion
        _ = hash_password("wc_dummy_constant_work", "0000000000000000")
        return False
    salt, _ = stored_hash.split("$", 1)
    return secrets.compare_digest(hash_password(pwd, salt), stored_hash)


def generate_api_key() -> str:
    return "wc_" + secrets.token_urlsafe(24)


def init_db():
    with _DB_LOCK:
        default_storage = os.path.join(ROOT, "storage")
        allow_seed = os.environ.get("WC_SKIP_OPEN_SEED") != "1" and (
            os.path.abspath(STORAGE_ROOT) == os.path.abspath(default_storage)
            or os.environ.get("WC_ENABLE_OPEN_SEED") == "1"
        )
        if allow_seed and not os.path.exists(EXP_DB_PATH) and os.path.isfile(OPEN_SEED_EXP_DB_PATH):
            os.makedirs(os.path.dirname(os.path.abspath(EXP_DB_PATH)), exist_ok=True)
            shutil.copy2(OPEN_SEED_EXP_DB_PATH, EXP_DB_PATH)

        need_vacuum_main = False
        conn = _connect()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS main.system_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS main.users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'user',
                    api_key TEXT UNIQUE NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    note TEXT DEFAULT '',
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS main.assets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    kind TEXT NOT NULL DEFAULT 'paired_dataset',
                    rel_path TEXT NOT NULL,
                    source_path TEXT NOT NULL DEFAULT '',
                    target_path TEXT NOT NULL DEFAULT '',
                    seq_path TEXT NOT NULL DEFAULT '',
                    frame_count INTEGER NOT NULL DEFAULT 0,
                    seq_frame_count INTEGER NOT NULL DEFAULT 0,
                    demo_key TEXT NOT NULL DEFAULT '',
                    width INTEGER NOT NULL DEFAULT 0,
                    height INTEGER NOT NULL DEFAULT 0,
                    in_channels INTEGER NOT NULL DEFAULT 3,
                    out_channels INTEGER NOT NULL DEFAULT 3,
                    avg_mae REAL NOT NULL DEFAULT 0.0,
                    max_mae REAL NOT NULL DEFAULT 0.0,
                    strategy TEXT NOT NULL DEFAULT 'standard',
                    task_type TEXT NOT NULL DEFAULT '',
                    size_bytes INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    UNIQUE(user_id, name),
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS main.tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    username TEXT NOT NULL,
                    node_name TEXT NOT NULL,
                    job_type TEXT NOT NULL DEFAULT 'train',
                    status TEXT NOT NULL DEFAULT 'queued',
                    priority INTEGER NOT NULL DEFAULT 0,
                    preferred_gpu TEXT NOT NULL DEFAULT 'auto',
                    assigned_gpu TEXT DEFAULT NULL,
                    pid INTEGER DEFAULT NULL,
                    config_path TEXT NOT NULL,
                    run_dir TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    last_step INTEGER NOT NULL DEFAULT 0,
                    total_steps INTEGER NOT NULL DEFAULT 0,
                    last_loss REAL DEFAULT NULL,
                    error_msg TEXT DEFAULT '',
                    created_at REAL NOT NULL,
                    started_at REAL DEFAULT NULL,
                    finished_at REAL DEFAULT NULL,
                    UNIQUE(user_id, node_name),
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS main.identity_bindings (
                    bind_key TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    ip_scope TEXT NOT NULL DEFAULT '',
                    updated_at REAL NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS main.ip_quota_ledger (
                    ip_scope TEXT PRIMARY KEY,
                    anon_users_created INTEGER NOT NULL DEFAULT 0,
                    anon_tasks_submitted INTEGER NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS main.auth_lockouts (
                    scope_key TEXT PRIMARY KEY,
                    fail_count INTEGER NOT NULL DEFAULT 0,
                    first_fail_at REAL NOT NULL DEFAULT 0.0,
                    last_fail_at REAL NOT NULL DEFAULT 0.0,
                    locked_until REAL NOT NULL DEFAULT 0.0
                );

                CREATE TABLE IF NOT EXISTS exp.experiment_runs (
                    task_id INTEGER PRIMARY KEY,
                    user_id INTEGER NOT NULL DEFAULT 1,
                    node_name TEXT UNIQUE NOT NULL,
                    status TEXT NOT NULL DEFAULT 'completed',
                    batch_id TEXT NOT NULL DEFAULT '',
                    batch_name TEXT NOT NULL DEFAULT '',
                    sort_order INTEGER NOT NULL DEFAULT 0,
                    exp_label TEXT NOT NULL DEFAULT '',
                    exp_note TEXT NOT NULL DEFAULT '',
                    dataset_name TEXT NOT NULL DEFAULT '',
                    dataset_frames INTEGER NOT NULL DEFAULT 0,
                    width INTEGER NOT NULL DEFAULT 0,
                    height INTEGER NOT NULL DEFAULT 0,
                    model_size TEXT NOT NULL DEFAULT '',
                    in_channels INTEGER NOT NULL DEFAULT 3,
                    out_channels INTEGER NOT NULL DEFAULT 3,
                    epochs INTEGER NOT NULL DEFAULT 0,
                    last_step INTEGER NOT NULL DEFAULT 0,
                    total_steps INTEGER NOT NULL DEFAULT 0,
                    last_loss REAL DEFAULT NULL,
                    batch_size INTEGER NOT NULL DEFAULT 0,
                    crop_size INTEGER NOT NULL DEFAULT 256,
                    multi_resolution INTEGER NOT NULL DEFAULT 1,
                    precision_mode TEXT NOT NULL DEFAULT 'adaptive',
                    seed INTEGER NOT NULL DEFAULT 0,
                    is_nuke_native INTEGER NOT NULL DEFAULT 1,
                    adv_crop_grid INTEGER NOT NULL DEFAULT 4,
                    adv_loss_enabled INTEGER NOT NULL DEFAULT 0,
                    adv_charb_ratio REAL NOT NULL DEFAULT 0.0,
                    adv_grad_weight REAL NOT NULL DEFAULT 0.0,
                    adv_hdr_alpha REAL NOT NULL DEFAULT 0.0,
                    adv_stage_enabled INTEGER NOT NULL DEFAULT 0,
                    adv_r_quarter REAL NOT NULL DEFAULT 33.333,
                    adv_r_half REAL NOT NULL DEFAULT 33.333,
                    adv_r_full REAL NOT NULL DEFAULT 33.334,
                    adv_warmup_steps INTEGER NOT NULL DEFAULT 0,
                    adv_ema_enabled INTEGER NOT NULL DEFAULT 0,
                    adv_ema_decay REAL NOT NULL DEFAULT 0.0,
                    adv_aug_enabled INTEGER NOT NULL DEFAULT 0,
                    adv_hflip_prob REAL NOT NULL DEFAULT 0.0,
                    adv_exposure_ev REAL NOT NULL DEFAULT 0.0,
                    duration_sec REAL DEFAULT NULL,
                    avg_sps REAL DEFAULT NULL,
                    best_val_mse REAL DEFAULT NULL,
                    best_val_step INTEGER DEFAULT NULL,
                    last_val_mse REAL DEFAULT NULL,
                    best_psnr_srgb REAL DEFAULT NULL,
                    best_psnr_aces REAL DEFAULT NULL,
                    best_edge_l1 REAL DEFAULT NULL,
                    best_seq_hfi REAL DEFAULT NULL,
                    best_seq_delta REAL DEFAULT NULL,
                    config_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL DEFAULT 0.0,
                    started_at REAL DEFAULT NULL,
                    finished_at REAL DEFAULT NULL,
                    updated_at REAL NOT NULL DEFAULT 0.0
                );

                CREATE TABLE IF NOT EXISTS exp.experiment_groups (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    batch_id TEXT NOT NULL,
                    batch_name TEXT NOT NULL,
                    sort_order INTEGER NOT NULL DEFAULT 0,
                    note TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(user_id, batch_id)
                );

                CREATE TABLE IF NOT EXISTS exp.task_metrics (
                    task_id INTEGER PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    synced_step INTEGER NOT NULL DEFAULT 0,
                    log_mtime REAL NOT NULL DEFAULT 0.0,
                    step_points INTEGER NOT NULL DEFAULT 0,
                    val_points INTEGER NOT NULL DEFAULT 0,
                    min_train_loss REAL DEFAULT NULL,
                    final_loss REAL DEFAULT NULL,
                    best_val_mse REAL DEFAULT NULL,
                    best_val_step INTEGER DEFAULT NULL,
                    last_val_mse REAL DEFAULT NULL,
                    avg_sps REAL DEFAULT NULL,
                    max_sps REAL DEFAULT NULL,
                    is_final INTEGER NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS exp.task_loss_points (
                    task_id INTEGER NOT NULL,
                    step INTEGER NOT NULL,
                    scale INTEGER NOT NULL DEFAULT 1,
                    total_loss REAL NOT NULL,
                    lr REAL DEFAULT NULL,
                    sps REAL DEFAULT NULL,
                    PRIMARY KEY(task_id, step)
                );

                CREATE TABLE IF NOT EXISTS exp.task_val_points (
                    task_id INTEGER NOT NULL,
                    step INTEGER NOT NULL,
                    val_mse REAL NOT NULL,
                    PRIMARY KEY(task_id, step)
                );

                CREATE TABLE IF NOT EXISTS exp.experiment_evals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    eval_key TEXT NOT NULL,
                    ckpt_mode TEXT NOT NULL DEFAULT 'best',
                    ckpt_step INTEGER NOT NULL DEFAULT 0,
                    ckpt_file TEXT NOT NULL DEFAULT '',
                    eval_asset_id INTEGER DEFAULT NULL,
                    eval_asset_name TEXT NOT NULL DEFAULT '',
                    device_name TEXT NOT NULL DEFAULT '',
                    sup_frames INTEGER NOT NULL DEFAULT 0,
                    mse REAL DEFAULT NULL,
                    mae REAL DEFAULT NULL,
                    psnr_aces REAL DEFAULT NULL,
                    psnr_srgb REAL DEFAULT NULL,
                    edge_l1 REAL DEFAULT NULL,
                    p99_err REAL DEFAULT NULL,
                    bg_drift_mae REAL DEFAULT NULL,
                    seq_frames INTEGER NOT NULL DEFAULT 0,
                    is_paired_seq INTEGER NOT NULL DEFAULT 0,
                    seq_mse REAL DEFAULT NULL,
                    seq_mae REAL DEFAULT NULL,
                    seq_psnr_aces REAL DEFAULT NULL,
                    seq_psnr_srgb REAL DEFAULT NULL,
                    seq_edge_l1 REAL DEFAULT NULL,
                    seq_p99_err REAL DEFAULT NULL,
                    seq_bg_drift_mae REAL DEFAULT NULL,
                    edit_l1 REAL DEFAULT NULL,
                    flicker_delta REAL DEFAULT NULL,
                    flicker_delta_max REAL DEFAULT NULL,
                    flicker_hf REAL DEFAULT NULL,
                    flicker_static REAL DEFAULT NULL,
                    peak_flicker_idx INTEGER DEFAULT NULL,
                    peak_flicker_val REAL DEFAULT NULL,
                    elapsed_sec REAL DEFAULT NULL,
                    evaluated_at TEXT NOT NULL DEFAULT '',
                    file_mtime REAL NOT NULL DEFAULT 0.0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(task_id, eval_key)
                );

                CREATE TABLE IF NOT EXISTS exp.experiment_eval_frames (
                    eval_id INTEGER NOT NULL,
                    frame_idx INTEGER NOT NULL,
                    frame_name TEXT NOT NULL DEFAULT '',
                    file_name TEXT NOT NULL DEFAULT '',
                    edit_l1 REAL DEFAULT NULL,
                    flicker_delta REAL DEFAULT NULL,
                    flicker_hf REAL DEFAULT NULL,
                    flicker_static REAL DEFAULT NULL,
                    mse REAL DEFAULT NULL,
                    psnr_srgb REAL DEFAULT NULL,
                    psnr_aces REAL DEFAULT NULL,
                    PRIMARY KEY(eval_id, frame_idx),
                    FOREIGN KEY(eval_id) REFERENCES experiment_evals(id) ON DELETE CASCADE
                );
                """
            )

            # One-time automatic split migration: move legacy experiment tables out of main (wildcat.db) into exp (experiments.db)
            legacy_tables_ordered = [
                "experiment_groups",
                "task_metrics",
                "task_loss_points",
                "task_val_points",
                "experiment_evals",
                "experiment_eval_frames",
            ]
            existing_main_exp = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM main.sqlite_master WHERE type='table'"
                ).fetchall()
            } & set(legacy_tables_ordered)
            if existing_main_exp:
                conn.execute("PRAGMA foreign_keys=OFF;")
                for tbl in legacy_tables_ordered:
                    if tbl in existing_main_exp:
                        conn.execute(f"INSERT OR IGNORE INTO exp.{tbl} SELECT * FROM main.{tbl}")
                for tbl in reversed(legacy_tables_ordered):
                    if tbl in existing_main_exp:
                        conn.execute(f"DROP TABLE IF EXISTS main.{tbl}")
                conn.execute("PRAGMA foreign_keys=ON;")
                need_vacuum_main = True

            # Idempotent schema migrations for existing SQLite databases
            user_cols = {r[1] for r in conn.execute("PRAGMA main.table_info(users)").fetchall()}
            user_migrations = [
                ("recovery_code", "TEXT NOT NULL DEFAULT ''"),
                ("recovery_code_hash", "TEXT NOT NULL DEFAULT ''"),
                ("max_tasks", "INTEGER NOT NULL DEFAULT 2"),
                ("submitted_task_count", "INTEGER NOT NULL DEFAULT 0"),
                ("max_epochs", "INTEGER NOT NULL DEFAULT 500000"),
                ("perm_lab", "INTEGER NOT NULL DEFAULT 0"),
                ("perm_api", "INTEGER NOT NULL DEFAULT 0"),
                ("perm_converter", "INTEGER NOT NULL DEFAULT 0"),
                ("device_fp_hash", "TEXT NOT NULL DEFAULT ''"),
                ("created_ip", "TEXT NOT NULL DEFAULT ''"),
                ("last_ip", "TEXT NOT NULL DEFAULT ''"),
                ("last_seen_at", "REAL NOT NULL DEFAULT 0.0"),
            ]
            for col_name, col_def in user_migrations:
                if col_name not in user_cols:
                    conn.execute(f"ALTER TABLE main.users ADD COLUMN {col_name} {col_def}")
            asset_cols = {r[1] for r in conn.execute("PRAGMA main.table_info(assets)").fetchall()}
            if "seq_path" not in asset_cols:
                conn.execute("ALTER TABLE main.assets ADD COLUMN seq_path TEXT NOT NULL DEFAULT ''")
            if "seq_frame_count" not in asset_cols:
                conn.execute("ALTER TABLE main.assets ADD COLUMN seq_frame_count INTEGER NOT NULL DEFAULT 0")
            if "demo_key" not in asset_cols:
                conn.execute("ALTER TABLE main.assets ADD COLUMN demo_key TEXT NOT NULL DEFAULT ''")
            if "task_type" not in asset_cols:
                conn.execute("ALTER TABLE main.assets ADD COLUMN task_type TEXT NOT NULL DEFAULT ''")

            # Strict numeric hyperparameter & KPI columns on tasks (all with DEFAULT so legacy INSERTs never fail)
            task_cols = {r[1] for r in conn.execute("PRAGMA main.table_info(tasks)").fetchall()}
            task_migrations = [
                ("dataset_id", "INTEGER DEFAULT NULL"),
                ("dataset_name", "TEXT NOT NULL DEFAULT ''"),
                ("dataset_frames", "INTEGER NOT NULL DEFAULT 0"),
                ("width", "INTEGER NOT NULL DEFAULT 0"),
                ("height", "INTEGER NOT NULL DEFAULT 0"),
                ("model_size", "TEXT NOT NULL DEFAULT ''"),
                ("in_channels", "INTEGER NOT NULL DEFAULT 3"),
                ("out_channels", "INTEGER NOT NULL DEFAULT 3"),
                ("epochs", "INTEGER NOT NULL DEFAULT 0"),
                ("batch_size", "INTEGER NOT NULL DEFAULT 0"),
                ("crop_size", "INTEGER NOT NULL DEFAULT 256"),
                ("multi_resolution", "INTEGER NOT NULL DEFAULT 1"),
                ("precision_mode", "TEXT NOT NULL DEFAULT 'adaptive'"),
                ("seed", "INTEGER NOT NULL DEFAULT 0"),
                ("is_nuke_native", "INTEGER NOT NULL DEFAULT 1"),
                ("adv_crop_grid", "INTEGER NOT NULL DEFAULT 4"),
                ("adv_loss_enabled", "INTEGER NOT NULL DEFAULT 0"),
                ("adv_charb_ratio", "REAL NOT NULL DEFAULT 0.0"),
                ("adv_grad_weight", "REAL NOT NULL DEFAULT 0.0"),
                ("adv_hdr_alpha", "REAL NOT NULL DEFAULT 0.0"),
                ("adv_stage_enabled", "INTEGER NOT NULL DEFAULT 0"),
                ("adv_r_quarter", "REAL NOT NULL DEFAULT 33.333"),
                ("adv_r_half", "REAL NOT NULL DEFAULT 33.333"),
                ("adv_r_full", "REAL NOT NULL DEFAULT 33.334"),
                ("adv_warmup_steps", "INTEGER NOT NULL DEFAULT 0"),
                ("adv_ema_enabled", "INTEGER NOT NULL DEFAULT 0"),
                ("adv_ema_decay", "REAL NOT NULL DEFAULT 0.0"),
                ("adv_aug_enabled", "INTEGER NOT NULL DEFAULT 0"),
                ("adv_hflip_prob", "REAL NOT NULL DEFAULT 0.0"),
                ("adv_exposure_ev", "REAL NOT NULL DEFAULT 0.0"),
                ("batch_id", "TEXT NOT NULL DEFAULT ''"),
                ("batch_name", "TEXT NOT NULL DEFAULT ''"),
                ("sort_order", "INTEGER NOT NULL DEFAULT 0"),
                ("exp_label", "TEXT NOT NULL DEFAULT ''"),
                ("exp_note", "TEXT NOT NULL DEFAULT ''"),
                ("duration_sec", "REAL DEFAULT NULL"),
                ("avg_sps", "REAL DEFAULT NULL"),
                ("best_val_mse", "REAL DEFAULT NULL"),
                ("best_val_step", "INTEGER DEFAULT NULL"),
                ("last_val_mse", "REAL DEFAULT NULL"),
                ("best_psnr_srgb", "REAL DEFAULT NULL"),
                ("best_psnr_aces", "REAL DEFAULT NULL"),
                ("best_edge_l1", "REAL DEFAULT NULL"),
                ("best_seq_hfi", "REAL DEFAULT NULL"),
                ("best_seq_delta", "REAL DEFAULT NULL"),
                ("files_pruned", "INTEGER NOT NULL DEFAULT 0"),
                ("analytics_synced_at", "REAL NOT NULL DEFAULT 0.0"),
            ]
            for col_name, col_def in task_migrations:
                if col_name not in task_cols:
                    conn.execute(f"ALTER TABLE main.tasks ADD COLUMN {col_name} {col_def}")

            conn.executescript(
                """
                CREATE INDEX IF NOT EXISTS main.idx_tasks_user_status ON tasks(user_id, status);
                CREATE INDEX IF NOT EXISTS main.idx_tasks_user_batch ON tasks(user_id, batch_id, sort_order);
                CREATE INDEX IF NOT EXISTS main.idx_tasks_user_dataset ON tasks(user_id, dataset_id);
                CREATE INDEX IF NOT EXISTS main.idx_tasks_user_psnr ON tasks(user_id, best_psnr_srgb);
                CREATE INDEX IF NOT EXISTS main.idx_tasks_user_hfi ON tasks(user_id, best_seq_hfi);
                CREATE INDEX IF NOT EXISTS main.idx_users_recovery_hash ON users(recovery_code_hash);
                CREATE INDEX IF NOT EXISTS main.idx_identity_bindings_user ON identity_bindings(user_id);

                CREATE INDEX IF NOT EXISTS exp.idx_exp_runs_batch ON experiment_runs(batch_id, sort_order);
                CREATE INDEX IF NOT EXISTS exp.idx_exp_runs_dataset ON experiment_runs(dataset_name);
                CREATE INDEX IF NOT EXISTS exp.idx_exp_groups_user_order ON experiment_groups(user_id, sort_order);
                CREATE INDEX IF NOT EXISTS exp.idx_task_metrics_user_val ON task_metrics(user_id, best_val_mse);
                CREATE INDEX IF NOT EXISTS exp.idx_exp_evals_user_asset ON experiment_evals(user_id, eval_asset_id);
                CREATE INDEX IF NOT EXISTS exp.idx_exp_evals_task ON experiment_evals(task_id);
                CREATE INDEX IF NOT EXISTS exp.idx_exp_evals_psnr ON experiment_evals(user_id, psnr_srgb);
                CREATE INDEX IF NOT EXISTS exp.idx_exp_evals_hfi ON experiment_evals(user_id, flicker_hf);
                """
            )

            now = time.time()
            # Default system settings (admin password stored strictly as salted PBKDF2-SHA256 hash)
            env_pwd = os.environ.get("WC_PASSWORD", "").strip()
            env_cache_root = os.environ.get("WC_CACHE_ROOT", "").strip()
            generated_pwd = ""
            if not env_pwd:
                generated_pwd = secrets.token_urlsafe(12)
            init_admin_hash = hash_password(env_pwd or generated_pwd)
            defaults = {
                "max_jobs_per_gpu": "1",
                "default_gpu_policy": "auto",
                "data_cache_root": env_cache_root,
                "enable_temp_user_restrictions": "1",
                "default_user_max_tasks": "2",
                "default_user_max_epochs": "500000",
                "session_hmac_secret": secrets.token_hex(32),
                "admin_password_hash": init_admin_hash,
                "admin_session_token": secrets.token_urlsafe(32),
            }
            for k, v in defaults.items():
                conn.execute(
                    "INSERT OR IGNORE INTO main.system_settings (key, value, updated_at) VALUES (?, ?, ?)",
                    (k, v, now),
                )

            # Ensure built-in admin account exists and backfill roaming codes & quotas for legacy accounts
            conn.execute(
                """
                INSERT OR IGNORE INTO main.users (
                    username, password_hash, role, api_key, is_active, note, created_at,
                    max_tasks, max_epochs, perm_lab, perm_api, perm_converter
                ) VALUES ('admin', 'admin_builtin', 'admin', ?, 1, 'System Administrator', ?, 99999, 10000000, 1, 1, 1)
                """,
                (generate_api_key(), now),
            )
            # Fresh open-source clone hydration: if main.tasks is empty and exp.experiment_runs has seeded benchmark runs
            main_task_cnt = int(conn.execute("SELECT COUNT(*) FROM main.tasks").fetchone()[0] or 0)
            exp_run_cnt = int(conn.execute("SELECT COUNT(*) FROM exp.experiment_runs").fetchone()[0] or 0)
            if main_task_cnt == 0 and exp_run_cnt > 0:
                admin_row = conn.execute("SELECT id, username FROM main.users WHERE role = 'admin' ORDER BY id ASC LIMIT 1").fetchone()
                seed_uid = int(admin_row["id"]) if admin_row else 1
                seed_uname = str(admin_row["username"]) if admin_row else "admin"
                conn.execute(
                    """
                    INSERT OR IGNORE INTO main.tasks (
                        id, user_id, username, node_name, job_type, status, priority,
                        preferred_gpu, assigned_gpu, pid, config_path, run_dir, config_json,
                        last_step, total_steps, last_loss, error_msg, created_at, started_at, finished_at,
                        dataset_name, dataset_frames, width, height, model_size, in_channels, out_channels,
                        epochs, batch_size, crop_size, multi_resolution, precision_mode, seed, is_nuke_native,
                        adv_crop_grid, adv_loss_enabled, adv_charb_ratio, adv_grad_weight, adv_hdr_alpha,
                        adv_stage_enabled, adv_r_quarter, adv_r_half, adv_r_full, adv_warmup_steps,
                        adv_ema_enabled, adv_ema_decay, adv_aug_enabled, adv_hflip_prob, adv_exposure_ev,
                        batch_id, batch_name, sort_order, exp_label, exp_note,
                        duration_sec, avg_sps, best_val_mse, best_val_step, last_val_mse,
                        best_psnr_srgb, best_psnr_aces, best_edge_l1, best_seq_hfi, best_seq_delta,
                        files_pruned, analytics_synced_at
                    )
                    SELECT
                        task_id, ?, ?, node_name, 'train', status, 0,
                        'benchmark', 'benchmark', NULL,
                        'config/' || node_name || '.json', 'run/' || node_name, config_json,
                        last_step, total_steps, last_loss, '', created_at, started_at, finished_at,
                        dataset_name, dataset_frames, width, height, model_size, in_channels, out_channels,
                        epochs, batch_size, crop_size, multi_resolution, precision_mode, seed, is_nuke_native,
                        adv_crop_grid, adv_loss_enabled, adv_charb_ratio, adv_grad_weight, adv_hdr_alpha,
                        adv_stage_enabled, adv_r_quarter, adv_r_half, adv_r_full, adv_warmup_steps,
                        adv_ema_enabled, adv_ema_decay, adv_aug_enabled, adv_hflip_prob, adv_exposure_ev,
                        batch_id, batch_name, sort_order, exp_label, exp_note,
                        duration_sec, avg_sps, best_val_mse, best_val_step, last_val_mse,
                        best_psnr_srgb, best_psnr_aces, best_edge_l1, best_seq_hfi, best_seq_delta,
                        1, updated_at
                    FROM exp.experiment_runs
                    """,
                    (seed_uid, seed_uname),
                )

            legacy_users = conn.execute(
                "SELECT id, username, role FROM main.users WHERE recovery_code = ''"
            ).fetchall()
            if legacy_users:
                from . import security as _sec

                for lu in legacy_users:
                    luid = int(lu["id"])
                    luname = str(lu["username"])
                    rcode = _sec.generate_recovery_code(luname)
                    rhash = _sec.hash_recovery_code(rcode)
                    t_cnt = int(
                        conn.execute("SELECT COUNT(*) FROM main.tasks WHERE user_id = ?", (luid,)).fetchone()[0] or 0
                    )
                    if lu["role"] == "admin" or luname in ("admin", "DLI-DESKTOP"):
                        conn.execute(
                            """
                            UPDATE main.users
                            SET recovery_code = ?, recovery_code_hash = ?,
                                max_tasks = 99999, submitted_task_count = ?, max_epochs = 10000000,
                                perm_lab = 1, perm_api = 1, perm_converter = 1
                            WHERE id = ?
                            """,
                            (rcode, rhash, t_cnt, luid),
                        )
                    else:
                        conn.execute(
                            """
                            UPDATE main.users
                            SET recovery_code = ?, recovery_code_hash = ?, submitted_task_count = ?
                            WHERE id = ?
                            """,
                            (rcode, rhash, t_cnt, luid),
                        )
            conn.commit()
            if generated_pwd:
                row = conn.execute("SELECT value FROM main.system_settings WHERE key = 'admin_password_hash'").fetchone()
                if row and verify_password(generated_pwd, row[0]):
                    from . import i18n
                    print(i18n.t("server.initAdminPasswordBanner", password=generated_pwd), flush=True)
        finally:
            conn.close()

        if need_vacuum_main:
            vconn = sqlite3.connect(DB_PATH, timeout=30.0)
            try:
                vconn.execute("VACUUM;")
                vconn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            finally:
                vconn.close()


def verify_admin_password(password: str, min_fail_delay_sec: float = 0.25) -> bool:
    t0 = time.perf_counter()
    stored = get_setting("admin_password_hash", _DEFAULT_ADMIN_HASH)
    ok = verify_password(password or "", stored)
    if not ok and min_fail_delay_sec > 0:
        rem = float(min_fail_delay_sec) - (time.perf_counter() - t0)
        if rem > 0:
            time.sleep(rem)
    return ok


def update_admin_password(new_password: str):
    if not new_password or len(new_password) < 8:
        from . import i18n

        raise ValueError(i18n.t("server.adminPasswordMinLen"))
    if len(new_password) > MAX_PASSWORD_CHARS:
        from . import i18n

        raise ValueError(i18n.t("server.invalidTrainParam", field="new_password"))
    set_setting("admin_password_hash", hash_password(new_password))
    # Rotate signing secret to revoke all previously issued X-Admin-Token sessions immediately
    set_setting("admin_session_token", secrets.token_urlsafe(32))


def _get_admin_token_secret() -> bytes:
    sec = get_setting("admin_session_token", "")
    if not sec or len(sec) < 32:
        sec = secrets.token_urlsafe(32)
        set_setting("admin_session_token", sec)
    return sec.encode("utf-8")


def issue_admin_token() -> str:
    ts = int(time.time())
    nonce = secrets.token_hex(8)
    payload = f"{ts}.{nonce}"
    sig = hmac.new(_get_admin_token_secret(), f"wc_adm_v1|{payload}".encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{payload}.{sig}"


def verify_admin_token(token: str) -> bool:
    if not token:
        return False
    raw = str(token).strip()
    if len(raw) < 20 or len(raw) > 200:
        return False
    parts = raw.split(".")
    if len(parts) != 3:
        return False
    ts_s, nonce_s, sig_s = parts
    if not (ts_s.isdigit() and 8 <= len(nonce_s) <= 32 and len(sig_s) == 64):
        return False
    age = time.time() - int(ts_s)
    if age < -30 or age > ADMIN_TOKEN_TTL_SEC:
        return False
    expected = hmac.new(
        _get_admin_token_secret(),
        f"wc_adm_v1|{ts_s}.{nonce_s}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return secrets.compare_digest(sig_s.lower(), expected.lower())


def get_setting(key: str, default=None):
    with _DB_LOCK:
        conn = _connect()
        try:
            row = conn.execute("SELECT value FROM system_settings WHERE key = ?", (key,)).fetchone()
            return row["value"] if row else default
        finally:
            conn.close()


def set_setting(key: str, value: str):
    with _DB_LOCK:
        conn = _connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                INSERT INTO system_settings (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                """,
                (key, str(value), time.time()),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


def get_all_settings() -> dict:
    with _DB_LOCK:
        conn = _connect()
        try:
            rows = conn.execute("SELECT key, value FROM system_settings").fetchall()
            out = {
                r["key"]: r["value"]
                for r in rows
                if r["key"] not in ("admin_password_hash", "admin_session_token", "session_hmac_secret")
            }
            return out
        finally:
            conn.close()


def query_all(sql: str, params=()):
    with _DB_LOCK:
        conn = _connect()
        try:
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


def query_one(sql: str, params=()):
    with _DB_LOCK:
        conn = _connect()
        try:
            row = conn.execute(sql, params).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()


def execute(sql: str, params=()) -> int:
    with _DB_LOCK:
        conn = _connect()
        try:
            conn.isolation_level = None
            conn.execute("BEGIN IMMEDIATE;")
            cur = conn.execute(sql, params)
            conn.execute("COMMIT;")
            return cur.lastrowid
        except Exception:
            try:
                conn.execute("ROLLBACK;")
            except Exception:
                pass
            raise
        finally:
            conn.close()


@contextlib.contextmanager
def transaction():
    """Atomic WAL transaction context manager yielding a live sqlite3.Connection."""
    with _DB_LOCK:
        conn = _connect()
        try:
            conn.isolation_level = None
            conn.execute("BEGIN IMMEDIATE;")
            yield conn
            conn.execute("COMMIT;")
        except Exception:
            try:
                conn.execute("ROLLBACK;")
            except Exception:
                pass
            raise
        finally:
            conn.close()


def delete_task_experiment_data(task_id: int):
    """Cascade-clean attached `experiments.db` rows when a task is deleted from `wildcat.db`."""
    tid = int(task_id)
    with transaction() as conn:
        conn.execute(
            "DELETE FROM exp.experiment_eval_frames WHERE eval_id IN (SELECT id FROM exp.experiment_evals WHERE task_id = ?)",
            (tid,),
        )
        conn.execute("DELETE FROM exp.experiment_evals WHERE task_id = ?", (tid,))
        conn.execute("DELETE FROM exp.task_loss_points WHERE task_id = ?", (tid,))
        conn.execute("DELETE FROM exp.task_val_points WHERE task_id = ?", (tid,))
        conn.execute("DELETE FROM exp.task_metrics WHERE task_id = ?", (tid,))
        conn.execute("DELETE FROM exp.experiment_runs WHERE task_id = ?", (tid,))


def delete_user_experiment_data(user_id: int):
    """Cascade-clean attached `experiments.db` rows when a user is deleted from `wildcat.db`."""
    uid = int(user_id)
    with transaction() as conn:
        conn.execute(
            "DELETE FROM exp.experiment_eval_frames WHERE eval_id IN (SELECT id FROM exp.experiment_evals WHERE user_id = ?)",
            (uid,),
        )
        conn.execute("DELETE FROM exp.experiment_evals WHERE user_id = ?", (uid,))
        conn.execute("DELETE FROM exp.task_metrics WHERE user_id = ?", (uid,))
        conn.execute("DELETE FROM exp.experiment_groups WHERE user_id = ?", (uid,))
        conn.execute("DELETE FROM exp.experiment_runs WHERE user_id = ?", (uid,))
