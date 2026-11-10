# -*- coding: utf-8 -*-
"""SQLite Database Layer for Multi-User Auth, System Settings, User Assets, and Task Queue."""
import hashlib
import json
import os
import secrets
import sqlite3
import threading
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STORAGE_ROOT = os.environ.get("WC_STORAGE_ROOT") or os.path.join(ROOT, "storage")
DB_PATH = os.path.join(STORAGE_ROOT, "wildcat.db")

_DB_LOCK = threading.RLock()


def _connect():
    os.makedirs(STORAGE_ROOT, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


# Precomputed PBKDF2-HMAC-SHA256 salted hash (no plaintext password stored in source code or DB)
_DEFAULT_ADMIN_HASH = ""  # no built-in password: set WC_PASSWORD or use the one generated on first start


def hash_password(password: str, salt: str = None) -> str:
    if not salt:
        salt = secrets.token_hex(8)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000)
    return f"{salt}${dk.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    if not stored_hash or "$" not in stored_hash:
        return False
    salt, _ = stored_hash.split("$", 1)
    return secrets.compare_digest(hash_password(password, salt), stored_hash)


def generate_api_key() -> str:
    return "wc_" + secrets.token_urlsafe(24)


def init_db():
    with _DB_LOCK:
        conn = _connect()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS system_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'user',
                    api_key TEXT UNIQUE NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    note TEXT DEFAULT '',
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS assets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    kind TEXT NOT NULL DEFAULT 'paired_dataset',
                    rel_path TEXT NOT NULL,
                    source_path TEXT NOT NULL DEFAULT '',
                    target_path TEXT NOT NULL DEFAULT '',
                    frame_count INTEGER NOT NULL DEFAULT 0,
                    width INTEGER NOT NULL DEFAULT 0,
                    height INTEGER NOT NULL DEFAULT 0,
                    in_channels INTEGER NOT NULL DEFAULT 3,
                    out_channels INTEGER NOT NULL DEFAULT 3,
                    avg_mae REAL NOT NULL DEFAULT 0.0,
                    max_mae REAL NOT NULL DEFAULT 0.0,
                    strategy TEXT NOT NULL DEFAULT 'standard',
                    size_bytes INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    UNIQUE(user_id, name),
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS tasks (
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
                """
            )

            # Ensure new paired-dataset columns exist on existing SQLite DB
            existing_cols = {
                r["name"] for r in conn.execute("PRAGMA table_info(assets)").fetchall()
            }
            new_cols = [
                ("source_path", "TEXT NOT NULL DEFAULT ''"),
                ("target_path", "TEXT NOT NULL DEFAULT ''"),
                ("width", "INTEGER NOT NULL DEFAULT 0"),
                ("height", "INTEGER NOT NULL DEFAULT 0"),
                ("in_channels", "INTEGER NOT NULL DEFAULT 3"),
                ("out_channels", "INTEGER NOT NULL DEFAULT 3"),
                ("avg_mae", "REAL NOT NULL DEFAULT 0.0"),
                ("max_mae", "REAL NOT NULL DEFAULT 0.0"),
                ("strategy", "TEXT NOT NULL DEFAULT 'standard'"),
            ]
            for cname, cdef in new_cols:
                if cname not in existing_cols:
                    conn.execute(f"ALTER TABLE assets ADD COLUMN {cname} {cdef}")

            # Clean up legacy unpaired demo entries where source_path or target_path is empty
            conn.execute("DELETE FROM assets WHERE source_path = '' OR target_path = ''")

            now = time.time()
            # Default system settings (admin password stored strictly as salted PBKDF2-SHA256 hash)
            env_pwd = os.environ.get("WC_PASSWORD", "").strip()
            generated_pwd = ""
            if not env_pwd:
                generated_pwd = secrets.token_urlsafe(12)
            init_admin_hash = hash_password(env_pwd or generated_pwd)
            defaults = {
                "max_jobs_per_gpu": "1",
                "default_gpu_policy": "auto",
                "admin_password_hash": init_admin_hash,
                "admin_session_token": secrets.token_urlsafe(32),
            }
            for k, v in defaults.items():
                conn.execute(
                    "INSERT OR IGNORE INTO system_settings (key, value, updated_at) VALUES (?, ?, ?)",
                    (k, v, now),
                )
            conn.commit()
            if generated_pwd:
                row = conn.execute("SELECT value FROM system_settings WHERE key = 'admin_password_hash'").fetchone()
                if row and verify_password(generated_pwd, row[0]):
                    print("[WildCat] 管理员初始密码（首次启动自动生成，登录管理后台后请修改）: %s" % generated_pwd, flush=True)
        finally:
            conn.close()


def verify_admin_password(password: str) -> bool:
    stored = get_setting("admin_password_hash", _DEFAULT_ADMIN_HASH)
    return verify_password(password or "", stored)


def update_admin_password(new_password: str):
    if not new_password or len(new_password) < 4:
        raise ValueError("管理员密码长度至少 4 位")
    set_setting("admin_password_hash", hash_password(new_password))
    set_setting("admin_session_token", secrets.token_urlsafe(32))


def issue_admin_token() -> str:
    tok = get_setting("admin_session_token", "")
    if not tok:
        tok = secrets.token_urlsafe(32)
        set_setting("admin_session_token", tok)
    return tok


def verify_admin_token(token: str) -> bool:
    if not token:
        return False
    stored = get_setting("admin_session_token", "")
    return bool(stored and secrets.compare_digest(token.strip(), stored))


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
            conn.execute(
                """
                INSERT INTO system_settings (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                """,
                (key, str(value), time.time()),
            )
            conn.commit()
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
                if r["key"] not in ("admin_password_hash", "admin_session_token")
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
            cur = conn.execute(sql, params)
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()
