# -*- coding: utf-8 -*-
"""Decoupled Business Service Layer for Users, Admin Settings, Assets, and Tasks/Queue."""
import glob
import json
import os
import secrets
import subprocess
import threading
import time
import zipfile

from . import analytics_db, db, i18n, scheduler, security, storage, telemetry

USER_SELECT_COLS = (
    "id, username, role, api_key, is_active, note, created_at, "
    "recovery_code, max_tasks, submitted_task_count, max_epochs, "
    "perm_lab, perm_api, perm_converter, device_fp_hash, created_ip, last_ip, last_seen_at"
)
MAX_ANON_USERS_PER_IP = 1
TEMP_USER_MAX_STORAGE_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB cap for temporary users
_MINT_LOCK = threading.Lock()


def is_temp_restrictions_enabled() -> bool:
    return str(db.get_setting("enable_temp_user_restrictions", "1") or "1").strip() != "0"


def _decorate_user(u: dict | None, is_admin: bool = False) -> dict | None:
    if not u:
        return None
    out = dict(u)
    restrict_on = is_temp_restrictions_enabled()
    priv = bool(is_admin or out.get("role") == "admin" or not restrict_on)
    max_t = int(out.get("max_tasks") if out.get("max_tasks") is not None else 2)
    used_t = int(out.get("submitted_task_count") or 0)
    max_ep = int(out.get("max_epochs") if out.get("max_epochs") is not None else 500000)
    raw_lab = bool(out.get("perm_lab"))
    raw_api = bool(out.get("perm_api"))
    raw_conv = bool(out.get("perm_converter"))
    out["raw_perm_lab"] = raw_lab
    out["raw_perm_api"] = raw_api
    out["raw_perm_converter"] = raw_conv
    out["perm_lab"] = bool(priv or raw_lab)
    out["perm_api"] = bool(priv or raw_api)
    out["perm_converter"] = bool(priv or raw_conv)
    out["max_tasks"] = max_t
    out["tasks_used"] = used_t
    out["tasks_remaining"] = 99999 if priv else max(0, max_t - used_t)
    out["max_epochs"] = 0 if not restrict_on else max_ep
    out["raw_max_epochs"] = max_ep
    out["restrictions_enabled"] = restrict_on
    out["is_admin"] = bool(is_admin)
    out["is_temp_user"] = bool(restrict_on and not (is_admin or out.get("role") == "admin") and not (raw_lab and raw_api))
    out.pop("device_fp_hash", None)
    return out


def get_user_by_id(user_id: int, is_admin: bool = False) -> dict | None:
    row = db.query_one(f"SELECT {USER_SELECT_COLS} FROM users WHERE id = ?", (int(user_id),))
    return _decorate_user(row, is_admin=is_admin)


def get_user_by_api_key(api_key: str, is_admin: bool = False) -> dict | None:
    if not api_key:
        return None
    row = db.query_one(
        f"SELECT {USER_SELECT_COLS} FROM users WHERE api_key = ?",
        (api_key.strip(),),
    )
    if row and row["is_active"]:
        return _decorate_user(row, is_admin=is_admin)
    return None


def _bind_device_identity(
    user_id: int,
    ip_scope: str,
    fp_key: str,
    now: float,
    prev_seen: float = 0.0,
    prev_ip: str = "",
    prev_fp: str = "",
) -> None:
    uid = int(user_id)
    ip_changed = bool(ip_scope and ip_scope != (prev_ip or ""))
    fp_changed = bool(fp_key and fp_key != (prev_fp or ""))
    need_touch = (now - float(prev_seen or 0.0)) >= 60.0 or ip_changed or fp_changed
    if not need_touch:
        return
    with db.transaction() as conn:
        if fp_key:
            conn.execute(
                """
                INSERT INTO identity_bindings (bind_key, user_id, ip_scope, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(bind_key) DO UPDATE SET user_id = excluded.user_id, ip_scope = excluded.ip_scope, updated_at = excluded.updated_at
                """,
                (fp_key, uid, ip_scope, now),
            )
        if ip_scope:
            conn.execute(
                """
                INSERT INTO identity_bindings (bind_key, user_id, ip_scope, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(bind_key) DO UPDATE SET user_id = excluded.user_id, ip_scope = excluded.ip_scope, updated_at = excluded.updated_at
                """,
                (f"ip:{ip_scope}", uid, ip_scope, now),
            )
        conn.execute(
            """
            UPDATE users
            SET last_ip = CASE WHEN ? != '' THEN ? ELSE last_ip END,
                device_fp_hash = CASE WHEN ? != '' THEN ? ELSE device_fp_hash END,
                last_seen_at = ?
            WHERE id = ?
            """,
            (ip_scope, ip_scope, fp_key, fp_key, now, uid),
        )


def _lookup_user_by_roam_code(raw_code: str) -> dict | None:
    norm = security.normalize_recovery_code(raw_code)
    if not norm or len(norm) < 8:
        return None
    rhash = security.hash_recovery_code(norm)
    if not rhash:
        return None
    row = db.query_one(
        f"SELECT {USER_SELECT_COLS} FROM users WHERE recovery_code_hash = ?",
        (rhash,),
    )
    if not row:
        return None
    stored_norm = security.normalize_recovery_code(row.get("recovery_code") or "")
    if not stored_norm or not secrets.compare_digest(norm, stored_norm):
        return None
    return row


def recover_user_by_code(raw_code: str, client_ip: str = "", raw_fp: str = "", is_admin: bool = False) -> dict:
    """Recover an existing server-minted user workspace via roaming code (`U-XXXXXX-YYYY-ZZZZ`). Never creates new accounts."""
    row = _lookup_user_by_roam_code(raw_code)
    if not row and is_admin and raw_code:
        row = db.query_one(
            f"SELECT {USER_SELECT_COLS} FROM users WHERE LOWER(username) = LOWER(?)",
            (str(raw_code).strip(),),
        )
    if not row:
        raise ValueError(i18n.t("server.recoveryCodeInvalid"))
    if not row["is_active"]:
        raise PermissionError(i18n.t("server.accountDisabled", username=row["username"]))
    ip_scope, fp_key = security.compute_device_fp_hash(client_ip, raw_fp)
    _bind_device_identity(row["id"], ip_scope, fp_key, time.time(), prev_seen=0.0)
    return get_user_by_id(row["id"], is_admin=is_admin)


def admin_bind_browser_to_user(target_uid: int, client_ip: str = "", raw_fp: str = "") -> dict | None:
    """Switch an administrator's current browser/IP binding & session directly to `target_uid`."""
    row = db.query_one(f"SELECT {USER_SELECT_COLS} FROM users WHERE id = ?", (int(target_uid),))
    if not row:
        return None
    ip_scope, fp_key = security.compute_device_fp_hash(client_ip, raw_fp)
    _bind_device_identity(int(target_uid), ip_scope, fp_key, time.time(), prev_seen=0.0)
    return _decorate_user(row, is_admin=True)


def resolve_or_create_session_user(
    cookie_uid: int | None = None,
    roam_code: str = "",
    client_ip: str = "",
    raw_fp: str = "",
    is_admin: bool = False,
) -> tuple[dict, bool]:
    """Resolve user identity without registration/password while blocking Incognito/cookie-clear/name-change quota bypass.
    Returns `(user_dict, should_set_cookie)`."""
    now = time.time()
    ip_scope, fp_key = security.compute_device_fp_hash(client_ip, raw_fp)

    # 1. Explicit cross-device roaming code (`X-Roam-Code`) wins over stale/proxy cookies
    if roam_code:
        row = _lookup_user_by_roam_code(roam_code)
        if row:
            if not row["is_active"]:
                raise PermissionError(i18n.t("server.accountDisabled", username=row["username"]))
            need_cookie = cookie_uid != int(row["id"])
            _bind_device_identity(
                row["id"], ip_scope, fp_key, now,
                prev_seen=0.0 if need_cookie else float(row.get("last_seen_at") or 0.0),
                prev_ip=str(row.get("last_ip") or ""),
                prev_fp=str(row.get("device_fp_hash") or ""),
            )
            return _decorate_user(row, is_admin=is_admin), need_cookie

    # 2. Verified signed HttpOnly session cookie
    if cookie_uid is not None:
        row = db.query_one(f"SELECT {USER_SELECT_COLS} FROM users WHERE id = ?", (int(cookie_uid),))
        if row:
            if not row["is_active"]:
                raise PermissionError(i18n.t("server.accountDisabled", username=row["username"]))
            _bind_device_identity(
                row["id"], ip_scope, fp_key, now,
                prev_seen=float(row.get("last_seen_at") or 0.0),
                prev_ip=str(row.get("last_ip") or ""),
                prev_fp=str(row.get("device_fp_hash") or ""),
            )
            return _decorate_user(row, is_admin=is_admin), False

    # 3. Hardware fingerprint + IP scope binding recognition (blocks Incognito, cookie clear, browser switch & header spoofing)
    if fp_key:
        fp_bind = db.query_one("SELECT user_id FROM identity_bindings WHERE bind_key = ?", (fp_key,))
        if fp_bind:
            row = db.query_one(f"SELECT {USER_SELECT_COLS} FROM users WHERE id = ?", (int(fp_bind["user_id"]),))
            if row:
                if not row["is_active"]:
                    raise PermissionError(i18n.t("server.accountDisabled", username=row["username"]))
                _bind_device_identity(row["id"], ip_scope, fp_key, now, prev_seen=0.0)
                return _decorate_user(row, is_admin=is_admin), True

    restrict_on = is_temp_restrictions_enabled()
    eff_max_anon = MAX_ANON_USERS_PER_IP if restrict_on else 64

    if ip_scope:
        ledger = db.query_one("SELECT anon_users_created FROM ip_quota_ledger WHERE ip_scope = ?", (ip_scope,))
        anon_cnt = int((ledger or {}).get("anon_users_created") or 0)
        if (not fp_key) or (anon_cnt >= eff_max_anon):
            ip_bind = db.query_one("SELECT user_id FROM identity_bindings WHERE bind_key = ?", (f"ip:{ip_scope}",))
            if ip_bind:
                row = db.query_one(f"SELECT {USER_SELECT_COLS} FROM users WHERE id = ?", (int(ip_bind["user_id"]),))
                if row:
                    if not row["is_active"]:
                        raise PermissionError(i18n.t("server.accountDisabled", username=row["username"]))
                    _bind_device_identity(row["id"], ip_scope, fp_key, now, prev_seen=0.0)
                    return _decorate_user(row, is_admin=is_admin), True

    # 4. Serialize & re-check inside _MINT_LOCK to prevent parallel curl race from minting multiple accounts per IP
    with _MINT_LOCK:
        if fp_key:
            fp_bind = db.query_one("SELECT user_id FROM identity_bindings WHERE bind_key = ?", (fp_key,))
            if fp_bind:
                row = db.query_one(f"SELECT {USER_SELECT_COLS} FROM users WHERE id = ?", (int(fp_bind["user_id"]),))
                if row:
                    if not row["is_active"]:
                        raise PermissionError(i18n.t("server.accountDisabled", username=row["username"]))
                    _bind_device_identity(row["id"], ip_scope, fp_key, now, prev_seen=0.0)
                    return _decorate_user(row, is_admin=is_admin), True

        if ip_scope:
            ledger = db.query_one("SELECT anon_users_created FROM ip_quota_ledger WHERE ip_scope = ?", (ip_scope,))
            anon_cnt = int((ledger or {}).get("anon_users_created") or 0)
            if (not fp_key) or (anon_cnt >= eff_max_anon):
                ip_bind = db.query_one("SELECT user_id FROM identity_bindings WHERE bind_key = ?", (f"ip:{ip_scope}",))
                if ip_bind:
                    row = db.query_one(f"SELECT {USER_SELECT_COLS} FROM users WHERE id = ?", (int(ip_bind["user_id"]),))
                    if row:
                        if not row["is_active"]:
                            raise PermissionError(i18n.t("server.accountDisabled", username=row["username"]))
                        _bind_device_identity(row["id"], ip_scope, fp_key, now, prev_seen=0.0)
                        return _decorate_user(row, is_admin=is_admin), True

        if restrict_on and not is_admin:
            security.enforce_rate_limit("mint_user", client_ip, limit=6, window_sec=300.0)

        def_max_tasks = max(0, int(db.get_setting("default_user_max_tasks", "2") or 2))
        def_max_epochs = max(100, int(db.get_setting("default_user_max_epochs", "500000") or 500000))

        minted_uid = None
        minted_uname = ""
        with db.transaction() as conn:
            for attempt in range(24):
                cand = security.generate_algorithmic_username(seq_hint=attempt)
                exists = conn.execute("SELECT id FROM users WHERE username = ?", (cand,)).fetchone()
                if not exists:
                    minted_uname = cand
                    break
            if not minted_uname:
                raise RuntimeError("Failed to allocate unique algorithmic username")

            rcode = security.generate_recovery_code(minted_uname)
            rhash = security.hash_recovery_code(rcode)
            cur = conn.execute(
                """
                INSERT INTO users (
                    username, password_hash, role, api_key, is_active, note, created_at,
                    recovery_code, recovery_code_hash, max_tasks, submitted_task_count, max_epochs,
                    perm_lab, perm_api, perm_converter, device_fp_hash, created_ip, last_ip, last_seen_at
                ) VALUES (?, 'temp_algorithmic', 'user', ?, 1, ?, ?, ?, ?, ?, 0, ?, 0, 0, 0, ?, ?, ?, ?)
                """,
                (
                    minted_uname,
                    db.generate_api_key(),
                    f"IP: {ip_scope}" if ip_scope else "",
                    now,
                    rcode,
                    rhash,
                    def_max_tasks,
                    def_max_epochs,
                    fp_key,
                    ip_scope,
                    ip_scope,
                    now,
                ),
            )
            minted_uid = int(cur.lastrowid)
            if fp_key:
                conn.execute(
                    """
                    INSERT INTO identity_bindings (bind_key, user_id, ip_scope, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(bind_key) DO UPDATE SET user_id = excluded.user_id, ip_scope = excluded.ip_scope, updated_at = excluded.updated_at
                    """,
                    (fp_key, minted_uid, ip_scope, now),
                )
            if ip_scope:
                conn.execute(
                    """
                    INSERT INTO identity_bindings (bind_key, user_id, ip_scope, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(bind_key) DO UPDATE SET user_id = excluded.user_id, updated_at = excluded.updated_at
                    """,
                    (f"ip:{ip_scope}", minted_uid, ip_scope, now),
                )
                conn.execute(
                    """
                    INSERT INTO ip_quota_ledger (ip_scope, anon_users_created, anon_tasks_submitted, updated_at)
                    VALUES (?, 1, 0, ?)
                    ON CONFLICT(ip_scope) DO UPDATE SET
                        anon_users_created = ip_quota_ledger.anon_users_created + 1,
                        updated_at = excluded.updated_at
                    """,
                    (ip_scope, now),
                )

    storage.user_root(minted_uname)
    seed_demo_assets_for_user(minted_uid)
    return get_user_by_id(minted_uid, is_admin=is_admin), True


DEMO_CATALOG_VERSION = "20261114_p23"
_DEMO_INSPECT_CACHE = {}

VALID_DATASET_TASK_TYPES = {
    "",
    "cleanup_grain",
    "keying_despill",
    "bokeh_focus",
    "degrain_clean",
    "roto_matte",
}


def normalize_dataset_task_type(raw: str | None) -> str:
    val = str(raw or "").strip()
    return val if val in VALID_DATASET_TASK_TYPES else ""


_DEFAULT_SEED_DEMO = "0"
DEMO_DATASETS = []


def _inspect_demo_cached(s_rel: str, t_rel: str, seq_rel: str) -> dict:
    s_abs = storage.abs_from_root(s_rel)
    seq_abs = storage.abs_from_root(seq_rel) if seq_rel else ""
    try:
        mt_s = int(os.path.getmtime(s_abs))
    except OSError:
        mt_s = 0
    try:
        mt_q = int(os.path.getmtime(seq_abs)) if (seq_abs and os.path.isdir(seq_abs)) else 0
    except OSError:
        mt_q = 0
    key = (s_rel, t_rel, seq_rel, mt_s, mt_q)
    if key not in _DEMO_INSPECT_CACHE:
        eff_seq = seq_rel if (seq_rel and os.path.isdir(seq_abs)) else ""
        _DEMO_INSPECT_CACHE[key] = (eff_seq, storage.inspect_paired_dirs(s_rel, t_rel, eff_seq))
    return {"eff_seq": _DEMO_INSPECT_CACHE[key][0], "stats": _DEMO_INSPECT_CACHE[key][1]}


def seed_demo_assets_for_user(user_id: int, force: bool = False) -> int:
    """Register or refresh bundled demo datasets for a user."""
    if not force and os.environ.get("WILDCAT_SEED_DEMO", _DEFAULT_SEED_DEMO) != "1":
        return 0
    users_prefix = storage.rel_to_root(storage.USERS_STORAGE) + "/"
    now = time.time()
    seeded_count = 0
    for dname, dkey, rel_dir, s_rel, t_rel, seq_rel, strat, d_task_type in DEMO_DATASETS:
        if not (os.path.isdir(storage.abs_from_root(s_rel)) and os.path.isdir(storage.abs_from_root(t_rel))):
            continue
        cached = _inspect_demo_cached(s_rel, t_rel, seq_rel)
        eff_seq = cached["eff_seq"]
        st = cached["stats"]
        if st["frame_count"] <= 0:
            continue

        existing = db.query_one(
            "SELECT * FROM assets WHERE user_id = ? AND name = ?",
            (user_id, dname),
        )
        if existing:
            # Never overwrite a user's own custom-uploaded dataset that happens to share the name
            if str(existing.get("rel_path") or "").startswith(users_prefix):
                continue
            cur_seq = str(existing.get("seq_path") or "")
            if cur_seq.startswith(users_prefix) and not eff_seq:
                keep_seq = cur_seq
                keep_seq_cnt, _ = storage.dir_stats(keep_seq)
            else:
                keep_seq = eff_seq
                keep_seq_cnt = st["seq_frame_count"]
            keep_task_type = normalize_dataset_task_type(existing.get("task_type")) or d_task_type
            db.execute(
                """
                UPDATE assets
                SET kind = 'paired_dataset', rel_path = ?, source_path = ?, target_path = ?,
                    seq_path = ?, frame_count = ?, seq_frame_count = ?, demo_key = ?,
                    width = ?, height = ?, in_channels = ?, out_channels = ?,
                    avg_mae = ?, max_mae = ?, strategy = ?, task_type = ?, size_bytes = ?
                WHERE id = ?
                """,
                (
                    rel_dir,
                    s_rel,
                    t_rel,
                    keep_seq,
                    st["frame_count"],
                    keep_seq_cnt,
                    dkey,
                    st["width"],
                    st["height"],
                    st["in_channels"],
                    st["out_channels"],
                    st["avg_mae"],
                    st["max_mae"],
                    strat,
                    keep_task_type,
                    st["size_bytes"],
                    existing["id"],
                ),
            )
            seeded_count += 1
        else:
            db.execute(
                """
                INSERT OR IGNORE INTO assets (
                    user_id, name, kind, rel_path, source_path, target_path, seq_path,
                    frame_count, seq_frame_count, demo_key, width, height, in_channels, out_channels,
                    avg_mae, max_mae, strategy, task_type, size_bytes, created_at
                ) VALUES (?, ?, 'paired_dataset', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    dname,
                    rel_dir,
                    s_rel,
                    t_rel,
                    eff_seq,
                    st["frame_count"],
                    st["seq_frame_count"],
                    dkey,
                    st["width"],
                    st["height"],
                    st["in_channels"],
                    st["out_channels"],
                    st["avg_mae"],
                    st["max_mae"],
                    strat,
                    d_task_type,
                    st["size_bytes"],
                    now,
                ),
            )
            seeded_count += 1
    return seeded_count


def sync_demo_catalog_for_all_users():
    """Ensure existing active users receive newly added default demo datasets once per catalog revision."""
    if os.environ.get("WILDCAT_SEED_DEMO", _DEFAULT_SEED_DEMO) != "1" or not DEMO_DATASETS:
        return
    avail = [
        d[1]
        for d in DEMO_DATASETS
        if os.path.isdir(storage.abs_from_root(d[3])) and os.path.isdir(storage.abs_from_root(d[4]))
    ]
    if not avail:
        return
    sig = f"{DEMO_CATALOG_VERSION}:{','.join(avail)}"
    if db.get_setting("seeded_demo_catalog_sig", "") == sig:
        return
    users = db.query_all("SELECT id FROM users WHERE is_active = 1")
    for u in users:
        seed_demo_assets_for_user(u["id"], force=True)
    db.set_setting("seeded_demo_catalog_sig", sig)


def create_user_account(
    username: str = "",
    note: str = "",
    seed_demos: bool = True,
    max_tasks: int | None = None,
    max_epochs: int | None = None,
    perm_lab: bool = False,
    perm_api: bool = False,
    perm_converter: bool = False,
) -> dict:
    raw_u = (username or "").strip()
    if not raw_u or raw_u.lower() == "auto":
        uname = ""
        for attempt in range(24):
            cand = security.generate_algorithmic_username(seq_hint=attempt)
            if not db.query_one("SELECT id FROM users WHERE username = ?", (cand,)):
                uname = cand
                break
        if not uname:
            raise RuntimeError("Failed to generate unique username")
    else:
        uname = storage.sanitize_name(raw_u, "")
        if not uname or len(uname) < 2:
            raise ValueError(i18n.t("server.machineNameMinLen"))
        existing = db.query_one("SELECT id FROM users WHERE username = ?", (uname,))
        if existing:
            raise ValueError(i18n.t("server.machineExists", machine=uname))

    eff_max_tasks = (
        max(0, int(max_tasks))
        if max_tasks is not None
        else max(0, int(db.get_setting("default_user_max_tasks", "2") or 2))
    )
    eff_max_epochs = (
        max(100, int(max_epochs))
        if max_epochs is not None
        else max(100, int(db.get_setting("default_user_max_epochs", "500000") or 500000))
    )
    rcode = security.generate_recovery_code(uname)
    rhash = security.hash_recovery_code(rcode)
    now = time.time()

    uid = db.execute(
        """
        INSERT INTO users (
            username, password_hash, role, api_key, is_active, note, created_at,
            recovery_code, recovery_code_hash, max_tasks, submitted_task_count, max_epochs,
            perm_lab, perm_api, perm_converter
        ) VALUES (?, 'admin_created', 'user', ?, 1, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?)
        """,
        (
            uname,
            db.generate_api_key(),
            note or "",
            now,
            rcode,
            rhash,
            eff_max_tasks,
            eff_max_epochs,
            1 if perm_lab else 0,
            1 if perm_api else 0,
            1 if perm_converter else 0,
        ),
    )
    storage.user_root(uname)
    if seed_demos:
        seed_demo_assets_for_user(uid)
    return get_user_by_id(uid)


def admin_update_user(
    user_id: int,
    is_active: bool = None,
    note: str = None,
    max_tasks: int = None,
    reset_used_tasks: bool = False,
    submitted_task_count: int = None,
    max_epochs: int = None,
    perm_lab: bool = None,
    perm_api: bool = None,
    perm_converter: bool = None,
    regenerate_recovery_code: bool = False,
) -> dict:
    u = db.query_one("SELECT * FROM users WHERE id = ?", (user_id,))
    if not u:
        raise ValueError(i18n.t("server.userNotFound"))
    if is_active is not None:
        if u["role"] == "admin" and not is_active:
            raise ValueError(i18n.t("server.cannotDisableAdmin"))
        db.execute("UPDATE users SET is_active = ? WHERE id = ?", (1 if is_active else 0, user_id))
    if note is not None:
        db.execute("UPDATE users SET note = ? WHERE id = ?", (str(note)[:500], user_id))
    if max_tasks is not None:
        db.execute("UPDATE users SET max_tasks = ? WHERE id = ?", (max(0, min(999999, int(max_tasks))), user_id))
    if reset_used_tasks:
        db.execute("UPDATE users SET submitted_task_count = 0 WHERE id = ?", (user_id,))
        for ip_col in (u.get("created_ip"), u.get("last_ip")):
            if ip_col:
                db.execute("UPDATE ip_quota_ledger SET anon_tasks_submitted = 0 WHERE ip_scope = ?", (str(ip_col),))
    elif submitted_task_count is not None:
        db.execute(
            "UPDATE users SET submitted_task_count = ? WHERE id = ?",
            (max(0, min(999999, int(submitted_task_count))), user_id),
        )
    if max_epochs is not None:
        db.execute(
            "UPDATE users SET max_epochs = ? WHERE id = ?",
            (max(100, min(100_000_000, int(max_epochs))), user_id),
        )
    if perm_lab is not None:
        db.execute("UPDATE users SET perm_lab = ? WHERE id = ?", (1 if perm_lab else 0, user_id))
    if perm_api is not None:
        db.execute("UPDATE users SET perm_api = ? WHERE id = ?", (1 if perm_api else 0, user_id))
    if perm_converter is not None:
        db.execute("UPDATE users SET perm_converter = ? WHERE id = ?", (1 if perm_converter else 0, user_id))
    if regenerate_recovery_code:
        rcode = security.generate_recovery_code(u["username"])
        rhash = security.hash_recovery_code(rcode)
        db.execute(
            "UPDATE users SET recovery_code = ?, recovery_code_hash = ? WHERE id = ?",
            (rcode, rhash, user_id),
        )
    return get_user_by_id(user_id)


def apply_default_quotas_to_temp_users(max_tasks: int, max_epochs: int) -> int:
    """Sync updated global default `max_tasks` and `max_epochs` to existing unprivileged temporary users."""
    mt = max(0, min(999999, int(max_tasks)))
    me = max(100, min(100_000_000, int(max_epochs)))
    with db.transaction() as conn:
        cur = conn.execute(
            """
            UPDATE users
            SET max_tasks = ?, max_epochs = ?
            WHERE role != 'admin' AND username != 'DLI-DESKTOP' AND perm_lab = 0 AND perm_api = 0
            """,
            (mt, me),
        )
        return int(cur.rowcount or 0)


def regenerate_user_api_key(user_id: int) -> str:
    new_key = db.generate_api_key()
    db.execute("UPDATE users SET api_key = ? WHERE id = ?", (new_key, user_id))
    return new_key


def admin_delete_user(user_id: int):
    u = db.query_one("SELECT id, username, role FROM users WHERE id = ?", (user_id,))
    if not u:
        return
    if u["role"] == "admin":
        raise ValueError(i18n.t("server.cannotDeleteAdmin"))
    # Stop & delete all tasks belonging to this user
    user_tasks = db.query_all("SELECT id FROM tasks WHERE user_id = ?", (user_id,))
    for t in user_tasks:
        scheduler.SCHEDULER.delete_task(t["id"])
    # Delete user's uploaded assets directory
    user_assets = db.query_all("SELECT rel_path, seq_path FROM assets WHERE user_id = ?", (user_id,))
    for a in user_assets:
        storage.delete_asset_files(a["rel_path"])
        if a.get("seq_path"):
            storage.delete_asset_files(a["seq_path"])
    db.delete_user_experiment_data(user_id)
    db.execute("DELETE FROM identity_bindings WHERE user_id = ?", (user_id,))
    db.execute("DELETE FROM users WHERE id = ?", (user_id,))


def list_all_users_with_stats() -> list:
    rows = db.query_all(f"SELECT {USER_SELECT_COLS} FROM users ORDER BY id ASC")
    out = []
    for r in rows:
        u = _decorate_user(r, is_admin=False)
        uid = u["id"]
        u["asset_count"] = (
            db.query_one("SELECT COUNT(*) AS c FROM assets WHERE user_id = ?", (uid,)) or {}
        ).get("c", 0)
        u["task_count"] = (
            db.query_one("SELECT COUNT(*) AS c FROM tasks WHERE user_id = ?", (uid,)) or {}
        ).get("c", 0)
        out.append(u)
    return out


# ─────────────────────── Asset Management ───────────────────────


def _enrich_asset_row(r: dict) -> dict:
    users_prefix = storage.rel_to_root(storage.USERS_STORAGE) + "/"
    is_eval_only = r.get("kind") == "eval_sequence"
    sp = r.get("source_path") or r.get("rel_path") or ""
    tp = "" if is_eval_only else (r.get("target_path") or r.get("rel_path") or "")
    seqp = sp if is_eval_only else (r.get("seq_path") or "")

    s_files = storage.expand_sequence_files(sp)
    t_files = [] if is_eval_only else storage.expand_sequence_files(tp)
    seq_files = storage.expand_sequence_files(seqp) if seqp else []

    r["frame_count"] = min(len(s_files), len(t_files)) if (s_files and t_files) else len(s_files)
    r["seq_path"] = seqp
    r["seq_frame_count"] = len(seq_files)
    r["has_eval_sequence"] = len(seq_files) >= 2
    r["demo_key"] = r.get("demo_key") or ""
    r["task_type"] = normalize_dataset_task_type(r.get("task_type"))
    r["is_builtin_demo"] = not str(r.get("rel_path") or "").startswith(users_prefix)
    r["has_custom_eval_sequence"] = bool(seqp and str(seqp).startswith(users_prefix))
    r["can_remove_eval_sequence"] = bool(seq_files) and not is_eval_only

    ext = os.path.splitext(s_files[0])[1].lower().lstrip(".") if s_files else "exr"
    r["file_format"] = ext.upper()
    r["colorspace"] = "ACEScg" if ext == "exr" else "sRGB"
    r["source_thumb_url"] = telemetry.to_jpg_url(s_files[0]) if s_files else None
    r["target_thumb_url"] = telemetry.to_jpg_url(t_files[0]) if t_files else None
    r["seq_thumb_url"] = telemetry.to_jpg_url(seq_files[0]) if seq_files else None
    r["thumb_url"] = r["target_thumb_url"] or r["source_thumb_url"]
    return r


def list_assets(user_id: int, all_users: bool = False) -> list:
    if all_users:
        rows = db.query_all(
            """
            SELECT a.*, u.username
            FROM assets a
            JOIN users u ON a.user_id = u.id
            ORDER BY a.id ASC
            """
        )
    else:
        rows = db.query_all(
            """
            SELECT a.*, u.username
            FROM assets a
            JOIN users u ON a.user_id = u.id
            WHERE a.user_id = ?
            ORDER BY CASE WHEN a.demo_key != '' THEN 1 ELSE 0 END ASC, a.created_at DESC, a.id ASC
            """,
            (user_id,),
        )
    for r in rows:
        _enrich_asset_row(r)
    return rows


def get_asset_frames(user_id: int, is_admin: bool, asset_id: int) -> dict:
    row = db.query_one("SELECT * FROM assets WHERE id = ?", (asset_id,))
    if not row:
        raise ValueError(i18n.t("server.assetNotFound"))
    if not is_admin and row["user_id"] != user_id:
        raise PermissionError(i18n.t("server.assetAccessForbidden"))

    is_eval_only = row.get("kind") == "eval_sequence"
    sp = row.get("source_path") or row.get("rel_path") or ""
    tp = "" if is_eval_only else (row.get("target_path") or row.get("rel_path") or "")
    seqp = sp if is_eval_only else (row.get("seq_path") or "")

    s_files = [] if is_eval_only else storage.expand_sequence_files(sp)
    t_files = [] if is_eval_only else storage.expand_sequence_files(tp)
    seq_files = storage.expand_sequence_files(seqp) if seqp else []

    n_pair = min(len(s_files), len(t_files))
    s_urls = telemetry.batch_to_jpg_urls(s_files[:n_pair], quality=88)
    t_urls = telemetry.batch_to_jpg_urls(t_files[:n_pair], quality=88)
    q_urls = telemetry.batch_to_jpg_urls(seq_files, quality=88)

    paired_frames = [
        {
            "idx": idx,
            "name": os.path.basename(s_files[idx]),
            "target_name": os.path.basename(t_files[idx]),
            "src_url": s_urls[idx],
            "tgt_url": t_urls[idx],
        }
        for idx in range(n_pair)
    ]
    seq_frames = [
        {
            "idx": idx,
            "name": os.path.basename(seq_files[idx]),
            "url": q_urls[idx],
        }
        for idx in range(len(seq_files))
    ]
    return {
        "asset_id": asset_id,
        "name": row["name"],
        "frame_count": n_pair,
        "seq_frame_count": len(seq_frames),
        "has_eval_sequence": len(seq_frames) >= 2,
        "paired_frames": paired_frames,
        "seq_frames": seq_frames,
    }


def _enforce_temp_user_upload_quota(
    user_id: int,
    username: str,
    is_admin: bool = False,
    replacing_name: str = "",
) -> None:
    if is_admin or not is_temp_restrictions_enabled():
        return
    u_row = db.query_one("SELECT role, max_tasks FROM users WHERE id = ?", (int(user_id),))
    if not u_row or u_row.get("role") == "admin":
        return
    max_t = int(u_row.get("max_tasks") if u_row.get("max_tasks") is not None else 2)
    max_custom_assets = max(4, max_t * 2)
    cnt_row = db.query_one(
        "SELECT COUNT(*) AS c FROM assets WHERE user_id = ? AND demo_key = '' AND name != ?",
        (int(user_id), str(replacing_name or "")),
    )
    custom_cnt = int((cnt_row or {}).get("c") or 0)
    if custom_cnt >= max_custom_assets:
        raise PermissionError(i18n.t("server.tempUserUploadQuotaExceeded", max_assets=max_custom_assets))
    user_rel = storage.rel_to_root(storage.user_root(username))
    if storage.dir_total_bytes(user_rel) >= TEMP_USER_MAX_STORAGE_BYTES:
        raise PermissionError(i18n.t("server.tempUserUploadQuotaExceeded", max_assets=max_custom_assets))


def create_paired_dataset(
    user_id: int,
    username: str,
    dataset_name: str,
    source_tuples: list,
    target_tuples: list,
    max_error_pct: float = 35.0,
    strategy: str = "standard",
    enable_check: bool = True,
    input_colorspace: str = "auto",
    working_colorspace: str = "ACEScg",
    seq_tuples: list | None = None,
    task_type: str = "",
    is_admin: bool = False,
) -> dict:
    clean_name = storage.sanitize_name(dataset_name, "dataset")
    clean_task_type = normalize_dataset_task_type(task_type)
    _enforce_temp_user_upload_quota(user_id, username, is_admin=is_admin, replacing_name=clean_name)
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
        seq_tuples=seq_tuples,
    )
    now = time.time()
    existing = db.query_one(
        "SELECT id, seq_path FROM assets WHERE user_id = ? AND name = ?", (user_id, clean_name)
    )
    if existing:
        aid = existing["id"]
        if existing.get("seq_path") and existing["seq_path"] != info["seq_path"]:
            storage.delete_asset_files(existing["seq_path"])
        db.execute(
            """
            UPDATE assets
            SET kind = 'paired_dataset', rel_path = ?, source_path = ?, target_path = ?,
                seq_path = ?, frame_count = ?, seq_frame_count = ?, demo_key = '',
                width = ?, height = ?, in_channels = ?, out_channels = ?,
                avg_mae = ?, max_mae = ?, strategy = ?, task_type = ?, size_bytes = ?, created_at = ?
            WHERE id = ?
            """,
            (
                info["rel_path"],
                info["source_path"],
                info["target_path"],
                info["seq_path"],
                info["frame_count"],
                info["seq_frame_count"],
                info["width"],
                info["height"],
                info["in_channels"],
                info["out_channels"],
                info["avg_mae"],
                info["max_mae"],
                info["strategy"],
                clean_task_type,
                info["size_bytes"],
                now,
                aid,
            ),
        )
    else:
        aid = db.execute(
            """
            INSERT INTO assets (
                user_id, name, kind, rel_path, source_path, target_path, seq_path,
                frame_count, seq_frame_count, demo_key, width, height, in_channels, out_channels,
                avg_mae, max_mae, strategy, task_type, size_bytes, created_at
            ) VALUES (?, ?, 'paired_dataset', ?, ?, ?, ?, ?, ?, '', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                clean_name,
                info["rel_path"],
                info["source_path"],
                info["target_path"],
                info["seq_path"],
                info["frame_count"],
                info["seq_frame_count"],
                info["width"],
                info["height"],
                info["in_channels"],
                info["out_channels"],
                info["avg_mae"],
                info["max_mae"],
                info["strategy"],
                clean_task_type,
                info["size_bytes"],
                now,
            ),
        )
    row = db.query_one("SELECT * FROM assets WHERE id = ?", (aid,))
    return _enrich_asset_row(row)


def update_asset_task_type(user_id: int, is_admin: bool, asset_id: int, task_type: str) -> dict:
    row = db.query_one("SELECT * FROM assets WHERE id = ?", (int(asset_id),))
    if not row:
        raise ValueError(i18n.t("server.assetNotFound"))
    if not is_admin and int(row["user_id"]) != int(user_id):
        raise PermissionError(i18n.t("server.assetAccessForbidden"))
    clean_task_type = normalize_dataset_task_type(task_type)
    db.execute("UPDATE assets SET task_type = ? WHERE id = ?", (clean_task_type, int(asset_id)))
    updated = db.query_one(
        """
        SELECT a.*, u.username
        FROM assets a
        JOIN users u ON a.user_id = u.id
        WHERE a.id = ?
        """,
        (int(asset_id),),
    )
    return _enrich_asset_row(updated)


def attach_asset_eval_sequence(
    user_id: int,
    is_admin: bool,
    username: str,
    asset_id: int,
    seq_tuples: list,
    input_colorspace: str = "auto",
    working_colorspace: str = "ACEScg",
) -> dict:
    row = db.query_one("SELECT * FROM assets WHERE id = ?", (asset_id,))
    if not row:
        raise ValueError(i18n.t("server.assetNotFound"))
    if not is_admin and row["user_id"] != user_id:
        raise PermissionError(i18n.t("server.assetAccessForbidden"))

    owner = get_user_by_id(row["user_id"])
    owner_name = (owner or {}).get("username") or username
    _enforce_temp_user_upload_quota(row["user_id"], owner_name, is_admin=is_admin, replacing_name=row["name"])
    seq_info = storage.attach_eval_sequence_to_dataset(
        username=owner_name,
        dataset_name=row["name"],
        ds_rel=row["rel_path"],
        old_seq_rel=row.get("seq_path") or "",
        expected_w=int(row.get("width") or 0),
        expected_h=int(row.get("height") or 0),
        expected_in_ch=int(row.get("in_channels") or 3),
        seq_tuples=seq_tuples,
        input_colorspace=input_colorspace,
        working_colorspace=working_colorspace,
    )
    _, sb = storage.dir_stats(row["source_path"])
    _, tb = storage.dir_stats(row["target_path"])
    new_bytes = sb + tb + int(seq_info["seq_bytes"])
    db.execute(
        "UPDATE assets SET seq_path = ?, seq_frame_count = ?, size_bytes = ? WHERE id = ?",
        (seq_info["seq_path"], seq_info["seq_frame_count"], new_bytes, asset_id),
    )
    updated = db.query_one("SELECT * FROM assets WHERE id = ?", (asset_id,))
    return _enrich_asset_row(updated)


def remove_asset_eval_sequence(user_id: int, is_admin: bool, asset_id: int) -> dict:
    row = db.query_one("SELECT * FROM assets WHERE id = ?", (asset_id,))
    if not row:
        raise ValueError(i18n.t("server.assetNotFound"))
    if not is_admin and row["user_id"] != user_id:
        raise PermissionError(i18n.t("server.assetAccessForbidden"))
    if row.get("seq_path"):
        storage.delete_asset_files(row["seq_path"])
    _, sb = storage.dir_stats(row["source_path"])
    _, tb = storage.dir_stats(row["target_path"])
    db.execute(
        "UPDATE assets SET seq_path = '', seq_frame_count = 0, size_bytes = ? WHERE id = ?",
        (sb + tb, asset_id),
    )
    updated = db.query_one("SELECT * FROM assets WHERE id = ?", (asset_id,))
    return _enrich_asset_row(updated)


def delete_asset(user_id: int, is_admin: bool, asset_id: int):
    row = db.query_one("SELECT * FROM assets WHERE id = ?", (asset_id,))
    if not row:
        raise ValueError(i18n.t("server.assetNotFound"))
    if not is_admin and row["user_id"] != user_id:
        raise PermissionError(i18n.t("server.assetDeleteForbidden"))
    storage.delete_asset_files(row["rel_path"])
    if row.get("seq_path"):
        storage.delete_asset_files(row["seq_path"])
    db.execute("DELETE FROM assets WHERE id = ?", (asset_id,))


# ─────────────────────── Task & Queue Management ───────────────────────

_TASK_DISK_STAT_CACHE: dict[str, tuple[tuple[int, int, str], float, dict]] = {}


def _inspect_task_run_dir_cached(run_dir_rel: str, status: str, now: float) -> dict:
    rdir_abs = storage.abs_from_root(run_dir_rel)
    try:
        d_mt = int(os.path.getmtime(rdir_abs) * 1000)
    except OSError:
        d_mt = 0
    ckpt_dir = os.path.join(rdir_abs, "ckpt")
    try:
        c_mt = int(os.path.getmtime(ckpt_dir) * 1000)
    except OSError:
        c_mt = 0

    sig = (d_mt, c_mt, str(status or ""))
    cached = _TASK_DISK_STAT_CACHE.get(run_dir_rel)
    is_active = status in ("running", "queued")
    if cached is not None:
        prev_sig, prev_ts, prev_val = cached
        if prev_sig == sig and (not is_active or (now - prev_ts) < 5.0):
            return prev_val

    has_nuke_models = bool(glob.glob(os.path.join(rdir_abs, "Training_*.[0-9]*.pt")))
    info = {}
    info["has_ckpt"] = os.path.isfile(os.path.join(ckpt_dir, "latest.pt")) or has_nuke_models
    info["size_mb"] = round(storage.dir_total_bytes(run_dir_rel) / (1024 * 1024), 1)
    _TASK_DISK_STAT_CACHE[run_dir_rel] = (sig, now, info)
    return info


def list_tasks(user_id: int, all_users: bool = False) -> list:
    analytics_db.sync_pending_tasks_to_db(user_id=None if all_users else user_id)
    qstat = scheduler.SCHEDULER.queue_status()

    if all_users:
        rows = db.query_all("SELECT * FROM tasks ORDER BY id DESC")
    else:
        rows = db.query_all("SELECT * FROM tasks WHERE user_id = ? ORDER BY id DESC", (user_id,))

    now = time.time()
    for r in rows:
        scheduler.SCHEDULER.enrich_task_with_queue(r, qstat=qstat)
        raw_cfg = r.pop("config_json", None)
        try:
            r["config"] = json.loads(raw_cfg or "{}")
        except Exception:
            r["config"] = {}
        r.update(_inspect_task_run_dir_cached(r["run_dir"], r.get("status") or "", now))
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
        raise ValueError(i18n.t("server.taskNotFound"))
    if not is_admin and row["user_id"] != user_id:
        raise PermissionError(i18n.t("server.taskDeleteForbidden"))
    scheduler.SCHEDULER.delete_task(task_id)


def stop_user_task(user_id: int, is_admin: bool, task_id: int) -> dict:
    row = db.query_one("SELECT * FROM tasks WHERE id = ?", (task_id,))
    if not row:
        raise ValueError(i18n.t("server.taskNotFound"))
    if not is_admin and row["user_id"] != user_id:
        raise PermissionError(i18n.t("server.taskStopForbidden"))
    return scheduler.SCHEDULER.stop_task(task_id)


def export_task_cat(user_id: int, is_admin: bool, task_id: int, step: int = None) -> dict:
    row = get_task_for_user(user_id, is_admin, task_id=task_id)
    if not row:
        raise ValueError(i18n.t("server.taskNotFound"))
    rdir_abs = storage.abs_from_root(row["run_dir"])
    pfx = telemetry.resolve_nuke_prefix(rdir_abs)
    inv = telemetry.list_task_models(row["run_dir"], row.get("config"))
    target_step = int(step) if step is not None else (inv["items"][0]["step"] if inv["items"] else int(row.get("last_step") or 0))
    nuke_pt = os.path.join(rdir_abs, f"{pfx}.{target_step}.pt") if target_step > 0 else os.path.join(rdir_abs, f"{row['node_name']}.pt")
    cfg_abs_arg = storage.abs_from_root(row["config_path"])
    cmd = [scheduler.PYTHON_BIN, "export_cat.py", cfg_abs_arg]
    if step is not None:
        st_int = int(step)
        step_ckpt = os.path.join(rdir_abs, "ckpt", f"step_{st_int:06d}.pt")
        lat_ckpt = os.path.join(rdir_abs, "ckpt", "latest.pt")
        if os.path.isfile(step_ckpt):
            cmd.append(str(st_int))
        elif os.path.isfile(lat_ckpt) and int(telemetry.inspect_ckpt_cached(lat_ckpt).get("step") or 0) == st_int:
            pass
        else:
            raise FileNotFoundError(i18n.t("server.ckptStepSourceMissing", step=st_int))
    cmd.extend(["--out-pt", nuke_pt])
    exp_env = storage.apply_subprocess_storage_env(os.environ.copy())
    res = subprocess.run(
        cmd,
        cwd=db.ROOT,
        env=exp_env,
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


def recommend_task_models(user_id: int, is_admin: bool, task_id: int, force: bool = False) -> dict:
    """Evaluate checkpoints of a completed/stopped task on its validation sequence and return updated models inventory."""
    row = get_task_for_user(user_id, is_admin, task_id=task_id)
    if not row:
        raise ValueError(i18n.t("server.taskNotFound"))
    if row.get("status") == "running":
        raise RuntimeError(i18n.t("server.taskRecommendBusy"))

    cfg = dict(row.get("config") or {})
    seq_path = telemetry._resolve_cfg_eval_seq(cfg)
    if not seq_path:
        raise ValueError(i18n.t("server.taskRecommendNoSeq"))

    rdir_abs = storage.abs_from_root(row["run_dir"])
    cfg_abs = storage.abs_from_root(row["config_path"])
    cmd = [scheduler.PYTHON_BIN, "-m", "wildcat.recommender", rdir_abs, cfg_abs, "--eval-seq", seq_path]
    if force:
        cmd.append("--force")

    exp_env = storage.apply_subprocess_storage_env(os.environ.copy())
    res = subprocess.run(
        cmd,
        cwd=db.ROOT,
        env=exp_env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=240,
    )
    if res.returncode != 0:
        raise RuntimeError(((res.stderr or "") + "\n" + (res.stdout or "")).strip() or "Model recommendation failed")

    parsed = telemetry.parse_train_log_full_cached(os.path.join(rdir_abs, "train.log"))
    return telemetry.list_task_models(row["run_dir"], cfg, parsed_log=parsed)


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
        raise ValueError(i18n.t("server.taskNotFound"))
    rdir_abs = storage.abs_from_root(row["run_dir"])
    inv = telemetry.list_task_models(row["run_dir"], row.get("config"))
    if not inv["items"]:
        raise ValueError(i18n.t("server.noCheckpointsYet"))

    if step is None:
        item = inv["items"][0]
    else:
        st_req = int(step)
        item = next((x for x in inv["items"] if x["step"] == st_req), None)
        if not item:
            raise ValueError(i18n.t("server.stepModelNotFound", step=st_req))

    st = int(item["step"])
    target_fn = item["pt_filename"]
    target_abs = os.path.join(rdir_abs, target_fn)
    if os.path.isfile(target_abs) and os.path.getsize(target_abs) > 0:
        return target_abs, target_fn

    export_task_cat(user_id, is_admin, task_id, step=st)
    if not os.path.isfile(target_abs) or os.path.getsize(target_abs) == 0:
        raise FileNotFoundError(i18n.t("server.exportedFileMissing", filename=target_fn))
    return target_abs, target_fn


def build_all_models_zip(user_id: int, is_admin: bool, task_id: int, ext: str) -> tuple[str, str]:
    """Bundle all checkpoint step models of this task (strictly Nuke-named) into a fast ZIP_STORED archive."""
    row = get_task_for_user(user_id, is_admin, task_id=task_id)
    if not row:
        raise ValueError(i18n.t("server.taskNotFound"))
    rdir_abs = storage.abs_from_root(row["run_dir"])
    inv = telemetry.list_task_models(row["run_dir"], row.get("config"))
    if not inv["items"]:
        raise ValueError(i18n.t("server.noDownloadableModels"))

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
