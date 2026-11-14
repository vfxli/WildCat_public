# -*- coding: utf-8 -*-
"""Security, Session Signing, Algorithmic Identity, Cross-Device Roaming Codes & Rate Limiting."""
import collections
import hashlib
import hmac
import ipaddress
import re
import secrets
import threading
import time
from urllib.parse import urlparse

from fastapi import HTTPException, Request, Response

from . import db, i18n

# Unambiguous Crockford Base32 alphabet (excludes I, L, O, U to avoid human visual confusion)
CROCKFORD_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
SESSION_COOKIE_NAME = "wc_uid_sig"
SESSION_COOKIE_MAX_AGE = 365 * 24 * 3600  # 1 year

# Request & Upload Safety Caps
MAX_REQUEST_BODY_BYTES = 512 * 1024 * 1024  # 512 MB max HTTP request body
MAX_UPLOAD_FILES_TOTAL = 300
MAX_SINGLE_UPLOAD_BYTES = 128 * 1024 * 1024  # 128 MB max per single file
ALLOWED_UPLOAD_EXTS = {".exr", ".png", ".tif", ".tiff", ".jpg", ".jpeg", ".zip"}


def random_crockford(length: int) -> str:
    return "".join(secrets.choice(CROCKFORD_ALPHABET) for _ in range(max(1, int(length))))


def generate_algorithmic_username(seq_hint: int = 0) -> str:
    """Mint a non-editable, collision-resistant short username like `U-8K3M9P`."""
    # Mix sequence/time low bits with 25 bits of CSPRNG entropy into 6 Crockford chars (30 bits = 1.07B namespace)
    mix = ((int(time.time() * 1000) ^ (int(seq_hint) * 2654435761)) & 0x1F) << 25
    val = mix | secrets.randbits(25)
    chars = []
    for _ in range(6):
        chars.append(CROCKFORD_ALPHABET[val & 0x1F])
        val >>= 5
    return "U-" + "".join(chars)


def generate_recovery_code(username: str) -> str:
    """Mint a passwordless cross-device roaming code bound to `username`, e.g. `U-8K3M9P-4R9W-7T2Q`."""
    clean_u = str(username or "USER").strip().upper()
    return f"{clean_u}-{random_crockford(4)}-{random_crockford(4)}"


def normalize_recovery_code(raw_code: str) -> str:
    """Normalize user-pasted roaming code (trim spaces, uppercase, map ambiguous O->0 / I,L->1 in suffix)."""
    s = re.sub(r"[\s]+", "", str(raw_code or "")).upper()
    if not s or len(s) > 80:
        return ""
    if "-" not in s and len(s) == 15 and s.startswith("U") and s.isalnum():
        body = s[1:].replace("O", "0").replace("I", "1").replace("L", "1")
        return f"U-{body[:6]}-{body[6:10]}-{body[10:14]}"
    parts = s.split("-")
    if len(parts) >= 3:
        prefix = "-".join(parts[:-2])
        if prefix.startswith("U-") and len(prefix) == 8:
            prefix = "U-" + prefix[2:].replace("O", "0").replace("I", "1").replace("L", "1")
        suf1 = parts[-2].replace("O", "0").replace("I", "1").replace("L", "1")
        suf2 = parts[-1].replace("O", "0").replace("I", "1").replace("L", "1")
        return f"{prefix}-{suf1}-{suf2}"
    return s


def hash_recovery_code(raw_code: str) -> str:
    norm = normalize_recovery_code(raw_code)
    if not norm:
        return ""
    return hashlib.sha256(("wc_roam_v1|" + norm).encode("utf-8")).hexdigest()


def _get_session_secret() -> bytes:
    sec = db.get_setting("session_hmac_secret", "")
    if not sec or len(sec) < 32:
        sec = secrets.token_hex(32)
        db.set_setting("session_hmac_secret", sec)
    return sec.encode("utf-8")


def sign_session_token(user_id: int) -> str:
    ts = int(time.time())
    payload = f"{int(user_id)}.{ts}"
    sig = hmac.new(_get_session_secret(), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{payload}.{sig}"


def verify_session_token(token: str) -> int | None:
    if not token or len(token) > 160:
        return None
    parts = str(token).strip().split(".")
    if len(parts) != 3:
        return None
    uid_s, ts_s, sig_s = parts
    if not (uid_s.isdigit() and ts_s.isdigit() and len(sig_s) == 64):
        return None
    payload = f"{uid_s}.{ts_s}"
    expected = hmac.new(_get_session_secret(), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    if not secrets.compare_digest(sig_s.lower(), expected.lower()):
        return None
    return int(uid_s)


def set_session_cookie(response: Response, request: Request, user_id: int) -> str:
    tok = sign_session_token(user_id)
    proto = (request.headers.get("x-forwarded-proto") or request.url.scheme or "http").lower()
    is_https = proto == "https"
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=tok,
        max_age=SESSION_COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
        secure=is_https,
        path="/",
    )
    return tok


def _is_trusted_proxy_peer(peer_ip: str) -> bool:
    if not peer_ip or peer_ip in ("testclient", "localhost"):
        return True
    try:
        ip_obj = ipaddress.ip_address(peer_ip)
        return bool(ip_obj.is_loopback or ip_obj.is_private)
    except ValueError:
        return False


def extract_client_ip(request: Request) -> str:
    """Extract real client IP safely; scan X-Forwarded-For right-to-left when direct peer is a trusted local/private proxy."""
    peer = (request.client.host if request.client else "") or ""
    if _is_trusted_proxy_peer(peer):
        xff = (request.headers.get("x-forwarded-for") or "").strip()
        if xff:
            # Walk right-to-left so client-forged left-side X-Forwarded-For values can never spoof IP identity or bypass lockouts
            for raw_hop in reversed(xff.split(",")):
                hop = raw_hop.strip()
                if not hop:
                    continue
                try:
                    ip_obj = ipaddress.ip_address(hop)
                except ValueError:
                    continue
                if hop == peer or ip_obj.is_loopback:
                    continue
                return str(ip_obj)
            # If all hops were loopback/peer, accept rightmost valid hop
            for raw_hop in reversed(xff.split(",")):
                hop = raw_hop.strip()
                try:
                    return str(ipaddress.ip_address(hop))
                except ValueError:
                    continue
        xri = (request.headers.get("x-real-ip") or "").strip()
        if xri:
            try:
                return str(ipaddress.ip_address(xri))
            except ValueError:
                pass
    return peer or "unknown"


def normalize_ip_scope(client_ip: str) -> str:
    """Normalize IPv4 to exact address and IPv6 to /64 network prefix (defeating IPv6 privacy address rotation)."""
    raw = (client_ip or "").strip()
    if not raw or raw in ("testclient", "localhost", "unknown"):
        return raw or "local"
    try:
        ip_obj = ipaddress.ip_address(raw)
        if ip_obj.version == 6:
            net = ipaddress.IPv6Network((ip_obj, 64), strict=False)
            return str(net.network_address) + "/64"
        return str(ip_obj)
    except ValueError:
        return raw[:64]


def compute_device_fp_hash(client_ip: str, raw_fp: str) -> tuple[str, str]:
    """Return `(ip_scope, fp_binding_key)` from normalized IP scope + sanitized hardware traits."""
    ip_scope = normalize_ip_scope(client_ip)
    clean_fp = re.sub(r"[^\x20-\x7E]", "", str(raw_fp or "")).strip()[:256]
    if clean_fp and len(clean_fp) >= 8:
        fp_key = "fp:" + hashlib.sha256(f"{ip_scope}|{clean_fp}".encode("utf-8")).hexdigest()[:40]
    else:
        fp_key = ""
    return ip_scope, fp_key


def verify_same_origin_csrf(request: Request) -> None:
    """Reject cross-site state-changing browser requests whose Origin/Referer mismatches Host."""
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return
    # Programmatic API key requests are stateless and non-ambient
    if request.headers.get("x-api-key") or request.headers.get("authorization"):
        return
    host = (request.headers.get("x-forwarded-host") or request.headers.get("host") or "").split(",")[0].strip().lower()
    if not host:
        return
    origin = (request.headers.get("origin") or "").strip()
    if origin and origin != "null":
        parsed = urlparse(origin)
        if parsed.netloc.lower() != host:
            raise HTTPException(status_code=403, detail=i18n.t("server.csrfForbidden"))
        return
    referer = (request.headers.get("referer") or "").strip()
    if referer:
        parsed_ref = urlparse(referer)
        if parsed_ref.netloc and parsed_ref.netloc.lower() != host:
            raise HTTPException(status_code=403, detail=i18n.t("server.csrfForbidden"))


class SlidingWindowRateLimiter:
    """Thread-safe, bounded-memory sliding-window rate limiter for brute-force & flood defense."""

    def __init__(self, max_buckets: int = 8192):
        self._lock = threading.Lock()
        self._buckets: dict[str, collections.deque] = {}
        self._max_buckets = max_buckets
        self._ops = 0

    def check(self, key: str, limit: int, window_sec: float, consume: bool = True) -> tuple[bool, int]:
        """Return `(allowed, retry_after_sec)`."""
        now = time.monotonic()
        cutoff = now - window_sec
        with self._lock:
            self._ops += 1
            if self._ops % 512 == 0 or len(self._buckets) > self._max_buckets:
                self._prune(now, window_sec)
            dq = self._buckets.get(key)
            if dq is None:
                dq = collections.deque()
                self._buckets[key] = dq
            while dq and dq[0] <= cutoff:
                dq.popleft()
            if len(dq) >= limit:
                retry_after = max(1, int(window_sec - (now - dq[0])) + 1)
                return False, retry_after
            if consume:
                dq.append(now)
            return True, 0

    def record_failure(self, key: str, window_sec: float) -> None:
        now = time.monotonic()
        cutoff = now - window_sec
        with self._lock:
            dq = self._buckets.setdefault(key, collections.deque())
            while dq and dq[0] <= cutoff:
                dq.popleft()
            dq.append(now)

    def clear(self, key: str) -> None:
        with self._lock:
            self._buckets.pop(key, None)

    def _prune(self, now: float, default_window: float) -> None:
        stale_keys = [
            k for k, dq in self._buckets.items() if not dq or (now - dq[-1]) > max(default_window, 900.0)
        ]
        for k in stale_keys:
            self._buckets.pop(k, None)
        while len(self._buckets) > self._max_buckets:
            oldest_key = next(iter(self._buckets))
            self._buckets.pop(oldest_key, None)


RATE_LIMITER = SlidingWindowRateLimiter()


def enforce_rate_limit(bucket_prefix: str, client_ip: str, limit: int, window_sec: float) -> None:
    ip_scope = normalize_ip_scope(client_ip)
    allowed, retry_sec = RATE_LIMITER.check(f"{bucket_prefix}:{ip_scope}", limit=limit, window_sec=window_sec)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=i18n.t("server.rateLimited", seconds=retry_sec),
            headers={"Retry-After": str(retry_sec)},
        )


# Progressive exponential lockout tiers: (fail_count_threshold, lock_duration_seconds)
ADMIN_LOGIN_TIERS = ((5, 900.0), (8, 3600.0), (12, 86400.0))
ADMIN_GLOBAL_TIERS = ((20, 300.0),)
RECOVER_TIERS = ((8, 900.0), (15, 7200.0), (25, 86400.0))
ADMIN_TOKEN_TIERS = ((12, 900.0), (24, 3600.0))
API_KEY_TIERS = ((15, 900.0), (30, 3600.0))


def check_persistent_lockout(bucket_prefix: str, client_ip: str = "", is_global: bool = False) -> str:
    """Verify persistent SQLite lockout state before processing sensitive authentication."""
    scope_key = f"{bucket_prefix}:__all__" if is_global else f"{bucket_prefix}:{normalize_ip_scope(client_ip)}"
    now = time.time()
    row = db.query_one("SELECT locked_until FROM auth_lockouts WHERE scope_key = ?", (scope_key,))
    if row:
        locked_until = float(row.get("locked_until") or 0.0)
        if locked_until > now:
            retry_sec = max(1, int(locked_until - now) + 1)
            msg_key = "server.adminGlobalCooldown" if is_global else "server.bruteForceLocked"
            raise HTTPException(
                status_code=429,
                detail=i18n.t(msg_key, seconds=retry_sec),
                headers={"Retry-After": str(retry_sec)},
            )
    return scope_key


def record_persistent_failure(
    scope_key: str,
    tiers: tuple[tuple[int, float], ...] = ADMIN_LOGIN_TIERS,
    reset_window_sec: float = 3600.0,
) -> tuple[int, int]:
    """Atomically increment persistent failure count and apply progressive exponential lockout when hitting a tier."""
    now = time.time()
    new_count = 1
    retry_sec = 0
    with db.transaction() as conn:
        row = conn.execute(
            "SELECT fail_count, first_fail_at, last_fail_at, locked_until FROM auth_lockouts WHERE scope_key = ?",
            (scope_key,),
        ).fetchone()
        if row:
            last_fail = float(row["last_fail_at"] or 0.0)
            prev_lock = float(row["locked_until"] or 0.0)
            carry_horizon = max(last_fail, prev_lock) + float(reset_window_sec)
            if now <= carry_horizon:
                new_count = int(row["fail_count"] or 0) + 1
                first_ts = float(row["first_fail_at"] or now)
            else:
                new_count = 1
                first_ts = now
        else:
            first_ts = now

        lock_dur = 0.0
        if tiers:
            max_thresh, max_dur = tiers[-1]
            if new_count >= int(max_thresh):
                lock_dur = float(max_dur)
            else:
                for thresh, dur in tiers[:-1]:
                    if new_count == int(thresh):
                        lock_dur = float(dur)
                        break

        locked_until = (now + lock_dur) if lock_dur > 0 else 0.0
        if lock_dur > 0:
            retry_sec = max(1, int(lock_dur))

        conn.execute(
            """
            INSERT INTO auth_lockouts (scope_key, fail_count, first_fail_at, last_fail_at, locked_until)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(scope_key) DO UPDATE SET
                fail_count = excluded.fail_count,
                first_fail_at = excluded.first_fail_at,
                last_fail_at = excluded.last_fail_at,
                locked_until = CASE
                    WHEN excluded.locked_until > auth_lockouts.locked_until THEN excluded.locked_until
                    ELSE auth_lockouts.locked_until
                END
            """,
            (scope_key, new_count, first_ts, now, locked_until),
        )
    return new_count, retry_sec


def clear_persistent_lockout(scope_key: str) -> None:
    if scope_key:
        db.execute("DELETE FROM auth_lockouts WHERE scope_key = ?", (scope_key,))


def verify_admin_header(request: Request, raw_token: str | None, required: bool = False) -> bool:
    """Verify X-Admin-Token with persistent brute-force lockout against header-stuffing attacks."""
    tok = str(raw_token or "").strip()
    if not tok:
        if required:
            raise HTTPException(status_code=403, detail=i18n.t("server.adminAuthRequired"))
        return False
    client_ip = extract_client_ip(request)
    scope_key = check_persistent_lockout("admin_token", client_ip)
    if db.verify_admin_token(tok):
        clear_persistent_lockout(scope_key)
        return True
    record_persistent_failure(scope_key, ADMIN_TOKEN_TIERS, reset_window_sec=900.0)
    if required:
        raise HTTPException(status_code=403, detail=i18n.t("server.adminAuthRequired"))
    return False


def check_brute_force_guard(bucket_prefix: str, client_ip: str, max_failures: int = 5, window_sec: float = 900.0) -> str:
    ip_scope = normalize_ip_scope(client_ip)
    key = f"bf:{bucket_prefix}:{ip_scope}"
    allowed, retry_sec = RATE_LIMITER.check(key, limit=max_failures, window_sec=window_sec, consume=False)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=i18n.t("server.bruteForceLocked", seconds=retry_sec),
            headers={"Retry-After": str(retry_sec)},
        )
    return key
