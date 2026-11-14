# -*- coding: utf-8 -*-
"""WildCat Decoupled FastAPI Server (REST API v1 + Multi-User Queue Scheduler + Vue 3 SPA Hosting)."""
import os
import sys
import threading
import time
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from wildcat.version import BUILD_VERSION

from . import analytics_db, api, db, i18n, scheduler, security, service, storage, telemetry

ROOT = db.ROOT
FRONTEND_DIR = os.path.join(ROOT, "frontend")

SPA_CSP_POLICY = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: blob:; "
    "font-src 'self' data:; "
    "connect-src 'self'; "
    "frame-ancestors 'none'; "
    "base-uri 'self'"
)


def _background_startup_warmup():
    time.sleep(0.2)
    try:
        analytics_db.sync_pending_tasks_to_db(user_id=None, force=False)
        service.list_tasks(user_id=0, all_users=True)
    except Exception:
        sys.stderr.write(f"[WildCat Startup Warmup Warning]\n{traceback.format_exc()}\n")
        sys.stderr.flush()


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    storage.refresh_storage_paths(force_db=True)
    service.sync_demo_catalog_for_all_users()
    scheduler.SCHEDULER.start()
    threading.Thread(target=_background_startup_warmup, name="wc-startup-warmup", daemon=True).start()
    yield


app = FastAPI(
    title="WildCat",
    version=BUILD_VERSION,
    description="WildCat API",
    lifespan=lifespan,
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 1. Mount Core REST API v1
app.include_router(api.router)


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    """Log full Python traceback to wildcat.log (stderr) while returning a sanitized 500 JSON message."""
    tb = traceback.format_exc()
    sys.stderr.write(f"[WildCat API 500 ERROR] {request.method} {request.url.path}\n{tb}\n")
    sys.stderr.flush()
    return JSONResponse(
        status_code=500,
        content={"detail": i18n.t("server.internalServerError")},
    )


@app.middleware("http")
async def _security_lang_and_cache_middleware(request: Request, call_next):
    """Enforce body-size caps, CSRF, IP rate limits, signed session cookie attachment, security headers & locale."""
    req_lang = request.headers.get("X-WildCat-Lang") or request.query_params.get("lang")
    tok = i18n.set_current_lang(req_lang)
    try:
        path = request.url.path
        cl_raw = request.headers.get("content-length")
        if cl_raw and cl_raw.isdigit() and int(cl_raw) > security.MAX_REQUEST_BODY_BYTES:
            return JSONResponse(status_code=413, content={"detail": i18n.t("server.payloadTooLarge")})

        if path.startswith("/api/"):
            try:
                security.verify_same_origin_csrf(request)
                client_ip = security.extract_client_ip(request)
                security.enforce_rate_limit("api_global", client_ip, limit=360, window_sec=60.0)
            except HTTPException as hexc:
                return JSONResponse(
                    status_code=hexc.status_code,
                    content={"detail": hexc.detail},
                    headers=hexc.headers or None,
                )

        resp = await call_next(request)

        set_uid = getattr(request.state, "set_session_uid", None)
        if set_uid is not None:
            security.set_session_cookie(resp, request, int(set_uid))

        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        resp.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"

        if path in ("/", "/login", "/admin"):
            resp.headers["Content-Security-Policy"] = SPA_CSP_POLICY

        if path in ("/", "/login", "/admin") or path.startswith("/static/"):
            resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            resp.headers["Pragma"] = "no-cache"
            resp.headers["Expires"] = "0"
        return resp
    finally:
        i18n.reset_current_lang(tok)


SAFE_PREVIEW_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _resolve_safe_file(
    rel_path: str,
    candidate_roots: list[str],
    allowed_exts: set[str] | None = None,
) -> str:
    clean = str(rel_path or "").replace("\\", "/").lstrip("/")
    if not clean or "\x00" in clean:
        raise HTTPException(status_code=404, detail="Not found")
    parts = [p for p in clean.split("/") if p and p != "."]
    if not parts or any(p == ".." for p in parts):
        raise HTTPException(status_code=404, detail="Not found")
    if allowed_exts is not None:
        ext = os.path.splitext(parts[-1])[1].lower()
        if ext not in allowed_exts:
            raise HTTPException(status_code=404, detail="Not found")

    for root_dir in candidate_roots:
        if not root_dir or not os.path.isdir(root_dir):
            continue
        root_abs = os.path.abspath(root_dir)
        cur = root_abs
        blocked = False
        for seg in parts:
            cur = os.path.join(cur, seg)
            if os.path.islink(cur):
                blocked = True
                break
        if blocked:
            continue
        cand_abs = os.path.abspath(cur)
        if not os.path.normcase(cand_abs).startswith(os.path.normcase(root_abs + os.sep)):
            continue
        if os.path.isfile(cand_abs):
            return cand_abs
    raise HTTPException(status_code=404, detail="Not found")


# 2. Serve Dynamic Storage Roots (/f/cache & /f/run support live Admin cache directory switching) + Frontend Static Bundle
os.makedirs(storage.get_cache_dir(), exist_ok=True)
os.makedirs(storage.get_run_root(), exist_ok=True)
os.makedirs(FRONTEND_DIR, exist_ok=True)


@app.api_route("/f/cache/{rel_path:path}", methods=["GET", "HEAD"], include_in_schema=False)
def serve_cache_file(rel_path: str):
    fpath = _resolve_safe_file(
        rel_path,
        [storage.get_cache_dir(), storage.DEFAULT_CACHE_DIR],
        allowed_exts=SAFE_PREVIEW_IMAGE_EXTS,
    )
    return FileResponse(fpath)


@app.api_route("/f/run/{rel_path:path}", methods=["GET", "HEAD"], include_in_schema=False)
def serve_run_file(rel_path: str):
    fpath = _resolve_safe_file(
        rel_path,
        [storage.get_run_root(), storage.DEFAULT_RUN_ROOT],
        allowed_exts=SAFE_PREVIEW_IMAGE_EXTS,
    )
    return FileResponse(fpath)


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="frontend_static")


# 3. Serve Vue 3 Single-Page Application Entrypoints
@app.get("/", include_in_schema=False)
@app.get("/login", include_in_schema=False)
@app.get("/admin", include_in_schema=False)
def serve_spa():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("WC_PORT", "8765"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
