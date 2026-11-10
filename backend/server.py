# -*- coding: utf-8 -*-
"""WildCat Decoupled FastAPI Server (REST API v1 + Multi-User Queue Scheduler + Vue 3 SPA Hosting)."""
import os
import sys
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from wildcat.version import BUILD_VERSION

from . import api, db, scheduler, storage, telemetry

ROOT = db.ROOT
FRONTEND_DIR = os.path.join(ROOT, "frontend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    scheduler.SCHEDULER.start()
    yield


app = FastAPI(
    title="WildCat",
    version=BUILD_VERSION,
    description="WildCat API",
    lifespan=lifespan,
)

# 1. Mount Core REST API v1
app.include_router(api.router)


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    """Ensure every unhandled server exception logs a full Python traceback to wildcat.log and returns clear details."""
    tb = traceback.format_exc()
    sys.stderr.write(f"[WildCat API 500 ERROR] {request.method} {request.url.path}\n{tb}\n")
    sys.stderr.flush()
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {exc}\n\n{tb}"},
    )


@app.middleware("http")
async def _no_stale_frontend(request, call_next):
    """Always revalidate the SPA and its static assets so UI updates show up without a hard refresh."""
    resp = await call_next(request)
    path = request.url.path
    if path in ("/", "/login", "/admin") or path.startswith("/static/"):
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
    return resp


# 2. Mount Static Directories for Preview Cache, Run Artifacts, and Frontend Bundle
os.makedirs(telemetry.CACHE_DIR, exist_ok=True)
os.makedirs(storage.RUN_ROOT, exist_ok=True)
os.makedirs(FRONTEND_DIR, exist_ok=True)

app.mount("/f/cache", StaticFiles(directory=telemetry.CACHE_DIR), name="cache_files")
app.mount("/f/run", StaticFiles(directory=storage.RUN_ROOT), name="run_files")
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
