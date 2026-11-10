# -*- coding: utf-8 -*-
"""WildCat Decoupled FastAPI Server (REST API v1 + Multi-User Queue Scheduler + Vue 3 SPA Hosting)."""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

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
    version="2.0.0",
    description="WildCat API",
    lifespan=lifespan,
)

# 1. Mount Core REST API v1
app.include_router(api.router)


@app.middleware("http")
async def _no_stale_frontend(request, call_next):
    """Always revalidate the SPA and its static assets so UI updates show up without a hard refresh."""
    resp = await call_next(request)
    path = request.url.path
    if path in ("/", "/login", "/admin") or path.startswith("/static/"):
        resp.headers["Cache-Control"] = "no-cache"
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
