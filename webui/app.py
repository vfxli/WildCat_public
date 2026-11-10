# -*- coding: utf-8 -*-
"""WildCat Web & API Server Entrypoint (FastAPI + Vue 3 SPA)."""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_VENV_SITE = os.path.join(ROOT, ".venv", "lib", "python3.12", "site-packages")
if os.path.isdir(_VENV_SITE) and _VENV_SITE not in sys.path:
    sys.path.insert(0, _VENV_SITE)
sys.path.insert(0, ROOT)

from backend.server import app  # noqa: E402,F401

if __name__ == "__main__":
    import argparse
    import uvicorn

    default_port = int(os.environ.get("WILDCAT_PORT") or os.environ.get("WC_PORT") or "8765")
    parser = argparse.ArgumentParser(description="WildCat Web & API Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=default_port)
    args, _ = parser.parse_known_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
