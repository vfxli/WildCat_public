# -*- coding: utf-8 -*-
import json
import os
import shutil
import sys

from . import log


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def prepare_run(cfg, cfg_path=None, root=None):
    """Create run/<run_name>/ (ckpt/), back up the json, init log."""
    custom_run = str((cfg.get("paths") or {}).get("run_dir") or "").strip()
    if custom_run:
        run_dir = custom_run
    else:
        base_root = root or os.environ.get("WC_RUN_ROOT") or "run"
        run_dir = os.path.join(base_root, cfg["run_name"])
    ck = os.path.join(run_dir, "ckpt")
    resume = bool(cfg.get("train", {}).get("resume", False))
    if not resume and os.path.isdir(ck) and os.listdir(ck):
        print("run already trained, refusing to overwrite: " + run_dir)
        sys.exit(1)
    os.makedirs(ck, exist_ok=True)
    if cfg_path:
        dst = os.path.join(run_dir, cfg["run_name"] + ".json")
        if os.path.abspath(cfg_path) != os.path.abspath(dst):
            shutil.copy(cfg_path, dst)
    else:
        with open(os.path.join(run_dir, cfg["run_name"] + ".json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    log.init(os.path.join(run_dir, "train.log"))
    return run_dir
