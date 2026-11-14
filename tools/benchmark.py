# -*- coding: utf-8 -*-
"""WildCat speed benchmark: steady-state steps/s per multi-resolution stage.

Usage: python tools/benchmark.py --src 'dir/src.####.exr' --gt 'dir/gt.####.exr' \
         [--model Medium] [--batch 8] [--crop 256] [--engine fast|reference] [--precision adaptive] \
         [--steps-per-stage 600] [--vram-gb 16]
"""
import argparse
import contextlib
import io
import json
import os
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--model", default="Medium")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--engine", default="fast")
    ap.add_argument("--precision", default="adaptive")
    ap.add_argument("--steps-per-stage", type=int, default=600)
    ap.add_argument("--vram-gb", type=float, default=0)
    ap.add_argument("--multires", type=int, default=1)
    ap.add_argument("--extra", default="{}", help="json merged into advanced config")
    a = ap.parse_args()
    if a.vram_gb:
        os.environ["WILDCAT_VRAM_LIMIT_GB"] = str(a.vram_gb)
    from wildcat.trainer import Trainer
    from wildcat import data

    n = len(data.expand_pattern(a.src))
    stages = 3 if a.multires else 1
    total = a.steps_per_stage * stages
    with open("config/default.json", "r", encoding="utf-8") as _f:
        c = json.load(_f)
    c["run_name"] = "_bench"
    shutil.rmtree("run/_bench", ignore_errors=True)
    c["paths"] = {"source": a.src, "target": a.gt}
    c["model"]["model_size"] = a.model
    c["train"].update(epochs=0, total_steps=total, batch_size_auto=False, batch_size=a.batch,
                      crop_size=a.crop, multi_resolution=bool(a.multires), precision=a.precision)
    c["optimizations"] = {"engine": a.engine, "lossless": {"preview_gpu": "off"}}
    extra = json.loads(a.extra)
    if extra:
        c["advanced"] = extra
    c["io"].update(preview_enabled=False, checkpoint_interval=10**9, log_interval=10, export_cat=False)
    rows = []
    t = Trainer(c, on_step=lambda d: rows.append((d["step"], d["stage_scale"], d["sps"])))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        t.train()
    import numpy as np
    out = {}
    for sc in sorted({r[1] for r in rows}, reverse=True):
        v = [r[2] for r in rows if r[1] == sc and r[2] > 0]
        v = v[len(v) // 3:]  # drop warm-up / adaptive benchmark
        out["1/%d" % sc] = round(float(np.median(v)), 1) if v else None
    res = re.findall(r"data=(\w+)", buf.getvalue())
    prec = re.findall(r"adaptive precision: ([^\n]*)", buf.getvalue())
    print(json.dumps({"model": a.model, "batch": t.batch_size, "crop": a.crop, "frames": n,
                      "engine": a.engine, "precision": a.precision, "vram_gb": a.vram_gb or None,
                      "sps": out, "data": res, "adaptive": prec}))


if __name__ == "__main__":
    main()
