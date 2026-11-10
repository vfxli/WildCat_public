# -*- coding: utf-8 -*-
"""WildCat smoke test: run after every code change.

    .venv/bin/python tools/smoke_test.py            # ~2-4 minutes on one GPU
    .venv/bin/python tools/smoke_test.py --gpu 0    # choose the GPU

What it checks (exit code 0 = all passed):
  1. imports of the backend and the training package, and the CopyCat training invariants
     (bicubic pyramid, sampling without replacement, 4-px crop grid, lr schedule, auto batch,
     multi-scale loss, Adam settings)
  2. a short real training run (7 frames, 3003 steps, default config) finishes, the loss
     goes down and the final full-frame MSE stays in the expected range
  3. checkpoints, contact sheets, .summary and the per-checkpoint model file are written
  4. the model exports (export_cat.py) and reloads
It does not touch the running web service or its database.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

RUN = "_smoke_test"
# Reference: 4 seeds of this exact setting gave 0.00056-0.00062 (Nuke 17 CopyCat: 0.00059-0.00062).
MSE_RANGE = (0.00045, 0.00075)


INVARIANTS = r"""
import inspect, numpy as np, torch
import wildcat.data as D, wildcat.trainer as T, wildcat.engine as E
# multi-resolution downscale = bicubic, align_corners=True
assert D.PYRAMID_MODE == "bicubic", "pyramid must be bicubic"
assert "align_corners=True" in inspect.getsource(D.downscale_tensor)
# frames without replacement: with batch == N every batch holds every frame once
s = D.EpochSampler(7, seed=0)
p = s.plan(50, 7, [(855, 772)] * 7, 256)
assert all(sorted(row[:, 0].tolist()) == list(range(7)) for row in p), "sampling must be without replacement"
# 4-px crop grid anchored at the bottom edge at full resolution
s.grid = 4
p = s.plan(200, 4, [(855, 772)] * 7, 256)
assert set(((855 - 256 - p[..., 1]) % 4).ravel().tolist()) == {0} and set((p[..., 2] % 4).ravel().tolist()) == {0}
# poly lr power 0.3 over the whole run
assert abs(T.poly_lr(1e-4, 3002, 3003) - 1e-4 * (1 / 3003) ** 0.3) < 1e-12
# Auto batch 8, multi-scale loss always on, Adam defaults
assert T.NUKE_AUTO_BATCH == 8
src = inspect.getsource(T.Trainer)
assert "multi_scale=True" in src and "betas=(0.9, 0.999), eps=1e-8" in src
assert "forward_all" in inspect.getsource(E.StepEngine._body)
print("bicubic pyramid, no-replacement sampling, 4-px crop grid, poly lr 0.3, auto batch 8, multi-scale loss, Adam")
"""


def fail(msg):
    print("FAIL:", msg)
    sys.exit(1)


def ok(msg):
    print("ok  :", msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "1"))
    ap.add_argument("--keep", action="store_true", help="keep run/_smoke_test afterwards")
    a = ap.parse_args()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu))

    # 1. imports
    r = subprocess.run([sys.executable, "-c", "import backend.server, wildcat.trainer, infer, export_cat"],
                       cwd=ROOT, capture_output=True, text=True, env=env)
    if r.returncode:
        fail("imports:\n" + r.stderr[-2000:])
    ok("imports")

    # 1b. CopyCat training invariants (verified against Nuke 17; never change without re-validating)
    r = subprocess.run([sys.executable, "-c", INVARIANTS], cwd=ROOT, capture_output=True, text=True, env=env)
    if r.returncode:
        fail("CopyCat training invariant broken:\n" + (r.stdout + r.stderr)[-2000:])
    ok("CopyCat training invariants: " + r.stdout.strip())

    src = "dataset/source/source.####.exr"
    tgt = "dataset/target/target.####.exr"
    if not glob.glob(src.replace("####", "*")):
        fail("test data missing: dataset/source and dataset/target (7 EXR frames each)")

    # 2. short training with the default config
    cfg = json.load(open("config/default.json"))
    cfg["run_name"] = RUN
    cfg["paths"] = {"source": src, "target": tgt}
    cfg["train"].update(epochs=1716, batch_size_auto=False, batch_size=4, seed=11, resume=False)
    cfg["io"].update(checkpoint_interval=1000, preview_interval=500)
    shutil.rmtree(os.path.join("run", RUN), ignore_errors=True)
    os.makedirs("run", exist_ok=True)
    cfg_path = os.path.join("run", RUN + ".json")
    json.dump(cfg, open(cfg_path, "w"), indent=1)
    t0 = time.time()
    r = subprocess.run([sys.executable, "train.py", cfg_path], cwd=ROOT, capture_output=True, text=True, env=env)
    os.remove(cfg_path)
    rd = os.path.join("run", RUN)
    log = open(os.path.join(rd, "train.log")).read() if os.path.isfile(os.path.join(rd, "train.log")) else ""
    if r.returncode or "done." not in log:
        fail("training did not finish:\n" + (r.stdout + r.stderr)[-3000:])
    ok("training finished in %.0fs" % (time.time() - t0))

    losses = []
    for line in log.splitlines():
        if "  total " in line and " step " in line:
            try:
                losses.append(float(line.split("total")[1].split()[0]))
            except (IndexError, ValueError):
                pass
    if len(losses) < 100:
        fail("too few loss lines in train.log (%d)" % len(losses))
    first, last = sum(losses[:20]) / 20, sum(losses[-50:]) / 50
    if not last < first * 0.5:
        fail("loss did not go down enough: first %.5f last %.5f" % (first, last))
    ok("loss went down: %.5f -> %.5f" % (first, last))

    vals = [l for l in log.splitlines() if "val(MSE)" in l]
    if not vals:
        fail("no val(MSE) line in train.log")
    nums = [float(x) for x in vals[-1].split("val(MSE):", 1)[1].split()[1::2]]
    mse = sum(nums) / len(nums)
    if not (MSE_RANGE[0] <= mse <= MSE_RANGE[1]):
        fail("final full-frame MSE %.6f outside expected %s - training behaviour changed?" % (mse, MSE_RANGE))
    ok("final full-frame MSE %.6f (expected %s)" % (mse, MSE_RANGE))

    # 3. artefacts
    need = {
        "checkpoint": glob.glob(os.path.join(rd, "ckpt", "step_*.pt")),
        "contact sheet": glob.glob(os.path.join(rd, "Training_*.png")),
        "summary": glob.glob(os.path.join(rd, "Training_*.summary")),
        "model per checkpoint": glob.glob(os.path.join(rd, "Training_*.cat")) + glob.glob(os.path.join(rd, "Training_*.[0-9]*.pt")),
        "run_info.json": glob.glob(os.path.join(rd, "run_info.json")),
        "previews": glob.glob(os.path.join(rd, "preview_sup", "*", "step_*.png")),
    }
    for k, v in need.items():
        if not v:
            fail("missing %s in %s" % (k, rd))
    ok("artefacts: " + ", ".join("%s %d" % (k, len(v)) for k, v in need.items()))

    # 4. export + reload
    run_cfg = os.path.join(rd, RUN + ".json")
    r = subprocess.run([sys.executable, "export_cat.py", run_cfg], cwd=ROOT, capture_output=True, text=True, env=env)
    pt = os.path.join(rd, RUN + ".pt")
    if r.returncode or not os.path.isfile(pt):
        fail("export_cat.py:\n" + (r.stdout + r.stderr)[-2000:])
    r = subprocess.run([sys.executable, "-c", "import torch,sys; m=torch.jit.load(sys.argv[1]); "
                        "y=m(torch.rand(1,3,256,256)); assert y.shape==(1,3,256,256); print('reload ok')", pt],
                       cwd=ROOT, capture_output=True, text=True, env=env)
    if r.returncode:
        fail("exported TorchScript does not reload:\n" + r.stderr[-2000:])
    ok("export + reload (%s)" % r.stdout.strip())

    if not a.keep:
        shutil.rmtree(rd, ignore_errors=True)
    print("\nALL PASSED")


if __name__ == "__main__":
    main()
