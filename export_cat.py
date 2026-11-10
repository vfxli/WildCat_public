# -*- coding: utf-8 -*-
"""Export a WildCat checkpoint as a TorchScript model (.pt) for Nuke's CatFileCreator.
Usage: python export_cat.py config/xxx.json [step] [--out-pt PATH]"""
import argparse
import json
import os
import sys
import traceback

_ROOT = os.path.dirname(os.path.abspath(__file__))
for _cand in (
    os.path.join(_ROOT, ".venv", "Lib", "site-packages"),
    os.path.join(_ROOT, ".venv", "lib", "python3.12", "site-packages"),
    os.path.join(_ROOT, "runtime", "python", "Lib", "site-packages"),
    os.path.join(_ROOT, "runtime", "python", "lib", "python3.12", "site-packages"),
    os.path.join(_ROOT, "..", "runtime", "python", "Lib", "site-packages"),
    os.path.join(_ROOT, "..", "runtime", "python", "lib", "python3.12", "site-packages"),
    os.path.join(_ROOT, "..", ".venv", "Lib", "site-packages"),
    os.path.join(_ROOT, "..", ".venv", "lib", "python3.12", "site-packages"),
):
    if os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.insert(0, _cand)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("config_path")
    parser.add_argument("step", nargs="?", default=None)
    parser.add_argument("--out-pt", dest="out_pt", default=None)
    args = parser.parse_args()

    import torch
    from wildcat import cat_format
    from wildcat.models import build_generator

    with open(args.config_path, "r", encoding="utf-8") as _f:
        cfg = json.load(_f)
    run_dir = os.path.join("run", cfg["run_name"])
    ck_name = "latest.pt" if args.step is None else "step_%06d.pt" % int(args.step)
    ck_path = os.path.join(run_dir, "ckpt", ck_name)
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)

    mo = ck.get("config", cfg).get("model", {})
    model_size = mo.get("model_size", "Medium")
    c_in = int(mo.get("in_channels", 3))
    c_out = int(mo.get("out_channels", c_in))

    net, _ = build_generator(mo)
    net.load_state_dict(ck["G"])
    net.eval()

    out_pt = args.out_pt or os.path.join(run_dir, cfg["run_name"] + ".pt")
    cat_format.export_torchscript(net.state_dict(), out_pt, mo)
    print(
        "Exported TorchScript %s (model=%s, in=%d, out=%d, step=%d)"
        % (out_pt, model_size, c_in, c_out, ck.get("step", -1))
    )
    print("Convert to .cat in Nuke: CatFileCreator node -> Torch Script File = this file, "
          "Channels In/Out = rgba.red, rgba.green, rgba.blue -> Create .cat file")
    if not cat_format.torch_version_ok_for_nuke():
        print("WARNING: PyTorch %s is newer than 2.11; Nuke 17 may not load this file." % torch.__version__)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except BaseException:
        sys.stderr.write(traceback.format_exc() + "\n")
        sys.stderr.flush()
        sys.exit(1)
