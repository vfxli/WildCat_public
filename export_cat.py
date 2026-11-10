# -*- coding: utf-8 -*-
"""Export a WildCat checkpoint as a TorchScript model (.pt) for Nuke's CatFileCreator.
Usage: python export_cat.py config/xxx.json [step]"""
import json
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
_VENV_SITE = os.path.join(_ROOT, ".venv", "lib", "python3.12", "site-packages")
if os.path.isdir(_VENV_SITE) and _VENV_SITE not in sys.path:
    sys.path.insert(0, _VENV_SITE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

from wildcat import cat_format
from wildcat.models import build_generator


def main():
    cfg = json.load(open(sys.argv[1]))
    run_dir = os.path.join("run", cfg["run_name"])
    ck_name = "latest.pt" if len(sys.argv) < 3 else "step_%06d.pt" % int(sys.argv[2])
    ck_path = os.path.join(run_dir, "ckpt", ck_name)
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)

    mo = ck.get("config", cfg).get("model", {})
    model_size = mo.get("model_size", "Medium")
    c_in = int(mo.get("in_channels", 3))
    c_out = int(mo.get("out_channels", c_in))

    net, _ = build_generator(mo)
    net.load_state_dict(ck["G"])
    net.eval()

    out_pt = os.path.join(run_dir, cfg["run_name"] + ".pt")
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
    main()
