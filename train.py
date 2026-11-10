# -*- coding: utf-8 -*-
"""WildCat trainer CLI. Usage: python train.py config/xxx.json"""
import os
import sys
import traceback

_ROOT = os.path.dirname(os.path.abspath(__file__))
for _cand in (
    os.path.join(_ROOT, ".venv", "lib", "python3.12", "site-packages"),
    os.path.join(_ROOT, "runtime", "python", "lib", "python3.12", "site-packages"),
    os.path.join(_ROOT, "..", "runtime", "python", "lib", "python3.12", "site-packages"),
    os.path.join(_ROOT, "..", ".venv", "lib", "python3.12", "site-packages"),
):
    if os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.insert(0, _cand)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from wildcat import config, log
from wildcat.trainer import Trainer


def main():
    cfg = None
    try:
        cfg = config.load(sys.argv[1])
        Trainer(cfg, cfg_path=sys.argv[1]).train()
    except KeyboardInterrupt:
        sys.exit(130)
    except BaseException:
        tb = traceback.format_exc()
        sys.stderr.write(tb + "\n")
        sys.stderr.flush()
        try:
            lp = getattr(log, "_FILE", None)
            if not lp and isinstance(cfg, dict) and cfg.get("run_name"):
                rdir = os.path.join(_ROOT, "run", str(cfg["run_name"]))
                os.makedirs(rdir, exist_ok=True)
                lp = os.path.join(rdir, "train.log")
            if lp:
                with open(lp, "a", encoding="utf-8") as f:
                    f.write(tb + "\n")
                    f.flush()
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
