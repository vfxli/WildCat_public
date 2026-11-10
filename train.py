# -*- coding: utf-8 -*-
"""WildCat trainer CLI. Usage: python train.py config/xxx.json"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
_VENV_SITE = os.path.join(_ROOT, ".venv", "lib", "python3.12", "site-packages")
if os.path.isdir(_VENV_SITE) and _VENV_SITE not in sys.path:
    sys.path.insert(0, _VENV_SITE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from wildcat import config
from wildcat.trainer import Trainer


def main():
    cfg = config.load(sys.argv[1])
    Trainer(cfg, cfg_path=sys.argv[1]).train()


if __name__ == "__main__":
    main()
