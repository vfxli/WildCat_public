# -*- coding: utf-8 -*-
"""WildCat trainer CLI. Usage: python train.py config/xxx.json"""
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


def _fallback_log_path(cfg_arg, cfg_dict):
    run_name = None
    if isinstance(cfg_dict, dict) and cfg_dict.get("run_name"):
        run_name = str(cfg_dict["run_name"])
    elif cfg_arg and os.path.isfile(cfg_arg):
        try:
            with open(cfg_arg, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            if isinstance(raw, dict) and raw.get("run_name"):
                run_name = str(raw["run_name"])
        except Exception:
            pass
    if run_name:
        run_root = os.environ.get("WC_RUN_ROOT") or os.path.join(_ROOT, "run")
        rdir = os.path.join(run_root, run_name)
        os.makedirs(rdir, exist_ok=True)
        return os.path.join(rdir, "train.log")
    return None


def main():
    cfg = None
    cfg_arg = sys.argv[1] if len(sys.argv) > 1 else ""
    log_mod = None
    try:
        from wildcat import config, log as _log
        from wildcat.trainer import Trainer

        log_mod = _log
        cfg = config.load(cfg_arg)
        Trainer(cfg, cfg_path=cfg_arg).train()
    except KeyboardInterrupt:
        sys.exit(130)
    except BaseException:
        tb = traceback.format_exc()
        sys.stderr.write(tb + "\n")
        sys.stderr.flush()
        try:
            lp = getattr(log_mod, "_FILE", None) if log_mod is not None else None
            if not lp:
                lp = _fallback_log_path(cfg_arg, cfg)
            if lp:
                with open(lp, "a", encoding="utf-8") as f:
                    f.write(tb + "\n")
                    f.flush()
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
