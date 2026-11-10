# -*- coding: utf-8 -*-
"""WildCat inference CLI (supports 1:1 baseline as well as Lossless & Lossy speedups).
Usage: python infer.py config/xxx.json [sequence_dir_or_pattern] [step]"""
from concurrent.futures import ThreadPoolExecutor
import json
import os
import sys
import time

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

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from wildcat import color
from wildcat.data import load_sequence
from wildcat.models import build_generator
from wildcat.preview import infer_image


@torch.no_grad()
def main():
    cfg = json.load(open(sys.argv[1]))
    args = sys.argv[2:]
    step_arg = None
    if args and args[-1].isdigit():
        step_arg = int(args.pop())
    pattern = args[0] if args else cfg["paths"]["source"]

    run_dir = os.path.join("run", cfg["run_name"])
    ck_name = "latest.pt" if step_arg is None else "step_%06d.pt" % step_arg
    ck = torch.load(
        os.path.join(run_dir, "ckpt", ck_name), map_location="cpu", weights_only=False
    )
    out_dir = os.path.join(run_dir, "infer")
    os.makedirs(out_dir, exist_ok=True)

    opts = cfg.get("optimizations", {})
    lossless = opts.get("lossless", {})
    lossy = opts.get("lossy", {})

    opt_cudnn_pad = bool(lossless.get("cudnn_pad", False))
    opt_async_io = bool(lossless.get("async_io", False))
    opt_tf32 = bool(lossy.get("tf32", False))
    prec = lossy.get("precision") or cfg.get("train", {}).get("precision", "fp32")
    opt_nhwc = bool(lossy.get("channels_last", False))

    torch.backends.cuda.matmul.allow_tf32 = opt_tf32
    torch.backends.cudnn.allow_tf32 = opt_tf32

    mo = ck.get("config", cfg).get("model", {})
    net, _ = build_generator(mo, cudnn_pad=True)
    net.load_state_dict(ck["G"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = net.to(device).eval()
    if opt_nhwc and device == "cuda":
        net = net.to(memory_format=torch.channels_last)

    amp_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": None,
    }.get(prec, None)
    ac = dict(dtype=amp_dtype, enabled=(amp_dtype is not None and device == "cuda"))

    c_in = int(mo.get("in_channels", 3))
    col = cfg.get("color") or ck.get("config", {}).get("color") or {}
    working_cs = col.get("working_colorspace", "ACEScg")
    source_cs = col.get("source_colorspace", "auto")
    output_cs = col.get("output_colorspace", working_cs)
    ocio_cfg = col.get("ocio_config", "")
    out_fmt = col.get("output_format", "auto")  # 'exr_half' | 'exr_float' | 'png_16' | 'auto'

    print(
        "ckpt step %d | input %s | color=%s->%s | cudnn_pad=%s tf32=%s prec=%s nhwc=%s"
        % (ck.get("step", -1), pattern, source_cs, working_cs, opt_cudnn_pad, opt_tf32, prec, opt_nhwc)
    )

    files, frames = load_sequence(
        pattern,
        c_in,
        input_colorspace=source_cs,
        working_colorspace=working_cs,
        ocio_config_path=ocio_cfg,
    )
    pool = ThreadPoolExecutor(max_workers=4) if opt_async_io else None
    t0 = time.time()

    def _save_output(out_path, arr_rgb, write_as_exr, exr_depth):
        if write_as_exr:
            color.write_exr_file(out_path, arr_rgb, bit_depth=exr_depth)
        else:
            clipped = np.clip(arr_rgb, 0.0, 1.0)
            if clipped.shape[2] >= 3:
                clipped = clipped[..., [2, 1, 0] + list(range(3, clipped.shape[2]))]
            u16 = (clipped * 65535.0).astype(np.uint16)
            cv2.imwrite(out_path, u16)
        print(out_path)

    for path, img in zip(files, frames):
        x = torch.from_numpy(np.asarray(img, dtype=np.float32)).permute(2, 0, 1)[None].to(device)
        y = infer_image(net, x, amp_dtype if ac["enabled"] else None, opt_nhwc and device == "cuda")
        y = y[0].permute(1, 2, 0).cpu().numpy()

        stem, ext = os.path.splitext(os.path.basename(path))
        write_exr = out_fmt in ("exr", "exr_half", "exr_float") or (out_fmt == "auto" and ext.lower() == ".exr")
        exr_depth = "float" if out_fmt == "exr_float" else "half"

        if y.shape[2] >= 3 and output_cs and output_cs != working_cs and output_cs != "Raw":
            y = color.transform_colorspace(y, working_cs, output_cs, ocio_cfg)

        out_ext = ".exr" if write_exr else (".png" if out_fmt == "png_16" else ext)
        out = os.path.join(out_dir, stem + out_ext)
        if pool is not None:
            pool.submit(_save_output, out, y, write_exr, exr_depth)
        else:
            _save_output(out, y, write_exr, exr_depth)

    if pool is not None:
        pool.shutdown(wait=True)
    dt = max(time.time() - t0, 1e-6)
    print(
        "Rendered %d frames in %.2fs (%.1f fps)"
        % (len(files), dt, len(files) / dt)
    )


if __name__ == "__main__":
    import traceback

    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except BaseException:
        sys.stderr.write(traceback.format_exc() + "\n")
        sys.stderr.flush()
        sys.exit(1)
