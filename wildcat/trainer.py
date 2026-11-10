# -*- coding: utf-8 -*-
"""WildCat trainer: CopyCat-compatible training semantics + fast execution engine.

Training semantics (identical in every engine; there are no options that deviate from them):
  * EncoderDecoder, xavier-uniform weights / zero bias, seeded
  * total_steps = ceil(epochs * N / batch); Auto batch = 8 (memory permitting)
  * frames drawn without replacement (stream of shuffled epoch permutations), uniform crops
  * multi-resolution: 3 equal stages (ceil(T/3)) at 1/4, 1/2, 1/1, crop/4, crop/2, crop,
    frames downscaled bicubic (align_corners=True)
  * Adam(lr=1e-4, betas=(0.9, 0.999), eps=1e-8), lr_k = lr * (1 - (k-1)/T) ** 0.3
  * loss = sum over the 3 internal pyramid levels of MSE(level output, GT resized to that level);
    the reported loss is the finest level (as CopyCat prints)
  * automatic mixed precision "Adaptive" (fp16 + GradScaler where it is faster)

Execution engines (all lossless w.r.t. the semantics above):
  * reference : plain eager PyTorch, data on host, per-step H2D (what Nuke does)
  * fast      : data resident in VRAM (fp16 when exact), one-gather batching, channels_last,
                cuDNN autotune, fused Adam, the whole step captured as a CUDA graph, async logging,
                previews/validation on a second GPU (or side stream), async checkpoint I/O.
"""
import collections
import math
import os
import signal
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from . import config as cfgmod
from . import data, models
from .engine import GraphSafeScaler, StepEngine
from .log import log
from .preview import PreviewWorker, forward_full

NUKE_AUTO_BATCH = 8


def poly_lr(lr_init, step, total_steps, power=0.3):
    """CopyCat schedule: lr at 1-based step k is lr_init * (1 - (k-1)/T) ** power (call with step=k-1)."""
    if total_steps <= 0:
        return lr_init
    clamped = min(max(0, step), total_steps)
    return lr_init * ((1.0 - float(clamped) / float(total_steps)) ** power)


def _bool(v, default=False):
    if v is None:
        return default
    if isinstance(v, str):
        return v.lower() in ("1", "true", "yes", "on", "auto")
    return bool(v)


class Trainer:
    def __init__(self, cfg, cfg_path=None, on_step=None):
        self.cfg = cfg
        self.on_step = on_step
        self.run_dir = cfgmod.prepare_run(cfg, cfg_path)
        io, tr, lo, mo = cfg.get("io", {}), cfg.get("train", {}), cfg.get("loss", {}), cfg.get("model", {})
        opts = cfg.get("optimizations", {}) or {}
        lossless = opts.get("lossless", {}) or {}
        lossy = opts.get("lossy", {}) or {}
        self.io, self.tr, self.lo, self.mo = io, tr, lo, mo
        self.gpu_lock = threading.RLock()

        # ---------------- engine / lossless options ----------------
        engine = opts.get("engine")
        if engine is None:
            # legacy configs: any lossless toggle on -> fast engine
            legacy_on = any(_bool(lossless.get(k)) for k in ("cudnn_pad", "gpu_pyramid", "fused_adam", "async_io", "gpu_resident", "cuda_graph"))
            engine = "fast" if legacy_on else "reference"
        self.engine = "fast" if engine == "fast" else "reference"
        fast = self.engine == "fast"
        self.opt_gpu_resident = lossless.get("gpu_resident", lossless.get("gpu_pyramid", "auto") if fast else False)
        self.opt_cuda_graph = _bool(lossless.get("cuda_graph", fast))
        self.opt_fused_adam = _bool(lossless.get("fused_adam", fast))
        self.opt_channels_last = _bool(lossless.get("channels_last", lossy.get("channels_last", fast)))
        self.opt_benchmark = _bool(lossless.get("cudnn_benchmark", fast))
        self.opt_async_io = _bool(lossless.get("async_io", fast))
        self.preview_gpu = str(os.environ.get("WILDCAT_PREVIEW_GPU") or lossless.get("preview_gpu", "auto"))

        # ---------------- mixed precision (CopyCat: Adaptive / Disabled / Force Enabled) ----------------
        prec = lossy.get("precision") or tr.get("precision") or "adaptive"
        if prec in ("amp", "auto", "nuke"):
            prec = "adaptive"
        if prec not in ("adaptive", "fp32", "fp16"):
            prec = "adaptive"  # bf16 and other non-CopyCat modes were removed
        self.prec = prec
        self.opt_tf32 = True  # libtorch default: cuDNN TF32 allowed

        seed = int(tr.get("seed", 0))
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.benchmark = self.opt_benchmark
        torch.backends.cudnn.allow_tf32 = self.opt_tf32
        torch.backends.cuda.matmul.allow_tf32 = self.opt_tf32

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.is_cuda = self.device.startswith("cuda")
        self.vram_limit = None
        if self.is_cuda:
            torch.cuda.set_device(0)
            self.device_label = "cuda:0 (%s)" % torch.cuda.get_device_name(0)
            lim = os.environ.get("WILDCAT_VRAM_LIMIT_GB") or (cfg.get("train", {}) or {}).get("vram_limit_gb")
            if lim:
                total = torch.cuda.get_device_properties(0).total_memory
                self.vram_limit = min(total, int(float(lim) * 2**30))
                torch.cuda.set_per_process_memory_fraction(self.vram_limit / total, 0)
                self.device_label += " [VRAM capped at %.1f GB]" % (self.vram_limit / 2**30)
        else:
            self.device_label = "CPU (WARNING: torch.cuda.is_available() is False!)"
            self.opt_cuda_graph = False
            self.opt_fused_adam = False
            if self.prec != "fp32":
                self.prec = "fp32"

        # ---------------- weights / architecture lock ----------------
        init_w = tr.get("initial_weights", "")
        if tr.get("resume", False):
            cand = os.path.join(self.run_dir, "ckpt", "latest.pt")
            if os.path.isfile(cand):
                init_w = cand
        if init_w and init_w != "None":
            w_spec = models.inspect_weight_source(
                init_w,
                fallback_model_size=mo.get("model_size", "Medium"),
                fallback_in=int(mo.get("in_channels", 3)),
                fallback_out=int(mo.get("out_channels", mo.get("in_channels", 3))),
            )
            if w_spec.get("lock_model_size"):
                mo["model_size"] = w_spec["model_size"]
            if w_spec.get("lock_channels"):
                mo["in_channels"] = w_spec["in_channels"]
                mo["out_channels"] = w_spec["out_channels"]

        model_size = mo.get("model_size", "Medium")
        self.model_size = model_size
        # cudnn_pad (symmetric padding folded into the conv) is bit-identical math -> always on
        self.raw_G, k_size = models.build_generator(mo, cudnn_pad=True)
        self.raw_G = self.raw_G.to(self.device)
        if self.opt_channels_last and self.is_cuda:
            self.raw_G = self.raw_G.to(memory_format=torch.channels_last)
        n_par = sum(p.numel() for p in self.raw_G.parameters())
        self.c_in = int(mo.get("in_channels", 3))
        self.c_out = int(mo.get("out_channels", self.c_in))

        # ---------------- colour ----------------
        col = cfg.get("color", {})
        self.working_colorspace = col.get("working_colorspace", "ACEScg")
        self.source_colorspace = col.get("source_colorspace", "auto")
        self.target_colorspace = col.get("target_colorspace", "auto")
        self.ocio_config = col.get("ocio_config", "")
        self.view_transform = col.get("view_transform", "ACES 1.0 - SDR Video")

        # ---------------- data (host RAM guarded, fp16 when exact) ----------------
        src_files, self.src_full = data.load_sequence(
            cfg["paths"]["source"], self.c_in, input_colorspace=self.source_colorspace,
            working_colorspace=self.working_colorspace, ocio_config_path=self.ocio_config, dtype="auto",
        )
        tgt_files, self.tgt_full = data.load_sequence(
            cfg["paths"]["target"], self.c_out, input_colorspace=self.target_colorspace,
            working_colorspace=self.working_colorspace, ocio_config_path=self.ocio_config, dtype="auto",
        )
        self.is_exr_sequence = any(p.lower().endswith(".exr") for p in (src_files[:1] + tgt_files[:1]))
        self.use_ocio_display_view = bool(
            self.is_exr_sequence
            or self.source_colorspace in ("ACEScg", "ACES2065-1", "ACEScct", "ACEScc", "Linear sRGB")
            or col.get("apply_ocio_view", False)
        )
        if len(self.src_full) != len(self.tgt_full):
            raise ValueError("Input (%d) and Ground Truth (%d) frame counts do not match" % (len(self.src_full), len(self.tgt_full)))
        for a, b, p in zip(self.src_full, self.tgt_full, tgt_files):
            if a.shape[:2] != b.shape[:2]:
                raise ValueError("Input/Ground Truth size mismatch at %s: %s vs %s" % (p, a.shape[:2], b.shape[:2]))
        self.pair_names = [os.path.splitext(os.path.basename(p))[0] for p in tgt_files]
        n_frames = len(self.src_full)

        # ---------------- crop / batch ----------------
        if tr.get("crop_size_auto", False) or not tr.get("crop_size"):
            self.crop_size = 256
        else:
            self.crop_size = int(tr.get("crop_size", 256))
        min_side = min(min(f.shape[:2]) for f in self.src_full)
        if min_side < min(self.crop_size, 32):
            raise ValueError("Sequences are smaller than the crop size. Please lower the crop size.")

        self.amp_candidates = self._amp_candidates()
        if tr.get("batch_size_auto", False) or not tr.get("batch_size"):
            self.batch_size = self._auto_batch(n_frames)
        else:
            self.batch_size = int(tr.get("batch_size", 8))
        if self.batch_size > n_frames:
            log("WARNING: batch size %d > dataset size %d (Nuke refuses this); clamping to %d"
                % (self.batch_size, n_frames, n_frames))
            self.batch_size = n_frames
        self.batch_size = max(1, self.batch_size)

        dataset_size = max(1, n_frames)
        if tr.get("epochs"):
            self.epochs = int(tr.get("epochs", 10000))
            self.total_steps = max(1, math.ceil(self.epochs * dataset_size / float(self.batch_size)))
        elif tr.get("total_steps"):
            self.total_steps = int(tr["total_steps"])
            self.epochs = max(1, math.ceil(self.total_steps * self.batch_size / float(dataset_size)))
        else:
            self.epochs = 10000
            self.total_steps = max(1, math.ceil(self.epochs * dataset_size / float(self.batch_size)))
        tr["total_steps"] = self.total_steps
        tr["epochs"] = self.epochs
        tr["batch_size"] = self.batch_size
        tr["crop_size"] = self.crop_size
        try:  # authoritative run facts for the web UI / scheduler (progress = step / total_steps)
            import json as _json
            with open(os.path.join(self.run_dir, "run_info.json"), "w", encoding="utf-8") as f:
                _json.dump({"total_steps": self.total_steps, "epochs": self.epochs, "batch_size": self.batch_size,
                            "crop_size": self.crop_size, "frames": n_frames}, f)
        except Exception:
            pass

        # ---------------- optimiser ----------------
        self.lr = float(tr.get("learning_rate", 1e-4))
        self.lr_power = float(tr.get("lr_power", 0.3))
        params = list(self.raw_G.parameters())
        if self.opt_fused_adam and self.is_cuda:
            self.lr_t = torch.tensor(self.lr, dtype=torch.float32, device=self.device)
            self.optG = torch.optim.Adam(params, lr=self.lr_t, betas=(0.9, 0.999), eps=1e-8,
                                         fused=True, capturable=True)
        else:
            self.lr_t = None
            self.optG = torch.optim.Adam(params, lr=self.lr, betas=(0.9, 0.999), eps=1e-8)
        self.scaler = GraphSafeScaler(self.device, enabled=self.is_cuda)

        self.step0 = 0
        if init_w and init_w != "None":
            ck = models.load_weights_into_model(self.raw_G, init_w, model_size=model_size, device=self.device)
            if isinstance(ck, dict) and tr.get("resume", False):
                if "optG" in ck:
                    try:
                        self.optG.load_state_dict(ck["optG"])
                    except Exception as e:
                        log("optimizer state not restored (%s)" % e)
                if "scaler" in ck:
                    self.scaler.load_state_dict(ck["scaler"])
                self.step0 = int(ck.get("step", 0))
            log("loaded weights from %s (start step %d)" % (init_w, self.step0))
        if self.lr_t is not None:
            for pg in self.optG.param_groups:
                pg["lr"] = self.lr_t
        self.G = self.raw_G

        # CopyCat-style artefacts in the run directory: Training_YYMMDD_HHMMSS.<step>.cat/.png + .summary
        self.nuke_prefix = os.path.join(self.run_dir, "Training_%s" % time.strftime("%y%m%d_%H%M%S"))
        if tr.get("resume", False):
            import glob as _glob
            prev = sorted(_glob.glob(os.path.join(self.run_dir, "Training_*.summary")))
            if prev:
                self.nuke_prefix = prev[-1][: -len(".summary")]  # continue the same CopyCat-style run name
        try:
            import json as _json
            with open(os.path.join(self.run_dir, "run_info.json"), "w", encoding="utf-8") as f:
                _json.dump(
                    {
                        "total_steps": self.total_steps,
                        "epochs": self.epochs,
                        "batch_size": self.batch_size,
                        "crop_size": self.crop_size,
                        "frames": n_frames,
                        "nuke_prefix": os.path.basename(self.nuke_prefix),
                    },
                    f,
                )
        except Exception:
            pass
        self.export_cat_on_checkpoint = _bool(io.get("export_cat", True))
        self.io_pool = ThreadPoolExecutor(max_workers=1) if self.opt_async_io else None
        self.preview_worker = None
        self.preview_device = self._resolve_preview_device()

        from . import cat_format as _cf
        if not _cf.torch_version_ok_for_nuke():
            log("WARNING: PyTorch %s is newer than 2.11; Nuke 17 may not load the exported TorchScript files."
                % torch.__version__)
        log(
            "WildCat | device=%s | engine=%s | model=%s (k=%d, %.2fM params) | pairs=%d (%dx%d) | "
            "batch=%d crop=%d | epochs=%d steps=%d | lr=%.1e multires=%s | precision=%s | "
            "lossless=[resident=%s graph=%s fused=%s nhwc=%s autotune=%s async_io=%s preview=%s]"
            % (
                self.device_label, self.engine, model_size, k_size, n_par / 1e6, n_frames,
                self.src_full[0].shape[1], self.src_full[0].shape[0], self.batch_size, self.crop_size,
                self.epochs, self.total_steps, self.lr, bool(tr.get("multi_resolution", True)), self.prec,
                self.opt_gpu_resident, self.opt_cuda_graph, self.opt_fused_adam, self.opt_channels_last,
                self.opt_benchmark, self.opt_async_io, self.preview_device,
            )
        )

    # ======================================================================== helpers
    def _amp_candidates(self):
        if not self.is_cuda:
            return [None]
        p = self.prec
        if p == "fp32":
            return [None]
        if p == "fp16":
            return [torch.float16]
        return [torch.float16, None]  # adaptive: benchmark per stage, keep the faster

    def _resolve_preview_device(self):
        if not self.is_cuda:
            return "cpu"
        pg = self.preview_gpu
        if pg in ("same", "0"):
            return "cuda:0"
        if pg in ("off", "none", "cpu"):
            return "cpu" if pg == "cpu" else "off"
        if pg == "auto":
            return "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
        try:
            i = int(pg)
            return "cuda:%d" % i if i < torch.cuda.device_count() else "cuda:0"
        except ValueError:
            return "cuda:0"

    def _free_vram(self):
        free, _ = torch.cuda.mem_get_info()
        if self.vram_limit is not None:
            free = min(free, self.vram_limit - torch.cuda.memory_reserved())
        return max(0, free)

    def _peak_bytes(self, bs, c, amp_dtype):
        """Measure peak VRAM of one fwd/bwd at (bs, c) without touching weights or optimiser."""
        torch.cuda.synchronize()
        # cuDNN autotune workspaces would inflate the peak (they grow with free memory): measure
        # with heuristic algorithms; the 1.6x margin used by callers covers autotune and graphs.
        bench_prev = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        mf = torch.channels_last if self.opt_channels_last else torch.contiguous_format
        x = torch.rand((bs, self.c_in, c, c), device=self.device).contiguous(memory_format=mf)
        with torch.autocast("cuda", dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
            out = self.raw_G(x)
        loss = out.float().square().mean()
        loss.backward()
        for p in self.raw_G.parameters():
            p.grad = None
        del x, out, loss
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - base
        torch.backends.cudnn.benchmark = bench_prev
        torch.cuda.empty_cache()
        return peak

    def _auto_batch(self, n_frames):
        """Nuke 'Auto' batch = 8 when memory allows (it refuses datasets smaller than 8 frames)."""
        bs = NUKE_AUTO_BATCH
        if not self.is_cuda:
            return min(bs, n_frames)
        free = self._free_vram()
        amp = self.amp_candidates[-1] if self.prec == "adaptive" else self.amp_candidates[0]
        try:
            while bs > 1:
                need = self._peak_bytes(bs, self.crop_size, amp) * 1.6 + 512 * 2**20
                if need < free * 0.9:
                    break
                bs //= 2
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            bs = max(1, bs // 2)
        return min(bs, n_frames)

    # ======================================================================== checkpoint
    def _opt_state_cpu(self):
        sd = self.optG.state_dict()

        def cpu(o):
            if isinstance(o, torch.Tensor):
                return o.detach().cpu().clone()
            if isinstance(o, dict):
                return {k: cpu(v) for k, v in o.items()}
            if isinstance(o, list):
                return [cpu(v) for v in o]
            return o

        sd = cpu(sd)
        for g in sd.get("param_groups", []):
            if isinstance(g.get("lr"), torch.Tensor):
                g["lr"] = float(g["lr"])
        return sd

    def _save_checkpoint(self, step):
        if self.is_cuda:
            torch.cuda.synchronize()
        sd_cpu = {k: v.detach().cpu().contiguous() for k, v in self.raw_G.state_dict().items()}
        ck = {
            "G": sd_cpu,
            "step": step,
            "config": self.cfg,
            "optG": self._opt_state_cpu(),
            "scaler": self.scaler.state_dict() if self.is_cuda else {},
        }
        p = os.path.join(self.run_dir, "ckpt", "step_%06d.pt" % step)
        latest_p = os.path.join(self.run_dir, "ckpt", "latest.pt")

        model_p = "%s.%d.pt" % (self.nuke_prefix, step)  # TorchScript, convert with Nuke CatFileCreator

        def _write():
            try:
                tmp = latest_p + ".tmp"
                torch.save(ck, tmp)
                os.replace(tmp, latest_p)
                p_tmp = p + ".tmp"
                torch.save(ck, p_tmp)
                os.replace(p_tmp, p)
                log("checkpoint -> %s" % p)
            except Exception:
                log("checkpoint write failed:\n%s" % traceback.format_exc())
                return
            if self.export_cat_on_checkpoint:
                try:
                    from . import cat_format
                    cat_format.export_torchscript(sd_cpu, model_p, self.mo)
                    log(" [*] TorchScript model saved at %s" % model_p)
                except Exception:
                    log("model export failed:\n%s" % traceback.format_exc())

        if self.io_pool is not None:
            self.io_pool.submit(_write)
        else:
            _write()

    # ======================================================================== preview / val
    def _snapshot(self):
        with torch.no_grad():
            snap = {k: v.detach().clone() for k, v in self.raw_G.state_dict().items()}
        ev = None
        if self.is_cuda:
            ev = torch.cuda.Event()
            ev.record()
        return snap, ev

    def _preview(self, step, do_preview, do_val, crops=None, scale=1):
        if self.preview_device == "off":
            if do_val:
                self.validate(step)
            return
        if self.preview_worker is None:
            amp = None if self.preview_device != self.device else torch.float16 if self.is_cuda else None
            self.preview_worker = PreviewWorker(self, self.preview_device, amp_dtype=amp)
        if not do_val and (self.preview_worker.busy.is_set() or self.preview_worker.q.full()):
            return  # skip this preview rather than stall training
        snap, ev = self._snapshot()
        if crops is not None:
            crops = (crops[0][:3].detach().clone(), crops[1][:3].detach().clone(), int(scale))
        self.preview_worker.submit(step, snap, ev, do_preview, do_val, crops)

    @torch.no_grad()
    def validate(self, step):
        self.raw_G.eval()
        parts = []
        for i, (a, b) in enumerate(zip(self.src_full, self.tgt_full)):
            out = forward_full(self.raw_G, a, self.device, None, self.opt_channels_last)
            gt = torch.from_numpy(np.asarray(b, dtype=np.float32)).permute(2, 0, 1)[None].to(self.device)
            parts.append("f%d %.6f" % (i + 1, torch.mean((out - gt) ** 2).item()))
        log("val(MSE): " + "  ".join(parts))
        self.raw_G.train()

    def save_preview_sup(self, step):
        """Synchronous preview (kept for API compatibility)."""
        self._preview(step, True, False)
        if self.preview_worker is not None:
            while self.preview_worker.busy.is_set() or not self.preview_worker.q.empty():
                time.sleep(0.05)

    # ======================================================================== training
    def _stage_list(self):
        total = self.total_steps
        if self.tr.get("multi_resolution", True):
            s = math.ceil(total / 3.0)
            return [(4, min(total, s)), (2, min(total, 2 * s)), (1, total)]
        return [(1, total)]

    def _make_source(self, scale):
        """VRAM-resident source when it fits (fp16 storage when lossless), else host prefetch."""
        want = self.opt_gpu_resident
        if self.is_cuda:
            log("vram before stage 1/%d: allocated %.2f GB, reserved %.2f GB"
                % (scale, torch.cuda.memory_allocated() / 2**30, torch.cuda.memory_reserved() / 2**30))
        if self.is_cuda and want not in (False, "false", "off", "0"):
            div2 = float(scale * scale)
            per_val = 2 if self.src_full[0].dtype == np.float16 and self.tgt_full[0].dtype == np.float16 else 4
            need = data.dataset_vram_bytes(self.src_full, self.tgt_full, per_val) / div2
            free = self._free_vram()
            reserve = self._stage_peak * 1.6 + 768 * 2**20
            # Keep >= 40% of VRAM free for cuDNN workspaces: measured on a 5090, a 21 GB resident set
            # slowed the full-res stage by ~11% while host prefetch costs only 0-4%.
            if want in (True, "true", "on", "1") or (need + reserve < free * 0.92 and need < free * 0.60):
                try:
                    sdt = torch.float16 if per_val == 2 else torch.float32
                    return data.GPUCropSource(self.src_full, self.tgt_full, scale, self.device, sdt)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    log("dataset does not fit in VRAM, falling back to host prefetch")
            else:
                log("dataset (%.2f GB @1/%d) + training (%.2f GB) exceed free VRAM (%.2f GB): host prefetch"
                    % (need / 1e9, scale, reserve / 1e9, free / 1e9))
        return data.CPUCropSource(self.src_full, self.tgt_full, scale, self.device)

    def train(self):
        io, tr = self.io, self.tr
        total = self.total_steps
        stages = self._stage_list()
        log_iv = max(1, int(io.get("log_interval", 10)))
        prev_iv = max(1, int(io.get("preview_interval", 100)))
        ckpt_iv = max(1, int(io.get("checkpoint_interval", 1000)))
        preview_on = bool(io.get("preview_enabled", True))

        sampler = data.EpochSampler(len(self.src_full), seed=self.seed + 1000003 * (self.step0 > 0) + self.step0)
        step = self.step0
        pending = collections.deque()
        self._log_events = []
        last_ev = [None, None]  # (event, step) of previous log point
        stop = {"flag": False}
        stop_file = os.path.join(self.run_dir, ".stop")
        if os.path.exists(stop_file):
            try:
                os.remove(stop_file)
            except OSError:
                pass

        def _sig(*_):
            stop["flag"] = True

        try:  # pause (SIGINT/SIGBREAK from the scheduler) / stop: finish the current step, then checkpoint
            signal.signal(signal.SIGTERM, _sig)
            signal.signal(signal.SIGINT, _sig)
            if hasattr(signal, "SIGBREAK"):
                signal.signal(signal.SIGBREAK, _sig)
        except Exception:
            pass

        def emit_ready(block=False):
            while pending:
                s, lr_v, scl, buf, ev = pending[0]
                if not block and ev is not None and not ev.query():
                    return
                if ev is not None:
                    ev.synchronize()
                pending.popleft()
                loss_val = float(buf.item()) if isinstance(buf, torch.Tensor) else float(buf)
                if ev is not None and last_ev[0] is not None and s > last_ev[1]:
                    dt = last_ev[0].elapsed_time(ev) / 1000.0
                    sps = (s - last_ev[1]) / max(dt, 1e-9)
                elif last_ev[0] is None and ev is None and getattr(self, "_cpu_last", None):
                    t0, s0 = self._cpu_last
                    sps = (s - s0) / max(time.time() - t0, 1e-9)
                else:
                    sps = 0.0
                if ev is not None:
                    last_ev[0], last_ev[1] = ev, s
                else:
                    self._cpu_last = (time.time(), s)
                log("step %6d  total %.6f  lr %.2e  scale %d  sps %.1f" % (s, loss_val, lr_v, scl, sps))
                try:
                    with open(self.nuke_prefix + ".summary", "a", encoding="utf-8") as sf:
                        sf.write("%d,%g," % (s - 1, loss_val))
                except Exception:
                    pass
                if self.on_step:
                    self.on_step({"step": s, "loss/total": loss_val, "lr": lr_v, "stage_scale": scl, "sps": sps})
                if not math.isfinite(loss_val):
                    raise FloatingPointError("Training loss reached NaN value. Please try to restart training with different training settings.")

        t_start = time.time()
        try:
            for scale, end in stages:
                if step >= end or stop["flag"]:
                    continue
                h0 = min(f.shape[0] for f in self.src_full)
                w0 = min(f.shape[1] for f in self.src_full)
                sh, sw = data.stage_size(h0, w0, scale)
                crop = data.stage_crop_size(self.crop_size, scale, sh, sw)
                bs, lr_mul = self.batch_size, 1.0

                self._stage_peak = self._peak_bytes(bs, crop, self.amp_candidates[0]) if self.is_cuda else 0
                sampler.grid = max(1, 4 // scale)  # CopyCat: 4-px full-resolution crop grid
                src = self._make_source(scale)
                resident = isinstance(src, data.GPUCropSource)
                log("== stage: 1/%d resolution (crop %d, batch %d, data=%s) ==" % (scale, crop, bs, "vram" if resident else "host"))

                n_left = end - step
                chunk = 512
                plan_np = None
                plan_dev = None
                plan_pos = [chunk]

                def next_plan_rows():
                    nonlocal plan_np, plan_dev
                    if plan_pos[0] >= chunk:
                        plan_np = sampler.plan(chunk, bs, src.sizes, crop)
                        if resident:
                            plan_dev = torch.from_numpy(plan_np).to(self.device, non_blocking=False)
                        plan_pos[0] = 0
                    i = plan_pos[0]
                    plan_pos[0] += 1
                    return i

                if not resident:
                    def host_iter():
                        rem = n_left
                        while rem > 0:
                            k = min(chunk, rem)
                            pl = sampler.plan(k, bs, src.sizes, crop)
                            for r in pl:
                                yield r
                            rem -= k
                    src.start(host_iter(), bs, crop)

                def feed(engine):
                    if resident:
                        i = next_plan_rows()
                        src.gather(plan_dev, i, bs, crop, engine.x, engine.y)
                    else:
                        src.next_into(engine.x, engine.y)

                def make_engine(amp):
                    return StepEngine(self.G, self.optG, self.scaler, self.device, amp, bs, crop, self.c_in, self.c_out,
                                      channels_last=self.opt_channels_last, use_graph=self.opt_cuda_graph,
                                      warmup=3, lock=self.gpu_lock, multi_scale=True)

                def do_step(engine):
                    nonlocal step
                    step += 1
                    cur_lr = poly_lr(self.lr, step - 1, total, self.lr_power) * lr_mul
                    if self.lr_t is not None:
                        self.lr_t.fill_(cur_lr)
                    else:
                        for pg in self.optG.param_groups:
                            pg["lr"] = cur_lr
                    feed(engine)
                    loss = engine.step()
                    if step % log_iv == 0 or step == 1 or step == total:
                        if os.path.exists(stop_file):
                            stop["flag"] = True
                        if self.is_cuda:
                            buf = torch.empty((), dtype=torch.float32, pin_memory=True)
                            buf.copy_(loss, non_blocking=True)
                            ev = torch.cuda.Event(enable_timing=True)
                            ev.record()
                        else:
                            buf, ev = float(loss.item()), None
                        pending.append((step, cur_lr, scale, buf, ev))
                    emit_ready()
                    do_ck = step % ckpt_iv == 0 or step == total
                    do_pv = preview_on and (step % prev_iv == 0 or step == 1 or step == total)
                    if do_ck:
                        self._save_checkpoint(step)
                    if do_pv or do_ck:
                        self._preview(step, do_pv, do_ck, (engine.x, engine.y) if do_pv else None, scale=scale)

                # ---- precision choice ("Adaptive": keep the faster of fp16-AMP / fp32) ----
                engine = None
                timings = None
                cands = self.amp_candidates
                if len(cands) > 1 and n_left > 200:
                    timings = []
                    for amp in cands:
                        e = None
                        try:
                            e = make_engine(amp)
                            for _ in range(e.warmup + 1):
                                do_step(e)
                            torch.cuda.synchronize()
                            t0 = time.perf_counter()
                            for _ in range(25):
                                do_step(e)
                            torch.cuda.synchronize()
                            timings.append(((time.perf_counter() - t0) / 25, amp, e))
                        except torch.cuda.OutOfMemoryError:
                            if e is not None:
                                e.release()
                            e = None
                            import gc
                            gc.collect()
                            torch.cuda.empty_cache()
                            log("adaptive precision: %s does not fit in VRAM, skipped"
                                % ("fp16" if amp == torch.float16 else "fp32"))
                    if not timings:
                        raise RuntimeError(
                            "You have hit the maximum amount of memory that your compute device can use. "
                            "Please try reducing the crop or batch size.")
                    timings.sort(key=lambda z: z[0])
                    engine = timings[0][2]
                    for _, _, e in timings[1:]:
                        e.release()
                    log("adaptive precision: %s" % ", ".join(
                        "%s %.2f ms/step" % ("fp16" if a == torch.float16 else "fp32", t * 1e3) for t, a, _ in timings))
                else:
                    engine = make_engine(cands[0])

                try:
                    while step < end and not stop["flag"]:
                        do_step(engine)
                except torch.cuda.OutOfMemoryError:
                    raise RuntimeError(
                        "You have hit the maximum amount of memory that your compute device can use. "
                        "Please try reducing the crop or batch size.")
                engine.release()
                if not resident:
                    src.stop()
                del src, engine, plan_dev
                timings = None
                import gc
                gc.collect()  # captured graphs sit in reference cycles; free their private pools now
                if self.is_cuda:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

            emit_ready(block=True)
            if stop["flag"]:
                raise KeyboardInterrupt
            log("done. (%.1fs)" % (time.time() - t_start))
        except KeyboardInterrupt:
            emit_ready(block=True)
            log("interrupted at step %d, saving checkpoint..." % step)
            if step > self.step0:
                self._save_checkpoint(step)
            log("stopped.")
        finally:
            if os.path.exists(stop_file):
                try:
                    os.remove(stop_file)
                except OSError:
                    pass
            if self.preview_worker is not None:
                self.preview_worker.drain()
            if self.io_pool is not None:
                self.io_pool.shutdown(wait=True)
