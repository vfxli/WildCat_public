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

from . import advanced
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
        self.io, self.tr, self.lo, self.mo = io, tr, lo, mo
        self.adv = advanced.parse_advanced_config(cfg)
        self.gpu_lock = threading.RLock()

        # ---------------- engine / lossless options ----------------
        self.engine = "reference" if opts.get("engine") == "reference" else "fast"
        fast = self.engine == "fast"
        self.opt_gpu_resident = lossless.get("gpu_resident", "auto" if fast else False)
        self.opt_cuda_graph = _bool(lossless.get("cuda_graph", fast))
        self.opt_fused_adam = _bool(lossless.get("fused_adam", fast))
        if not self.opt_fused_adam:
            self.opt_cuda_graph = False
        self.opt_channels_last = _bool(lossless.get("channels_last", fast))
        self.opt_benchmark = _bool(lossless.get("cudnn_benchmark", fast))
        self.opt_async_io = _bool(lossless.get("async_io", fast))
        self.preview_gpu = str(os.environ.get("WILDCAT_PREVIEW_GPU") or lossless.get("preview_gpu", "auto"))

        # ---------------- mixed precision (CopyCat: Adaptive / Disabled / Force Enabled) ----------------
        prec = tr.get("precision") or "adaptive"
        if prec not in ("adaptive", "fp32", "fp16"):
            prec = "adaptive"
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
            col.get("apply_ocio_view", False)
            or (
                self.working_colorspace != "Raw"
                and (self.is_exr_sequence or self.source_colorspace != "Raw")
            )
        )
        if len(self.src_full) != len(self.tgt_full):
            raise ValueError("Input (%d) and Ground Truth (%d) frame counts do not match" % (len(self.src_full), len(self.tgt_full)))
        for a, b, p in zip(self.src_full, self.tgt_full, tgt_files):
            if a.shape[:2] != b.shape[:2]:
                raise ValueError("Input/Ground Truth size mismatch at %s: %s vs %s" % (p, a.shape[:2], b.shape[:2]))
        self.pair_names = [os.path.splitext(os.path.basename(p))[0] for p in tgt_files]
        n_frames = len(self.src_full)

        # ---------------- crop / batch ----------------
        self.crop_size = int(tr.get("crop_size") or 256)
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
        ck = None
        if init_w and init_w != "None":
            ck = models.load_weights_into_model(self.raw_G, init_w, model_size=model_size, device=self.device)
            if isinstance(ck, dict) and tr.get("resume", False):
                if isinstance(ck.get("G_raw"), dict):
                    self.raw_G.load_state_dict(ck["G_raw"], strict=True)
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
        self._val_G = None

        # ---------------- optional advanced state (100% off by default) ----------------
        if self.adv["ema_lr"]["enabled"]:
            self.ema_param_names = [name for name, _ in self.raw_G.named_parameters()]
            if isinstance(ck, dict) and tr.get("resume", False) and isinstance(ck.get("G_ema"), dict):
                self.ema_params = [
                    ck["G_ema"][name].to(device=self.device, dtype=p.dtype).detach().clone()
                    for name, p in self.raw_G.named_parameters()
                ]
            else:
                self.ema_params = [p.detach().clone() for p in self.raw_G.parameters()]
            self.ema_alpha_t = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        else:
            self.ema_param_names = None
            self.ema_params = None
            self.ema_alpha_t = None

        if self.adv["augment"]["enabled"]:
            self.batch_augmenter = advanced.BatchAugmenter(self.adv["augment"], self.c_out, self.device, seed=self.seed)
        else:
            self.batch_augmenter = None

        # CopyCat-style artefacts in the run directory: Training_YYMMDD_HHMMSS.<step>.cat/.png + .summary
        self.nuke_prefix = os.path.join(self.run_dir, "Training_%s" % time.strftime("%y%m%d_%H%M%S"))
        if tr.get("resume", False):
            import glob as _glob
            prev = sorted(_glob.glob(os.path.join(self.run_dir, "Training_*.summary")))
            if prev:
                self.nuke_prefix = prev[-1][: -len(".summary")]  # continue the same CopyCat-style run name
        self.n_frames = n_frames
        self.initial_epochs = self.epochs
        self.stop_after_optimal_epochs = max(0, int(tr.get("stop_after_optimal_epochs") or 0))
        self._io_futures = []
        self._write_run_info()
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
        if self.adv["any_enabled"]:
            on_keys = [k for k in advanced.ADV_SECTION_KEYS if self.adv[k]["enabled"]]
            log("WildCat | advanced=[%s]" % ", ".join(on_keys))
        if self.stop_after_optimal_epochs > 0:
            log("WildCat | optimal_patience=+%d epochs after 1/1 peak" % self.stop_after_optimal_epochs)

    # ======================================================================== helpers
    def _write_run_info(self, extra=None):
        try:
            import json as _json

            payload = {
                "total_steps": self.total_steps,
                "epochs": self.epochs,
                "initial_epochs": getattr(self, "initial_epochs", self.epochs),
                "batch_size": self.batch_size,
                "crop_size": self.crop_size,
                "frames": getattr(self, "n_frames", len(self.src_full)),
                "nuke_prefix": os.path.basename(self.nuke_prefix),
            }
            if isinstance(extra, dict):
                payload.update(extra)
            info_p = os.path.join(self.run_dir, "run_info.json")
            tmp_p = "%s.tmp.%d" % (info_p, os.getpid())
            with open(tmp_p, "w", encoding="utf-8") as f:
                _json.dump(payload, f)
            os.replace(tmp_p, info_p)
        except Exception:
            pass

    def _flush_io_pool(self):
        if not getattr(self, "_io_futures", None):
            return
        for fut in list(self._io_futures):
            try:
                fut.result()
            except Exception:
                pass
        self._io_futures = []

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
        has_tc = torch.cuda.get_device_capability(0)[0] >= 7
        amp = (
            (torch.float16 if (has_tc and torch.float16 in self.amp_candidates) else self.amp_candidates[-1])
            if self.prec == "adaptive"
            else self.amp_candidates[0]
        )
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

    def _select_adaptive_amp(self, bs, crop, cands, n_left):
        """Side-effect-free 'Adaptive' AMP micro-benchmark on an isolated scratch model/optimizer.
        Never mutates self.raw_G, self.optG, self.scaler, self.ema_params, sampler, step, or RNG."""
        if len(cands) <= 1 or not self.is_cuda:
            return cands[0]
        has_tc = torch.cuda.get_device_capability(0)[0] >= 7
        if n_left < 1000:
            return torch.float16 if (has_tc and torch.float16 in cands) else cands[0]

        import gc

        lf_cfg = self.adv["loss_family"] if self.adv["loss_family"]["enabled"] else None
        timings = []
        gpu_warmed = False
        with self.gpu_lock, torch.random.fork_rng(devices=[0]):
            for amp in cands:
                tmp_G = tmp_opt = tmp_scaler = tmp_ema = tmp_alpha = tmp_e = None
                try:
                    tmp_G, _ = models.build_generator(self.mo, cudnn_pad=True)
                    tmp_G = tmp_G.to(self.device)
                    if self.opt_channels_last:
                        tmp_G = tmp_G.to(memory_format=torch.channels_last)
                    t_params = list(tmp_G.parameters())
                    if self.opt_fused_adam:
                        t_lr = torch.tensor(self.lr, dtype=torch.float32, device=self.device)
                        tmp_opt = torch.optim.Adam(
                            t_params, lr=t_lr, betas=(0.9, 0.999), eps=1e-8, fused=True, capturable=True
                        )
                    else:
                        tmp_opt = torch.optim.Adam(t_params, lr=self.lr, betas=(0.9, 0.999), eps=1e-8)
                    tmp_scaler = GraphSafeScaler(self.device, enabled=True)
                    if self.ema_params is not None:
                        tmp_ema = [p.detach().clone() for p in t_params]
                        tmp_alpha = torch.tensor(0.001, dtype=torch.float32, device=self.device)
                    tmp_e = StepEngine(
                        tmp_G, tmp_opt, tmp_scaler, self.device, amp, bs, crop, self.c_in, self.c_out,
                        channels_last=self.opt_channels_last, use_graph=self.opt_cuda_graph,
                        warmup=3, lock=self.gpu_lock, multi_scale=True,
                        loss_family_cfg=lf_cfg, ema_params=tmp_ema, ema_alpha_t=tmp_alpha,
                    )
                    tmp_e.x.fill_(0.25)
                    tmp_e.y.fill_(0.20)
                    # Ensure graph capture + >=220ms continuous load on first candidate so GPU exits P8/P5 low-clock state
                    min_warmup = max(tmp_e.warmup + 6, 10)
                    for _ in range(min_warmup):
                        tmp_e.step()
                    if not gpu_warmed:
                        torch.cuda.synchronize()
                        t_wake = time.perf_counter()
                        while time.perf_counter() - t_wake < 0.22:
                            for _ in range(6):
                                tmp_e.step()
                            torch.cuda.synchronize()
                        gpu_warmed = True
                    torch.cuda.synchronize()
                    bench_iters = 30
                    t0 = time.perf_counter()
                    for _ in range(bench_iters):
                        tmp_e.step()
                    torch.cuda.synchronize()
                    timings.append(((time.perf_counter() - t0) / bench_iters, amp))
                except torch.cuda.OutOfMemoryError:
                    log(
                        "adaptive precision: %s does not fit in VRAM, skipped"
                        % ("fp16" if amp == torch.float16 else "fp32")
                    )
                finally:
                    if tmp_e is not None:
                        tmp_e.release()
                    del tmp_e, tmp_opt, tmp_scaler, tmp_ema, tmp_alpha, tmp_G
                    gc.collect()
                    torch.cuda.empty_cache()

        if not timings:
            raise RuntimeError(
                "You have hit the maximum amount of memory that your compute device can use. "
                "Please try reducing the crop or batch size."
            )
        timings.sort(key=lambda z: z[0])
        chosen = timings[0][1]
        by_amp = {a: t for t, a in timings}
        if has_tc and torch.float16 in by_amp and None in by_amp:
            if by_amp[torch.float16] <= by_amp[None] * 1.05:
                chosen = torch.float16
        chosen_name = "fp16" if chosen == torch.float16 else "fp32"
        log(
            "adaptive precision: %s -> selected %s"
            % (
                ", ".join(
                    "%s %.2f ms/step" % ("fp16" if a == torch.float16 else "fp32", t * 1e3)
                    for t, a in timings
                ),
                chosen_name,
            )
        )
        return chosen

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

    def eval_state_dict(self, cpu=False):
        with torch.no_grad():
            raw_sd = self.raw_G.state_dict()
            if self.ema_params is None:
                if cpu:
                    return {k: v.detach().cpu().contiguous() for k, v in raw_sd.items()}
                return {k: v.detach().clone() for k, v in raw_sd.items()}
            ema_map = dict(zip(self.ema_param_names, self.ema_params))
            out = {}
            for k, v in raw_sd.items():
                src_t = ema_map.get(k, v)
                out[k] = src_t.detach().cpu().contiguous() if cpu else src_t.detach().clone()
            return out

    def _save_checkpoint(self, step):
        if self.is_cuda:
            torch.cuda.synchronize()
        sd_cpu = self.eval_state_dict(cpu=True)
        ck = {
            "G": sd_cpu,
            "step": step,
            "config": self.cfg,
            "optG": self._opt_state_cpu(),
            "scaler": self.scaler.state_dict() if self.is_cuda else {},
        }
        if self.ema_params is not None:
            ck["G_raw"] = {k: v.detach().cpu().contiguous() for k, v in self.raw_G.state_dict().items()}
            ck["G_ema"] = sd_cpu
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
            self._io_futures = [f for f in self._io_futures if not f.done()]
            self._io_futures.append(self.io_pool.submit(_write))
        else:
            _write()

    # ======================================================================== preview / val
    def _snapshot(self):
        snap = self.eval_state_dict(cpu=False)
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
            amp = None  # Match Nuke Inference default (halfPrecision=False, FP32) on both 1-GPU and 2-GPU setups
            self.preview_worker = PreviewWorker(self, self.preview_device, amp_dtype=amp)
        if not do_val and (self.preview_worker.busy.is_set() or self.preview_worker.q.full()):
            return  # skip this preview rather than stall training
        snap, ev = self._snapshot()
        if crops is not None:
            crops = (crops[0][:3].detach().clone(), crops[1][:3].detach().clone(), int(scale))
        self.preview_worker.submit(step, snap, ev, do_preview, do_val, crops)

    @torch.no_grad()
    def validate(self, step):
        if self.ema_params is not None:
            if self._val_G is None:
                vg, _ = models.build_generator(self.mo, cudnn_pad=True)
                self._val_G = vg.to(self.device).eval()
            self._val_G.load_state_dict(self.eval_state_dict(cpu=False), strict=True)
            eval_net = self._val_G
        else:
            self.raw_G.eval()
            eval_net = self.raw_G
        parts = []
        for i, (a, b) in enumerate(zip(self.src_full, self.tgt_full)):
            out = forward_full(eval_net, a, self.device, None, self.opt_channels_last)
            gt = torch.from_numpy(np.asarray(b, dtype=np.float32)).permute(2, 0, 1)[None].to(self.device)
            parts.append("f%d %.6f" % (i + 1, torch.mean((out - gt) ** 2).item()))
        log("@step %d val(MSE): %s" % (step, "  ".join(parts)))
        if eval_net is self.raw_G:
            self.raw_G.train()

    # ======================================================================== training
    def _stage_list(self):
        total = self.total_steps
        if not self.adv["stage_schedule"]["enabled"]:
            if self.tr.get("multi_resolution", True):
                s = math.ceil(total / 3.0)
                return [(4, min(total, s)), (2, min(total, 2 * s)), (1, total)]
            return [(1, total)]
        return advanced.compute_stage_list(
            total, bool(self.tr.get("multi_resolution", True)), self.adv["stage_schedule"]
        )

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

        from . import recommender

        init_epochs = int(getattr(self, "initial_epochs", self.epochs) or self.epochs)
        init_total = int(total)
        stage_1_start = 0
        for _s_idx, (_sc_v, _) in enumerate(stages):
            if _sc_v == 1:
                stage_1_start = 0 if _s_idx == 0 else stages[_s_idx - 1][1]
                break
        patience_steps = (
            max(2, math.ceil(self.stop_after_optimal_epochs * len(self.src_full) / float(self.batch_size)))
            if self.stop_after_optimal_epochs > 0
            else 0
        )
        stage_1_len = max(1, total - stage_1_start)
        settle_steps = (
            min(max(2, patience_steps // 3), max(2, min(200, stage_1_len // 10)))
            if patience_steps > 0
            else 0
        )
        opt_check_iv = (
            max(2, min(ckpt_iv, patience_steps // 3 if patience_steps < 30 else max(10, patience_steps // 4)))
            if patience_steps > 0
            else ckpt_iv
        )
        opt_tracker = {
            "best_step": 0,
            "best_score": -1e9,
            "triggered": False,
            "stopped_step": 0,
            "live_rec": None,
        }

        def get_live_rec():
            if opt_tracker["live_rec"] is None:
                lr_inst = recommender.CheckpointRecommender(
                    self.run_dir,
                    self.cfg,
                    device=self.device,
                    src_full=self.src_full,
                    tgt_full=self.tgt_full,
                    max_seq_frames=18,
                )
                if self.step0 > stage_1_start + settle_steps:
                    ck_map = lr_inst.discover_checkpoint_paths()
                    for prev_st in sorted(s for s in ck_map if stage_1_start + settle_steps <= s <= self.step0):
                        m_prev = lr_inst.evaluate_step(prev_st, ckpt_path=ck_map[prev_st])
                        sc_prev = lr_inst.live_score_step(m_prev)
                        if opt_tracker["best_step"] <= 0 or sc_prev > opt_tracker["best_score"] + 0.015:
                            opt_tracker["best_score"] = sc_prev
                            opt_tracker["best_step"] = prev_st
                opt_tracker["live_rec"] = lr_inst
            return opt_tracker["live_rec"]

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

        def run_stage_segment(scale, end, stage_start_step, is_stage_transition, ext_mode=False, ext_start_step=0):
            nonlocal step, total
            if step >= end or stop["flag"]:
                return
            h0 = min(f.shape[0] for f in self.src_full)
            w0 = min(f.shape[1] for f in self.src_full)
            sh, sw = data.stage_size(h0, w0, scale)
            crop = data.stage_crop_size(self.crop_size, scale, sh, sw)
            bs, lr_mul = self.batch_size, 1.0

            n_left = end - step
            short_stage = n_left < 1000
            bench_prev = torch.backends.cudnn.benchmark
            if short_stage and self.is_cuda:
                torch.backends.cudnn.benchmark = False
            chosen_amp = self._select_adaptive_amp(bs, crop, self.amp_candidates, n_left)
            self._stage_peak = self._peak_bytes(bs, crop, chosen_amp) if self.is_cuda else 0
            if self.adv["crop_grid"]["enabled"]:
                sampler.grid = max(1, int(self.adv["crop_grid"]["full_res_grid"]) // scale)
            else:
                sampler.grid = max(1, 4 // scale)  # CopyCat: 4-px full-resolution crop grid
            src = self._make_source(scale)
            resident = isinstance(src, data.GPUCropSource)
            if ext_mode:
                log(
                    "== auto-extend: 1/%d resolution (steps %d -> %d, crop %d, batch %d, data=%s) =="
                    % (scale, step, end, crop, bs, "vram" if resident else "host")
                )
            else:
                log("== stage: 1/%d resolution (crop %d, batch %d, data=%s) ==" % (scale, crop, bs, "vram" if resident else "host"))

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
                if self.batch_augmenter is not None:
                    self.batch_augmenter.apply_(engine.x, engine.y)

            def make_engine(amp):
                lf_cfg = self.adv["loss_family"] if self.adv["loss_family"]["enabled"] else None
                return StepEngine(
                    self.G, self.optG, self.scaler, self.device, amp, bs, crop, self.c_in, self.c_out,
                    channels_last=self.opt_channels_last, use_graph=(self.opt_cuda_graph and not short_stage),
                    warmup=3, lock=self.gpu_lock, multi_scale=True,
                    loss_family_cfg=lf_cfg,
                    ema_params=self.ema_params, ema_alpha_t=self.ema_alpha_t,
                )

            ext_len = max(1, end - ext_start_step) if ext_mode else 1
            ext_mid_ckpt = (ext_start_step + ext_len // 2) if (ext_mode and ext_len >= 20 and ckpt_iv > ext_len // 2) else -1
            ext_lr_floor = poly_lr(self.lr, max(0, ext_start_step - 2), max(1, ext_start_step), self.lr_power) * lr_mul
            ext_lr_peak = self.lr * 0.35 * lr_mul

            def do_step(engine):
                nonlocal step
                step += 1
                if ext_mode:
                    k_ext = max(1, step - ext_start_step)
                    ramp = min(50, max(4, ext_len // 5))
                    if k_ext <= ramp:
                        cur_lr = ext_lr_floor + (ext_lr_peak - ext_lr_floor) * (k_ext / float(ramp))
                    else:
                        prog = (k_ext - ramp) / float(max(1, ext_len - ramp))
                        cur_lr = ext_lr_peak * ((1.0 - 0.72 * min(1.0, prog)) ** 0.5)
                elif not self.adv["stage_schedule"]["enabled"]:
                    cur_lr = poly_lr(self.lr, step - 1, total, self.lr_power) * lr_mul
                else:
                    cur_lr = advanced.compute_scheduled_lr(
                        self.lr, step - 1, total, self.lr_power,
                        self.adv["stage_schedule"],
                        stage_start_step_zero_based=stage_start_step,
                        is_stage_transition=is_stage_transition,
                    ) * lr_mul
                if self.lr_t is not None:
                    self.lr_t.fill_(cur_lr)
                else:
                    for pg in self.optG.param_groups:
                        pg["lr"] = cur_lr
                if self.ema_alpha_t is not None:
                    k_ema = step if ext_mode else max(1, step - stage_start_step)
                    self.ema_alpha_t.fill_(advanced.compute_ema_alpha(self.adv["ema_lr"]["ema_decay"], k_ema))
                feed(engine)
                loss = engine.step()
                stop_iv = min(log_iv, 10)
                if (step % stop_iv == 0 or step == total) and os.path.exists(stop_file):
                    stop["flag"] = True
                do_ck = step % ckpt_iv == 0 or step == total or step == ext_mid_ckpt
                do_opt_chk = (
                    scale == 1
                    and patience_steps > 0
                    and not stop["flag"]
                    and step >= stage_1_start + settle_steps
                    and (do_ck or step % opt_check_iv == 0 or step == end)
                )
                if step % log_iv == 0 or step == 1 or step == total or do_ck or do_opt_chk or stop["flag"]:
                    if self.is_cuda:
                        buf = torch.empty((), dtype=torch.float32, pin_memory=True)
                        buf.copy_(loss, non_blocking=True)
                        ev = torch.cuda.Event(enable_timing=True)
                        ev.record()
                    else:
                        buf, ev = float(loss.item()), None
                    pending.append((step, cur_lr, scale, buf, ev))
                emit_ready()
                do_pv = preview_on and (step % prev_iv == 0 or step == 1 or step == total)
                if do_ck:
                    self._save_checkpoint(step)
                if do_pv or do_ck:
                    self._preview(step, do_pv, do_ck, (engine.x, engine.y) if do_pv else None, scale=scale)
                if do_opt_chk:
                    rec_live = get_live_rec()
                    m_live = rec_live.evaluate_step(step, state_dict=self.eval_state_dict(cpu=True))
                    sc_live = rec_live.live_score_step(m_live)
                    if opt_tracker["best_step"] <= 0 or sc_live > opt_tracker["best_score"] + 0.015:
                        opt_tracker["best_score"] = sc_live
                        opt_tracker["best_step"] = step
                        if not do_ck:
                            self._save_checkpoint(step)
                    elif step - opt_tracker["best_step"] >= patience_steps:
                        if not do_ck:
                            self._save_checkpoint(step)
                        opt_tracker["triggered"] = True
                        opt_tracker["stopped_step"] = step

            engine = None
            try:
                engine = make_engine(chosen_amp)
                while step < end and not stop["flag"] and not opt_tracker["triggered"]:
                    do_step(engine)
            except torch.cuda.OutOfMemoryError:
                raise RuntimeError(
                    "You have hit the maximum amount of memory that your compute device can use. "
                    "Please try reducing the crop or batch size.")
            finally:
                if short_stage and self.is_cuda:
                    torch.backends.cudnn.benchmark = bench_prev
            if engine is not None:
                engine.release()
            if not resident:
                src.stop()
            del src, engine, plan_dev
            import gc
            gc.collect()  # captured graphs sit in reference cycles; free their private pools now
            if self.is_cuda:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        t_start = time.time()
        try:
            for stage_idx, (scale, end) in enumerate(stages):
                if step >= end or stop["flag"] or opt_tracker["triggered"]:
                    continue
                stage_start_step = 0 if stage_idx == 0 else stages[stage_idx - 1][1]
                is_stage_transition = stage_idx > 0
                run_stage_segment(scale, end, stage_start_step, is_stage_transition, ext_mode=False)

            emit_ready(block=True)
            if stop["flag"]:
                raise KeyboardInterrupt

            if opt_tracker["live_rec"] is not None:
                opt_tracker["live_rec"].release()
                opt_tracker["live_rec"] = None

            # ── Post-Budget Validation-Sequence Optimal Recommender, Patience Early-Stop & Auto-Extend ──
            seq_files = recommender.resolve_eval_sequence_files(self.cfg)
            auto_ext_on = _bool(tr.get("auto_extend_optimal", False))
            max_ep_cap = int(tr.get("max_auto_extend_epochs") or 0)
            ext_round = 0

            def _finalize_patience_stop():
                nonlocal total
                final_ep = max(1, int(round(step * self.batch_size / float(max(1, len(self.src_full))))))
                best_st = int(opt_tracker["best_step"] or step)
                best_ep = max(1, int(round(best_st * self.batch_size / float(max(1, len(self.src_full))))))
                elapsed_st = max(0, step - best_st)
                self.epochs = final_ep
                self.total_steps = int(step)
                total = int(step)
                self._write_run_info(
                    {
                        "initial_epochs": init_epochs,
                        "initial_total_steps": init_total,
                        "auto_extended": ext_round > 0,
                        "auto_extended_rounds": ext_round,
                        "stopped_after_optimal": True,
                        "stop_after_optimal_epochs": self.stop_after_optimal_epochs,
                        "optimal_step": best_st,
                        "optimal_epoch": best_ep,
                    }
                )
                log(
                    f"[Optimal-Stop] Best model locked at Step {best_st} (~Epoch {best_ep}, score={opt_tracker['best_score']:.2f}); "
                    f"+{self.stop_after_optimal_epochs} epochs ({elapsed_st} steps) elapsed without further gain -> stopping cleanly."
                )

            if opt_tracker["triggered"]:
                self._flush_io_pool()
                _finalize_patience_stop()
                if len(seq_files) >= 2:
                    rec_engine = recommender.CheckpointRecommender(
                        self.run_dir,
                        self.cfg,
                        device=self.device,
                        src_full=self.src_full,
                        tgt_full=self.tgt_full,
                    )
                    try:
                        summary = rec_engine.evaluate_and_summarize(
                            current_step=step,
                            current_state_dict=self.eval_state_dict(cpu=True),
                            epochs=self.epochs,
                            total_steps=total,
                            initial_epochs=init_epochs,
                            initial_total_steps=init_total,
                            auto_extended_rounds=ext_round,
                            stop_reason=f"stopped_after_optimal_+{self.stop_after_optimal_epochs}ep",
                            stopped_after_optimal=True,
                            stop_after_optimal_epochs=self.stop_after_optimal_epochs,
                            optimal_step=int(opt_tracker["best_step"] or step),
                            persist=True,
                        )
                        picks = summary.get("picks") or {}
                        if picks:
                            b_p = picks.get("balanced") or {}
                            f_p = picks.get("flicker") or {}
                            d_p = picks.get("detail") or {}
                            hfi_str = f"{f_p['seq_hfi']:.5f}" if f_p.get("seq_hfi") is not None else "n/a"
                            log(
                                f"[Optimal-Eval] Evaluated {int(summary.get('evaluated_checkpoint_count') or 0)} ckpts "
                                f"on {int(summary.get('seq_evaluated_frames') or 0)} validation frames "
                                f"({float(summary.get('elapsed_sec') or 0.0):.1f}s) -> "
                                f"Balanced: Step {int(b_p.get('step') or step)} | "
                                f"Low-Flicker: Step {int(f_p.get('step') or step)} (HFI={hfi_str}) | "
                                f"Peak-Detail: Step {int(d_p.get('step') or step)} (ROI={float(d_p.get('roi_psnr') or 0.0):.2f}dB)"
                            )
                    finally:
                        rec_engine.release()
            elif len(seq_files) >= 2 or (patience_steps > 0 and auto_ext_on):
                self._flush_io_pool()
                rec_engine = recommender.CheckpointRecommender(
                    self.run_dir,
                    self.cfg,
                    device=self.device,
                    src_full=self.src_full,
                    tgt_full=self.tgt_full,
                )
                try:
                    while True:
                        summary = (
                            rec_engine.evaluate_and_summarize(
                                current_step=step,
                                current_state_dict=self.eval_state_dict(cpu=True),
                                epochs=self.epochs,
                                total_steps=total,
                                initial_epochs=init_epochs,
                                initial_total_steps=init_total,
                                auto_extended_rounds=ext_round,
                                optimal_step=int(opt_tracker["best_step"] or 0),
                                stop_after_optimal_epochs=self.stop_after_optimal_epochs,
                                persist=True,
                            )
                            if len(seq_files) >= 2
                            else {}
                        )
                        picks = summary.get("picks") or {}
                        if picks:
                            b_p = picks.get("balanced") or {}
                            f_p = picks.get("flicker") or {}
                            d_p = picks.get("detail") or {}
                            hfi_str = f"{f_p['seq_hfi']:.5f}" if f_p.get("seq_hfi") is not None else "n/a"
                            log(
                                f"[Optimal-Eval] Evaluated {int(summary.get('evaluated_checkpoint_count') or 0)} ckpts "
                                f"on {int(summary.get('seq_evaluated_frames') or 0)} validation frames "
                                f"({float(summary.get('elapsed_sec') or 0.0):.1f}s) -> "
                                f"Balanced: Step {int(b_p.get('step') or step)} | "
                                f"Low-Flicker: Step {int(f_p.get('step') or step)} (HFI={hfi_str}) | "
                                f"Peak-Detail: Step {int(d_p.get('step') or step)} (ROI={float(d_p.get('roi_psnr') or 0.0):.2f}dB)"
                            )
                        if not auto_ext_on or stop["flag"] or os.path.exists(stop_file):
                            break
                        should_ext, reason, next_epochs = rec_engine.decide_auto_extend(
                            summary,
                            initial_epochs=init_epochs,
                            initial_total_steps=init_total,
                            current_epochs=self.epochs,
                            current_total_steps=total,
                            auto_extend_rounds=ext_round,
                            max_epochs_cap=max_ep_cap,
                            stop_after_optimal_epochs=self.stop_after_optimal_epochs,
                            best_opt_step=int(opt_tracker["best_step"] or 0),
                        )
                        if not should_ext:
                            if len(seq_files) >= 2:
                                rec_engine.evaluate_and_summarize(
                                    current_step=step,
                                    epochs=self.epochs,
                                    total_steps=total,
                                    initial_epochs=init_epochs,
                                    initial_total_steps=init_total,
                                    auto_extended_rounds=ext_round,
                                    stop_reason=reason,
                                    stop_after_optimal_epochs=self.stop_after_optimal_epochs,
                                    optimal_step=int(opt_tracker["best_step"] or 0),
                                    persist=True,
                                )
                            log("[Optimal-Eval] Optimal checkpoint confirmed (%s)." % reason)
                            break

                        new_total = max(
                            total + 1,
                            math.ceil(next_epochs * len(self.src_full) / float(self.batch_size)),
                        )
                        ext_round += 1
                        ext_start = step
                        self.epochs = int(next_epochs)
                        self.total_steps = int(new_total)
                        total = int(new_total)
                        self._write_run_info(
                            {
                                "initial_epochs": init_epochs,
                                "initial_total_steps": init_total,
                                "auto_extended": True,
                                "auto_extended_rounds": ext_round,
                            }
                        )
                        log(
                            "[Auto-Extend #%d] Validation metrics still improving (%s) -> continuing to %d epochs (step %d)..."
                            % (ext_round, reason, self.epochs, total)
                        )
                        run_stage_segment(
                            scale=1,
                            end=total,
                            stage_start_step=ext_start,
                            is_stage_transition=False,
                            ext_mode=True,
                            ext_start_step=ext_start,
                        )
                        emit_ready(block=True)
                        self._flush_io_pool()
                        if opt_tracker["live_rec"] is not None:
                            opt_tracker["live_rec"].release()
                            opt_tracker["live_rec"] = None
                        if stop["flag"]:
                            if len(seq_files) >= 2:
                                rec_engine.evaluate_and_summarize(
                                    current_step=step,
                                    epochs=self.epochs,
                                    total_steps=total,
                                    initial_epochs=init_epochs,
                                    initial_total_steps=init_total,
                                    auto_extended_rounds=ext_round,
                                    stop_reason="user_stopped",
                                    persist=True,
                                )
                            raise KeyboardInterrupt
                        if opt_tracker["triggered"]:
                            _finalize_patience_stop()
                            if len(seq_files) >= 2:
                                rec_engine.evaluate_and_summarize(
                                    current_step=step,
                                    current_state_dict=self.eval_state_dict(cpu=True),
                                    epochs=self.epochs,
                                    total_steps=total,
                                    initial_epochs=init_epochs,
                                    initial_total_steps=init_total,
                                    auto_extended_rounds=ext_round,
                                    stop_reason=f"stopped_after_optimal_+{self.stop_after_optimal_epochs}ep",
                                    stopped_after_optimal=True,
                                    stop_after_optimal_epochs=self.stop_after_optimal_epochs,
                                    optimal_step=int(opt_tracker["best_step"] or step),
                                    persist=True,
                                )
                            break
                finally:
                    rec_engine.release()

            log("done. (%.1fs)" % (time.time() - t_start))
        except KeyboardInterrupt:
            emit_ready(block=True)
            log("interrupted at step %d, saving checkpoint..." % step)
            if step > self.step0:
                self._save_checkpoint(step)
            log("stopped.")
        finally:
            if opt_tracker.get("live_rec") is not None:
                try:
                    opt_tracker["live_rec"].release()
                except Exception:
                    pass
            if os.path.exists(stop_file):
                try:
                    os.remove(stop_file)
                except OSError:
                    pass
            if self.preview_worker is not None:
                self.preview_worker.drain()
            if self.io_pool is not None:
                self.io_pool.shutdown(wait=True)
