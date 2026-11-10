# -*- coding: utf-8 -*-
"""Single training-step engine (eager or CUDA-Graph captured).

One step = forward (autocast) -> sum of per-pyramid-level MSE (CopyCat multi-scale supervision,
GT resized bilinear/align_corners to each level) -> scaled backward -> inf-check/unscale -> fused Adam
-> loss-scale update.  Every op is graph-safe (no host sync), so in the fast path the whole step
is captured once per stage shape and replayed with a single launch.  The math is identical to
the eager path (and to PyTorch's GradScaler + Adam): only kernel scheduling changes.
"""
import time

import torch
import torch.nn.functional as F


class GraphSafeScaler:
    """Port of torch.amp.GradScaler semantics (init 2**16, growth 2.0, backoff 0.5, interval 2000)
    using only device tensors, so it can live inside a CUDA graph."""

    def __init__(self, device, enabled, init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000):
        self.enabled = bool(enabled)
        self.device = device
        self.scale = torch.full((), float(init_scale), dtype=torch.float32, device=device)
        self.growth_tracker = torch.zeros((), dtype=torch.int32, device=device)
        self.found_inf = torch.zeros((), dtype=torch.float32, device=device)
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval

    def state_dict(self):
        return {
            "scale": float(self.scale.item()),
            "growth_tracker": int(self.growth_tracker.item()),
        }

    def load_state_dict(self, sd):
        if not sd:
            return
        self.scale.fill_(float(sd.get("scale", 65536.0)))
        self.growth_tracker.fill_(int(sd.get("growth_tracker", 0)))


class StepEngine:
    def __init__(self, net, opt, scaler, device, amp_dtype, bs, c, c_in, c_out,
                 channels_last=True, use_graph=True, warmup=3, lock=None, multi_scale=True):
        self.net = net
        self.opt = opt
        self.scaler = scaler
        self.device = device
        self.amp_dtype = amp_dtype
        self.use_scaler = scaler is not None and scaler.enabled and amp_dtype == torch.float16
        self.is_cuda = str(device).startswith("cuda")
        mf = torch.channels_last if (channels_last and self.is_cuda) else torch.contiguous_format
        self.x = torch.empty((bs, c_in, c, c), device=device, memory_format=mf).zero_()
        self.y = torch.empty((bs, c_out, c, c), device=device, memory_format=mf).zero_()
        self.loss = torch.zeros((), device=device)
        self.use_graph = bool(use_graph and self.is_cuda)
        self.warmup = int(warmup)
        self.lock = lock
        self.multi_scale = bool(multi_scale)
        self.graph = None
        self._eager_steps = 0
        self._params = [p for g in opt.param_groups for p in g["params"]]

    # ------------------------------------------------------------------ the step itself
    def _body(self):
        ac = dict(dtype=self.amp_dtype or torch.float32, enabled=self.amp_dtype is not None and self.is_cuda)
        with torch.autocast("cuda" if self.is_cuda else "cpu", **ac):
            outs = self.net.forward_all(self.x) if self.multi_scale else [self.net(self.x)]
        y = self.y.float()
        loss = None
        total = None
        for o in outs:
            o = o.float()
            g = y if o.shape[-2:] == y.shape[-2:] else F.interpolate(
                y, size=o.shape[-2:], mode="bilinear", align_corners=True)
            l = F.mse_loss(o, g, reduction="mean")
            total = l if total is None else total + l
            loss = l  # reported loss = finest level (what CopyCat prints)
        self.opt.zero_grad(set_to_none=True)
        if self.use_scaler:
            s = self.scaler
            (total * s.scale).backward()
            s.found_inf.zero_()
            inv = s.scale.double().reciprocal().float()
            grads = [p.grad for p in self._params if p.grad is not None]
            torch._amp_foreach_non_finite_check_and_unscale_(grads, s.found_inf, inv)
            if self.opt.defaults.get("fused"):
                self.opt.found_inf = s.found_inf  # fused kernel skips the update on overflow (graph-safe)
                self.opt.step()
            else:
                # eager / reference path: same semantics as torch GradScaler.step for foreach Adam
                self.opt.found_inf = None
                if float(s.found_inf.item()) == 0.0:
                    self.opt.step()
            torch._amp_update_scale_(s.scale, s.growth_tracker, s.found_inf,
                                     s.growth_factor, s.backoff_factor, s.growth_interval)
        else:
            total.backward()
            self.opt.found_inf = None
            self.opt.step()
        self.loss.copy_(loss.detach())

    def _capture(self):
        cur = torch.cuda.current_stream()
        side = torch.cuda.Stream()
        side.wait_stream(cur)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.stream(side):
            with torch.cuda.graph(g, stream=side, capture_error_mode="thread_local"):
                self._body()
        cur.wait_stream(side)
        self.graph = g

    def step(self):
        """Run one optimisation step on the current contents of self.x / self.y."""
        if self.graph is not None:
            self.graph.replay()
            return self.loss
        self._body()
        self._eager_steps += 1
        if self.use_graph and self._eager_steps >= self.warmup:
            try:
                if self.lock is not None:
                    with self.lock:
                        torch.cuda.synchronize()
                        self._capture()
                else:
                    torch.cuda.synchronize()
                    self._capture()
            except Exception as e:  # pragma: no cover - fall back to eager on any capture issue
                self.use_graph = False
                self.graph = None
                torch.cuda.synchronize()
                print("[engine] CUDA graph capture failed, staying eager: %s" % e, flush=True)
        return self.loss

    def release(self):
        self.graph = None


def bench_engine(engine, feed, n=12):
    """Time n steps (after warm-up/capture) of an engine; feed() refills engine.x/y each step.
    These are real training steps (they update the model), so nothing is wasted."""
    for _ in range(engine.warmup + 1):
        feed(engine)
        engine.step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        feed(engine)
        engine.step()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n, engine.warmup + 1 + n
