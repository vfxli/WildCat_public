# -*- coding: utf-8 -*-
"""Multi-User Task Queue & GPU Scheduler for WildCat Core Compute Engine."""
import collections
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback

from . import analytics_db, db, gpu_speed_db, i18n, security, storage, telemetry

ROOT = db.ROOT


def _resolve_python_bin() -> str:
    if sys.executable and os.path.isfile(sys.executable):
        return sys.executable
    for cand in (
        os.path.join(ROOT, ".venv", "Scripts", "python.exe"),
        os.path.join(ROOT, "runtime", "python", "python.exe"),
        os.path.join(ROOT, "runtime", "python", "cpython-3.12", "python.exe"),
        os.path.join(ROOT, ".venv", "bin", "python"),
        os.path.join(ROOT, ".venv", "cpython-3.12", "bin", "python3.12"),
        os.path.join(ROOT, "runtime", "python", "cpython-3.12", "bin", "python3.12"),
    ):
        if os.path.isfile(cand):
            return cand
    return "python" if os.name == "nt" else "python3"


PYTHON_BIN = _resolve_python_bin()


def _is_pid_alive(pid) -> bool:
    """Cross-platform check if a process PID is currently alive (never calls os.kill(pid, 0) on Windows)."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                return False
            try:
                exit_code = ctypes.c_ulong()
                if ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                    return exit_code.value == STILL_ACTIVE
                return False
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _detect_gpus_meta():
    """Return list of (gpu_id_str, label_str, vram_mb_int). Strictly detects real physical GPUs."""
    smi_candidates = []
    which_smi = shutil.which("nvidia-smi")
    if which_smi:
        smi_candidates.append(which_smi)
    smi_candidates.extend(
        [
            "/usr/bin/nvidia-smi",
            "/usr/local/nvidia/bin/nvidia-smi",
            r"C:\Windows\System32\nvidia-smi.exe",
            r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
        ]
    )
    seen = set()
    for smi in smi_candidates:
        if not smi or smi in seen:
            continue
        seen.add(smi)
        if not os.path.isfile(smi):
            continue
        try:
            res = subprocess.run(
                [smi, "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
            )
            if res.returncode != 0 or not res.stdout:
                continue
            gpus = []
            for line in res.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2 and parts[0].isdigit():
                    idx, name = parts[0], parts[1]
                    mem = parts[2] if len(parts) >= 3 else ""
                    m_num = re.search(r"(\d+)", mem)
                    vram_mb = int(m_num.group(1)) if m_num else 0
                    if "gib" in mem.lower() or ("gb" in mem.lower() and vram_mb < 512):
                        vram_mb *= 1024
                    label = f"CUDA:{idx} — {name}" + (f" ({mem})" if mem else "")
                    gpus.append((idx, label.strip(), vram_mb))
            if gpus:
                return gpus
        except Exception:
            pass

    # Fallback: query PyTorch CUDA runtime directly
    try:
        import torch

        if torch.cuda.is_available():
            cnt = int(torch.cuda.device_count())
            if cnt > 0:
                gpus = []
                for i in range(cnt):
                    props = torch.cuda.get_device_properties(i)
                    vram_mb = int(props.total_memory // (1024 * 1024))
                    vram_gb = round(props.total_memory / (1024**3))
                    gpus.append((str(i), f"CUDA:{i} — {props.name} ({vram_gb} GB)", vram_mb))
                return gpus
    except Exception:
        pass

    # Final fallback: strictly 1 slot (never invent fake multi-GPU slots on a single-GPU host)
    return [("0", "CUDA:0", 0)]


def detect_gpus():
    """Return list of (gpu_id_str, label_str)."""
    return [(gid, glabel) for gid, glabel, _ in _detect_gpus_meta()]


_GPUS_META = _detect_gpus_meta()
GPUS = [(gid, glabel) for gid, glabel, _ in _GPUS_META]
_GPU_VRAM_MB = {gid: vram for gid, _, vram in _GPUS_META}
_GPU_LABEL_MAP = {str(gid): str(glabel) for gid, glabel, _ in _GPUS_META}


def _gpu_hw_rank_key(gid: str) -> tuple[int, int]:
    """Default hardware preference: highest VRAM first, then lowest device index."""
    sgid = str(gid)
    return (-int(_GPU_VRAM_MB.get(sgid, 0) or 0), int(sgid) if sgid.isdigit() else 9999)


def get_gpu_pool_policy() -> dict:
    """Return normalized dynamic GPU pool policy across all detected physical GPUs (0..N)."""
    all_ids = [str(gid) for gid, _ in (GPUS or [])]
    raw_str = db.get_setting("gpu_pool_policy", "{}") or "{}"
    try:
        raw = json.loads(raw_str)
        if not isinstance(raw, dict):
            raw = {}
    except Exception:
        raw = {}

    raw_disabled = {str(x).strip() for x in (raw.get("disabled_gpus") or []) if str(x).strip() in all_ids}
    enabled_ids = [gid for gid in all_ids if gid not in raw_disabled]

    cand_pri = str(raw.get("primary_gpu") if raw.get("primary_gpu") is not None else "").strip()
    if enabled_ids:
        primary_gpu = cand_pri if cand_pri in enabled_ids else sorted(enabled_ids, key=_gpu_hw_rank_key)[0]
    else:
        primary_gpu = None

    cand_sec = str(raw.get("secondary_gpu") if raw.get("secondary_gpu") is not None else "").strip()
    rem_enabled = [gid for gid in enabled_ids if gid != primary_gpu]
    if rem_enabled:
        secondary_gpu = cand_sec if cand_sec in rem_enabled else sorted(rem_enabled, key=_gpu_hw_rank_key)[0]
    else:
        secondary_gpu = None

    roles = {}
    for gid in all_ids:
        if gid in raw_disabled:
            roles[gid] = "disabled"
        elif gid == primary_gpu:
            roles[gid] = "primary"
        elif gid == secondary_gpu:
            roles[gid] = "secondary"
        else:
            roles[gid] = "worker"

    return {
        "all_gpu_ids": all_ids,
        "enabled_gpu_ids": enabled_ids,
        "disabled_gpus": [gid for gid in all_ids if gid in raw_disabled],
        "primary_gpu": primary_gpu,
        "secondary_gpu": secondary_gpu,
        "saved_primary_gpu": cand_pri if cand_pri in all_ids else None,
        "saved_secondary_gpu": cand_sec if cand_sec in all_ids else None,
        "roles": roles,
    }


def save_gpu_pool_policy(
    disabled_gpus: list[str] | None = None,
    primary_gpu: str | None = None,
    secondary_gpu: str | None = None,
) -> dict:
    """Persist dynamic per-GPU enable/disable + Primary/Secondary assignment across N GPUs."""
    all_ids = [str(gid) for gid, _ in (GPUS or [])]
    cur = get_gpu_pool_policy()
    prev_pri = cur.get("primary_gpu") or cur.get("saved_primary_gpu")
    prev_sec = cur.get("secondary_gpu") or cur.get("saved_secondary_gpu")

    if disabled_gpus is not None:
        dis_set = {str(x).strip() for x in disabled_gpus if str(x).strip() in all_ids}
    else:
        dis_set = set(cur["disabled_gpus"])

    req_pri = str(primary_gpu).strip() if primary_gpu is not None and str(primary_gpu).strip() in all_ids else None
    req_sec = str(secondary_gpu).strip() if secondary_gpu is not None and str(secondary_gpu).strip() in all_ids else None

    # If user explicitly promotes a card to primary or secondary without passing disabled_gpus, ensure it is enabled
    if disabled_gpus is None:
        if req_pri:
            dis_set.discard(req_pri)
        if req_sec:
            dis_set.discard(req_sec)

    enabled_ids = [gid for gid in all_ids if gid not in dis_set]

    new_pri = req_pri if (req_pri and req_pri in enabled_ids) else (prev_pri if prev_pri in enabled_ids else None)
    new_sec = req_sec if (req_sec and req_sec in enabled_ids) else (prev_sec if prev_sec in enabled_ids else None)

    # Seamless swap semantics when clicking Primary on current Secondary or Secondary on current Primary
    if req_pri and not req_sec and new_sec == req_pri:
        if prev_pri and prev_pri in enabled_ids and prev_pri != req_pri:
            new_sec = prev_pri
        else:
            rem = [g for g in enabled_ids if g != req_pri]
            new_sec = sorted(rem, key=_gpu_hw_rank_key)[0] if rem else None
    elif req_sec and not req_pri and new_pri == req_sec:
        if prev_sec and prev_sec in enabled_ids and prev_sec != req_sec:
            new_pri = prev_sec
        else:
            rem = [g for g in enabled_ids if g != req_sec]
            new_pri = sorted(rem, key=_gpu_hw_rank_key)[0] if rem else req_sec

    if enabled_ids and (not new_pri or new_pri not in enabled_ids):
        cands = [g for g in enabled_ids if g != new_sec] or enabled_ids
        new_pri = sorted(cands, key=_gpu_hw_rank_key)[0]

    rem_after_pri = [g for g in enabled_ids if g != new_pri]
    if rem_after_pri:
        if not new_sec or new_sec not in rem_after_pri:
            new_sec = sorted(rem_after_pri, key=_gpu_hw_rank_key)[0]
    else:
        new_sec = None

    payload = {
        "disabled_gpus": [gid for gid in all_ids if gid in dis_set],
        "primary_gpu": new_pri if new_pri is not None else (req_pri or prev_pri or ""),
        "secondary_gpu": new_sec if new_sec is not None else (req_sec or prev_sec or ""),
    }
    db.set_setting("gpu_pool_policy", json.dumps(payload, ensure_ascii=False))
    return get_gpu_pool_policy()


def parse_latest_log_metrics(run_dir_abs: str) -> dict:
    """Quickly parse latest step, total_steps, loss, smoothed sps, scale, and tail from train.log."""
    out = {}
    lp = os.path.join(run_dir_abs, "train.log")
    if os.path.isfile(lp):
        parsed = telemetry.parse_train_log(lp, max_bytes=32768)
        latest = parsed.get("latest") or {}
        if "step" in latest:
            out["last_step"] = int(latest["step"])
        if "total" in latest:
            out["last_loss"] = float(latest["total"])
        if "scale" in latest:
            out["scale"] = int(latest["scale"])
        if "sps" in latest:
            cur_scale = latest.get("scale")
            steady_sps = [
                float(s["sps"])
                for s in (parsed.get("steps") or [])[-8:]
                if int(s.get("step") or 0) > 35
                and float(s.get("sps") or 0) > 0
                and (cur_scale is None or s.get("scale") == cur_scale)
            ]
            if steady_sps:
                out["sps"] = sum(steady_sps[-5:]) / len(steady_sps[-5:])
        if parsed.get("tail"):
            out["tail"] = parsed["tail"][-20:]
    info_p = os.path.join(run_dir_abs, "run_info.json")
    if os.path.isfile(info_p):  # written by the trainer / submit_task: authoritative run facts
        try:
            with open(info_p, encoding="utf-8") as f:
                info = json.load(f)
            if info.get("total_steps"):
                out["total_steps"] = int(info["total_steps"])
            if info.get("batch_size"):
                out["batch_size"] = int(info["batch_size"])
            if info.get("crop_size"):
                out["crop_size"] = int(info["crop_size"])
            if info.get("frames"):
                out["frames"] = int(info["frames"])
            if info.get("auto_extended"):
                out["auto_extended"] = bool(info["auto_extended"])
                out["auto_extended_rounds"] = int(info.get("auto_extended_rounds") or 0)
            if info.get("stopped_after_optimal"):
                out["stopped_after_optimal"] = bool(info["stopped_after_optimal"])
                out["stop_after_optimal_epochs"] = int(info.get("stop_after_optimal_epochs") or 0)
                out["optimal_step"] = int(info.get("optimal_step") or 0)
                out["optimal_epoch"] = int(info.get("optimal_epoch") or 0)
        except Exception:
            pass
    return out


def _resolve_target_gpu_hw(gpu_id: str = None) -> tuple[str, int]:
    """Resolve (clean_gpu_name, vram_mb) for a given gpu_id (or active Primary GPU when auto)."""
    sgid = str(gpu_id) if gpu_id is not None else ""
    lbl_map = {str(g): str(lbl) for g, lbl in (GPUS or [])} or _GPU_LABEL_MAP
    if sgid and sgid != "auto" and sgid in lbl_map:
        raw_lbl = lbl_map[sgid]
        clean = raw_lbl.split(":", 1)[1].strip() if ":" in raw_lbl else raw_lbl
        return clean, int(_GPU_VRAM_MB.get(sgid, 0) or 0)
    try:
        pol = get_gpu_pool_policy()
        pri = pol.get("primary_gpu")
        if pri and str(pri) in lbl_map:
            raw_lbl = lbl_map[str(pri)]
            clean = raw_lbl.split(":", 1)[1].strip() if ":" in raw_lbl else raw_lbl
            return clean, int(_GPU_VRAM_MB.get(str(pri), 0) or 0)
    except Exception:
        pass
    if _GPUS_META:
        best = max(_GPUS_META, key=lambda x: (int(x[2] or 0), -(int(x[0]) if str(x[0]).isdigit() else 0)))
        raw_lbl = str(best[1])
        clean = raw_lbl.split(":", 1)[1].strip() if ":" in raw_lbl else raw_lbl
        return clean, int(best[2] or 0)
    return "", 0


def _baseline_stage_sps(cfg: dict, gpu_id: str = None) -> tuple[float, float, float, float]:
    """Return (sps_1_4, sps_1_2, sps_1_1, warmup_sec) from `data/gpu_benchmarks.db` calibrated on empirical runs."""
    gname, vram_mb = _resolve_target_gpu_hw(gpu_id)
    b4, b2, b1, wsec, _ = gpu_speed_db.predict_stage_sps(cfg or {}, gpu_name=gname, vram_mb=vram_mb)
    return (max(2.0, b4), max(1.5, b2), max(1.0, b1), max(3.0, wsec))


def estimate_task_seconds(
    cfg: dict,
    start_step: int = 0,
    total_steps: int = 0,
    live_sps: float = 0.0,
    live_scale: int = 0,
    started_at: float = None,
    now: float = None,
    gpu_id: str = None,
) -> float:
    """Estimate remaining (or full queued) wall-clock seconds using GPU-calibrated stage speeds."""
    cfg = cfg or {}
    tr = cfg.get("train") or {}
    io_cfg = cfg.get("io") or {}
    tot = int(total_steps or tr.get("total_steps") or 10000)
    cur = max(0, min(tot, int(start_step or 0)))
    rem_steps = tot - cur
    if rem_steps <= 0:
        return 0.0

    multires = bool(tr.get("multi_resolution", True))
    b4, b2, b1, wsec_base = _baseline_stage_sps(cfg, gpu_id=gpu_id)
    st_adv = ((cfg.get("advanced") or {}).get("stage_schedule")) or {}
    if multires and bool(st_adv.get("enabled", False)):
        rq = max(0.0, float(st_adv.get("r_quarter", 15.0)))
        rh = max(0.0, float(st_adv.get("r_half", 25.0)))
        rf = max(0.0, float(st_adv.get("r_full", 60.0)))
        r_sum = rq + rh + rf
        if r_sum > 1e-6:
            s1 = min(tot, max(0, int(round(tot * (rq / r_sum)))))
            s2 = min(tot, max(s1, int(round(tot * ((rq + rh) / r_sum)))))
        else:
            s = math.ceil(tot / 3.0)
            s1, s2 = min(tot, s), min(tot, 2 * s)
    else:
        s = math.ceil(tot / 3.0)
        s1, s2 = min(tot, s), min(tot, 2 * s)

    if live_sps and live_sps > 0:
        if live_scale in (4, 2, 1):
            sc = int(live_scale)
        elif not multires:
            sc = 1
        else:
            sc = 4 if cur < s1 else (2 if cur < s2 else 1)
        base_cur = {4: b4, 2: b2, 1: b1}[sc]
        ratio = min(8.0, max(0.08, float(live_sps) / max(1.0, base_cur)))
        c4, c2, c1 = b4 * ratio, b2 * ratio, b1 * ratio
    else:
        c4, c2, c1 = b4, b2, b1

    if multires:
        r4 = max(0, s1 - cur)
        r2 = max(0, s2 - max(cur, s1))
        r1 = max(0, tot - max(cur, s2))
        compute_sec = (r4 / c4) + (r2 / c2) + (r1 / c1)
    else:
        compute_sec = rem_steps / c1

    ckpt_iv = max(10, int(io_cfg.get("checkpoint_interval") or 1000))
    rem_ckpts = max(1, rem_steps // ckpt_iv)
    ckpt_overhead = rem_ckpts * 0.12

    if started_at and now:
        elapsed = max(0.0, float(now) - float(started_at))
        startup_overhead = 0.0 if (live_sps and live_sps > 0) else max(1.0, wsec_base - elapsed)
    else:
        startup_overhead = wsec_base

    return max(1.0, compute_sec + ckpt_overhead + startup_overhead)


_LOG_PREFIX_RE = re.compile(r"^\[[0-9]{2}:[0-9]{2}:[0-9]{2}\] ")


class _ActiveJob:
    def __init__(
        self,
        task_id: int,
        node_name: str,
        gpu: str,
        proc: subprocess.Popen,
        log_path: str = None,
        preview_gpu: str = None,
        run_dir: str = None,
    ):
        self.task_id = task_id
        self.node_name = node_name
        self.gpu = str(gpu)
        self.preview_gpu = str(preview_gpu) if preview_gpu is not None else None
        self.run_dir = run_dir
        self.proc = proc
        self.log_path = log_path
        self.tail = collections.deque(maxlen=300)
        self.stop_requested = False
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()

    def _read_stdout(self):
        try:
            for line in self.proc.stdout:
                s = line.rstrip("\n")
                self.tail.append(s)
                if self.log_path and not _LOG_PREFIX_RE.match(s):
                    try:
                        with open(self.log_path, "a", encoding="utf-8") as f:
                            f.write(s + "\n")
                            f.flush()
                    except Exception:
                        pass
        except Exception:
            pass

    @property
    def alive(self) -> bool:
        return self.proc.poll() is None


class TaskQueueScheduler:
    """Thread-safe Multi-User Task Queue & GPU Resource Scheduler."""

    def __init__(self):
        self._lock = threading.RLock()
        self._active: dict[int, _ActiveJob] = {}  # task_id -> _ActiveJob
        self._running = False
        self._thread = None

    def start(self):
        with self._lock:
            if self._running:
                return
            self._running = True
            self._reconcile_on_startup()
            self._thread = threading.Thread(target=self._loop, daemon=True, name="WildCatScheduler")
            self._thread.start()

    @staticmethod
    def _classify_exited_run(rdir_abs: str, fallback_step: int = 0, fallback_total: int = 0) -> tuple[str, str, dict]:
        """Classify final status ('completed' | 'stopped' | 'failed') for an orphaned DB running task whose PID exited."""
        m = parse_latest_log_metrics(rdir_abs) if rdir_abs and os.path.isdir(rdir_abs) else {}
        step = int(m.get("last_step") if m.get("last_step") is not None else (fallback_step or 0))
        tot = int(m.get("total_steps") or fallback_total or 0)
        tail_lines = m.get("tail") or []
        tail_text = "\n".join(tail_lines).strip()
        if any("done. (" in ln or ln.endswith("done.") for ln in tail_lines[-5:]) or (tot > 0 and step >= tot):
            return "completed", "", m
        if "Traceback (most recent call last):" in tail_text and not any(ln.endswith("stopped.") for ln in tail_lines[-4:]):
            return "failed", tail_text, m
        return "stopped", "", m

    def _reconcile_on_startup(self):
        """Sync existing disk runs into DB and reset stale 'running' tasks if process is dead."""
        db.init_db()
        now = time.time()
        admin = db.query_one("SELECT id, username FROM users WHERE username = 'admin'")
        if admin:
            admin_id = admin["id"]
            from . import service as _svc

            _svc.seed_demo_assets_for_user(admin_id)
            if os.path.isdir(storage.RUN_ROOT):
                for d in sorted(os.listdir(storage.RUN_ROOT)):
                    if d.startswith(".") or d.startswith("_") or d == "verification_report":
                        continue
                    rdir_abs = os.path.join(storage.RUN_ROOT, d)
                    if not os.path.isdir(rdir_abs):
                        continue
                    exists = db.query_one("SELECT id FROM tasks WHERE node_name = ?", (d,))
                    if exists:
                        continue
                    cfg_file = os.path.join(rdir_abs, f"{d}.json")
                    if not os.path.isfile(cfg_file):
                        cfg_file = os.path.join(storage.CONFIG_ROOT, f"{d}.json")
                    cfg_obj = {}
                    if os.path.isfile(cfg_file):
                        try:
                            with open(cfg_file, "r", encoding="utf-8") as f:
                                cfg_obj = json.load(f)
                        except Exception:
                            cfg_obj = {}
                    metrics = parse_latest_log_metrics(rdir_abs)
                    last_step = metrics.get("last_step", 0)
                    tot_steps = metrics.get("total_steps") or cfg_obj.get("train", {}).get("total_steps", 0)
                    last_loss = metrics.get("last_loss")
                    has_ckpt = os.path.isfile(os.path.join(rdir_abs, "ckpt", "latest.pt"))
                    st = "completed" if (has_ckpt and tot_steps and last_step >= tot_steps) else ("stopped" if has_ckpt else "idle")
                    db.execute(
                        """
                        INSERT OR IGNORE INTO tasks (
                            user_id, username, node_name, job_type, status, priority,
                            preferred_gpu, assigned_gpu, pid, config_path, run_dir, config_json,
                            last_step, total_steps, last_loss, created_at, finished_at
                        ) VALUES (?, ?, ?, 'train', ?, 0, 'auto', NULL, NULL, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            admin_id,
                            "admin",
                            d,
                            st,
                            f"config/{d}.json",
                            f"run/{d}",
                            json.dumps(cfg_obj),
                            last_step,
                            tot_steps,
                            last_loss,
                            now,
                            now,
                        ),
                    )

        # Reconcile 'running' tasks whose PID exited while server was offline
        running_rows = db.query_all("SELECT id, pid, run_dir, last_step, total_steps FROM tasks WHERE status = 'running'")
        for r in running_rows:
            pid = r.get("pid")
            if not _is_pid_alive(pid):
                rdir_abs = storage.abs_from_root(r["run_dir"]) if r.get("run_dir") else ""
                new_st, err_msg, m = self._classify_exited_run(rdir_abs, r.get("last_step", 0), r.get("total_steps", 0))
                db.execute(
                    """
                    UPDATE tasks
                    SET status = ?, pid = NULL, assigned_gpu = NULL, finished_at = ?, error_msg = ?,
                        last_step = COALESCE(?, last_step),
                        total_steps = COALESCE(?, total_steps),
                        last_loss = COALESCE(?, last_loss)
                    WHERE id = ?
                    """,
                    (new_st, now, err_msg, m.get("last_step"), m.get("total_steps"), m.get("last_loss"), r["id"]),
                )
                try:
                    analytics_db.sync_task_log_to_db(db.query_one("SELECT * FROM tasks WHERE id = ?", (r["id"],)), force=True)
                except Exception:
                    pass

    def _loop(self):
        while self._running:
            try:
                self.tick()
            except Exception:
                sys.stderr.write(f"[WildCat ERROR] Scheduler tick exception:\n{traceback.format_exc()}\n")
                sys.stderr.flush()
            time.sleep(1.0)

    def tick(self):
        with self._lock:
            now = time.time()
            # 1a. Check active in-process jobs and update live metrics
            finished_ids = []
            for tid, job in list(self._active.items()):
                task_row = db.query_one("SELECT run_dir, total_steps FROM tasks WHERE id = ?", (tid,))
                if task_row:
                    rdir_abs = storage.abs_from_root(task_row["run_dir"])
                    m = parse_latest_log_metrics(rdir_abs)
                    if m:
                        db.execute(
                            """
                            UPDATE tasks
                            SET last_step = COALESCE(?, last_step),
                                total_steps = COALESCE(?, total_steps),
                                last_loss = COALESCE(?, last_loss)
                            WHERE id = ?
                            """,
                            (m.get("last_step"), m.get("total_steps"), m.get("last_loss"), tid),
                        )
                if not job.alive:
                    try:
                        job._reader.join(timeout=2.0)
                    except Exception:
                        pass
                    rc = job.proc.poll()
                    if job.stop_requested or rc == -signal.SIGINT or rc == 130:
                        new_status = "stopped"
                        err = ""
                    elif rc == 0:
                        new_status = "completed"
                        err = ""
                    else:
                        new_status = "failed"
                        err = "\n".join(list(job.tail)[-100:]).strip()
                        if not err:
                            err = i18n.t("server.subprocAbnormalExit", rc=rc)
                        sys.stderr.write(
                            f"[WildCat ERROR] Task '{job.node_name}' (id={tid}, gpu={job.gpu}) failed with exit code {rc}:\n{err}\n"
                        )
                        sys.stderr.flush()
                        if job.log_path:
                            try:
                                existing_tail = ""
                                if os.path.isfile(job.log_path):
                                    with open(job.log_path, "r", encoding="utf-8", errors="replace") as rf:
                                        existing_tail = rf.read()[-4096:]
                                last_err_line = err.splitlines()[-1] if err.splitlines() else err
                                if not existing_tail or last_err_line not in existing_tail:
                                    with open(job.log_path, "a", encoding="utf-8") as lf:
                                        lf.write("\n" + err + "\n")
                                        lf.flush()
                            except Exception:
                                pass
                    db.execute(
                        """
                        UPDATE tasks
                        SET status = ?, pid = NULL, assigned_gpu = NULL, finished_at = ?, error_msg = ?
                        WHERE id = ?
                        """,
                        (new_status, now, err, tid),
                    )
                    finished_ids.append(tid)

            for tid in finished_ids:
                self._active.pop(tid, None)
                try:
                    t_done = db.query_one("SELECT * FROM tasks WHERE id = ?", (tid,))
                    if t_done:
                        analytics_db.sync_task_log_to_db(t_done, force=True)
                except Exception:
                    pass

            # 1b. Track any surviving DB 'running' processes (e.g. across hot-update / web restart)
            surviving_running = []
            db_running = db.query_all(
                "SELECT id, node_name, assigned_gpu, pid, run_dir, last_step, total_steps FROM tasks WHERE status = 'running'"
            )
            for r in db_running:
                rid = r["id"]
                if rid in self._active:
                    continue
                rdir_abs = storage.abs_from_root(r["run_dir"]) if r.get("run_dir") else ""
                if _is_pid_alive(r.get("pid")):
                    m = parse_latest_log_metrics(rdir_abs) if rdir_abs and os.path.isdir(rdir_abs) else {}
                    if m:
                        db.execute(
                            """
                            UPDATE tasks
                            SET last_step = COALESCE(?, last_step),
                                total_steps = COALESCE(?, total_steps),
                                last_loss = COALESCE(?, last_loss)
                            WHERE id = ?
                            """,
                            (m.get("last_step"), m.get("total_steps"), m.get("last_loss"), rid),
                        )
                    surviving_running.append(r)
                else:
                    new_st, err_msg, m = self._classify_exited_run(
                        rdir_abs, r.get("last_step", 0), r.get("total_steps", 0)
                    )
                    db.execute(
                        """
                        UPDATE tasks
                        SET status = ?, pid = NULL, assigned_gpu = NULL, finished_at = ?, error_msg = ?,
                            last_step = COALESCE(?, last_step),
                            total_steps = COALESCE(?, total_steps),
                            last_loss = COALESCE(?, last_loss)
                        WHERE id = ?
                        """,
                        (new_st, now, err_msg, m.get("last_step"), m.get("total_steps"), m.get("last_loss"), rid),
                    )
                    try:
                        analytics_db.sync_task_log_to_db(db.query_one("SELECT * FROM tasks WHERE id = ?", (rid,)), force=True)
                    except Exception:
                        pass

            # 2. Determine strict GPU occupancy & apply dynamic Admin GPU pool policy (enabled / Primary / Secondary)
            max_per_gpu = max(1, int(db.get_setting("max_jobs_per_gpu", "1") or 1))
            policy = get_gpu_pool_policy()
            gpu_Ids = [str(g[0]) for g in GPUS] or ["0"]
            enabled_gpu_ids = [gid for gid in policy["enabled_gpu_ids"] if gid in gpu_Ids]
            disabled_set = set(policy["disabled_gpus"])
            primary_gpu = policy["primary_gpu"]
            secondary_gpu = policy["secondary_gpu"]

            # Immediately yield any borrowed secondary preview_gpu that was disabled by Admin
            if disabled_set:
                for active_job in self._active.values():
                    if active_job.alive and active_job.preview_gpu and str(active_job.preview_gpu) in disabled_set:
                        if active_job.run_dir and os.path.isdir(active_job.run_dir):
                            try:
                                with open(os.path.join(active_job.run_dir, ".yield_preview_gpu"), "w", encoding="utf-8") as yf:
                                    yf.write("yield\n")
                            except OSError:
                                pass
                        active_job.preview_gpu = None

            gpu_counts = {gid: 0 for gid in gpu_Ids}
            for job in self._active.values():
                if job.alive:
                    gid = str(job.gpu) if str(job.gpu) in gpu_counts else gpu_Ids[0]
                    gpu_counts[gid] = gpu_counts.get(gid, 0) + 1
            for r in surviving_running:
                raw_gid = str(r.get("assigned_gpu") if r.get("assigned_gpu") is not None else "")
                gid = raw_gid if raw_gid in gpu_counts else gpu_Ids[0]
                gpu_counts[gid] = gpu_counts.get(gid, 0) + 1

            max_enabled_jobs = len(enabled_gpu_ids) * max_per_gpu
            surviving_gpus = {
                (str(r.get("assigned_gpu")) if str(r.get("assigned_gpu") or "") in gpu_counts else gpu_Ids[0])
                for r in surviving_running
            }

            # 3. Dispatch queued tasks onto enabled GPUs in priority + FIFO order (Primary GPU first, then Secondary, then workers)
            if enabled_gpu_ids:
                queued = db.query_all(
                    "SELECT * FROM tasks WHERE status = 'queued' ORDER BY priority DESC, created_at ASC, id ASC"
                )
                for q in queued:
                    if sum(gpu_counts.get(g, 0) for g in enabled_gpu_ids) >= max_enabled_jobs:
                        break
                    pref = str(q.get("preferred_gpu") or "auto")
                    if pref != "auto" and pref not in enabled_gpu_ids:
                        pref = "auto"

                    chosen_gpu = None
                    if pref == "auto":
                        ordered_candidates = sorted(
                            enabled_gpu_ids,
                            key=lambda g: (
                                gpu_counts.get(g, 0),
                                0 if g == primary_gpu else (1 if g == secondary_gpu else 2),
                                -_GPU_VRAM_MB.get(g, 0),
                                int(g) if str(g).isdigit() else 0,
                            ),
                        )
                        for gid in ordered_candidates:
                            if gpu_counts.get(gid, 0) < max_per_gpu:
                                chosen_gpu = gid
                                break
                    elif gpu_counts.get(pref, 0) < max_per_gpu:
                        chosen_gpu = pref

                    if chosen_gpu is not None:
                        try:
                            self._launch_queued_task(q, chosen_gpu, surviving_gpus=surviving_gpus)
                            gpu_counts[chosen_gpu] = gpu_counts.get(chosen_gpu, 0) + 1
                        except Exception:
                            tb = traceback.format_exc()
                            sys.stderr.write(
                                f"[WildCat ERROR] Failed to launch task '{q.get('node_name')}' (id={q.get('id')}):\n{tb}\n"
                            )
                            sys.stderr.flush()
                            try:
                                r_abs = storage.abs_from_root(q["run_dir"])
                                os.makedirs(r_abs, exist_ok=True)
                                with open(os.path.join(r_abs, "train.log"), "a", encoding="utf-8") as lf:
                                    lf.write(tb + "\n")
                            except Exception:
                                pass
                            db.execute(
                                "UPDATE tasks SET status = 'failed', pid = NULL, assigned_gpu = NULL, finished_at = ?, error_msg = ? WHERE id = ?",
                                (time.time(), tb, q["id"]),
                            )

    def _launch_queued_task(self, task_row: dict, gpu_id: str, surviving_gpus: set = None):
        tid = task_row["id"]
        jtype = task_row["job_type"]
        cfg_rel = task_row["config_path"]
        cfg_obj = {}
        try:
            cfg_obj = json.loads(task_row.get("config_json") or "{}")
        except Exception:
            pass

        # Reclaim gpu_id if any currently active training job was borrowing it as a secondary preview_gpu
        for active_job in self._active.values():
            if active_job.alive and active_job.preview_gpu == str(gpu_id):
                if active_job.run_dir and os.path.isdir(active_job.run_dir):
                    try:
                        with open(os.path.join(active_job.run_dir, ".yield_preview_gpu"), "w", encoding="utf-8") as yf:
                            yf.write("yield\n")
                    except OSError:
                        pass
                active_job.preview_gpu = None

        env = storage.apply_subprocess_storage_env(os.environ.copy())
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        pick = None
        # Preview / validation on an enabled secondary GPU: prefer designated Secondary GPU first, never touch disabled GPUs
        policy = get_gpu_pool_policy()
        enabled_set = set(policy["enabled_gpu_ids"])
        primary_gpu = policy["primary_gpu"]
        secondary_gpu = policy["secondary_gpu"]
        pv = str(((cfg_obj.get("optimizations") or {}).get("lossless") or {}).get("preview_gpu", "auto"))
        busy = (
            {str(j.gpu) for j in self._active.values() if j.alive}
            | {str(j.preview_gpu) for j in self._active.values() if j.alive and j.preview_gpu}
            | {str(x) for x in (surviving_gpus or set())}
            | {str(gpu_id)}
        )
        other_queued = db.query_one("SELECT id FROM tasks WHERE status = 'queued' AND id != ? LIMIT 1", (tid,))
        others = [str(g[0]) for g in GPUS if str(g[0]) != str(gpu_id) and str(g[0]) in enabled_set]
        if pv == "auto":
            if not other_queued:
                idle = sorted(
                    [g for g in others if g not in busy],
                    key=lambda g: (
                        0 if g == secondary_gpu else (2 if g == primary_gpu else 1),
                        -_GPU_VRAM_MB.get(g, 0),
                        int(g) if str(g).isdigit() else 0,
                    ),
                )
                pick = idle[0] if idle else None
        elif pv not in ("same", "off", "cpu", "") and pv in others and pv not in busy:
            pick = pv
        if pick is not None:
            env["CUDA_VISIBLE_DEVICES"] = "%s,%s" % (gpu_id, pick)
            env["WILDCAT_PREVIEW_GPU"] = "1"
        else:
            env["WILDCAT_PREVIEW_GPU"] = "off" if pv == "off" else "same"

        cfg_abs_arg = storage.abs_from_root(cfg_rel)
        cmd = [PYTHON_BIN, "-u", "train.py", cfg_abs_arg]

        run_abs = storage.abs_from_root(task_row["run_dir"], for_write=True)
        os.makedirs(run_abs, exist_ok=True)
        for stale_flag in (".stop", ".yield_preview_gpu"):
            fp = os.path.join(run_abs, stale_flag)
            if os.path.exists(fp):
                try:
                    os.remove(fp)
                except OSError:
                    pass
        log_path = os.path.join(run_abs, "train.log")

        popen_kwargs = dict(
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        if os.name == "nt":
            popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

        proc = subprocess.Popen(cmd, **popen_kwargs)
        job = _ActiveJob(
            tid,
            task_row["node_name"],
            str(gpu_id),
            proc,
            log_path=log_path,
            preview_gpu=pick,
            run_dir=run_abs,
        )
        self._active[tid] = job
        db.execute(
            """
            UPDATE tasks
            SET status = 'running', assigned_gpu = ?, pid = ?, started_at = ?, finished_at = NULL, error_msg = ''
            WHERE id = ?
            """,
            (str(gpu_id), proc.pid, time.time(), tid),
        )

    def enrich_task_with_queue(self, task_row: dict, qstat: dict = None) -> dict:
        if not task_row:
            return task_row
        qstat = qstat or self.queue_status()
        tid = task_row["id"]
        q_item = next((q for q in qstat["queued"] if q["id"] == tid), None)
        r_item = next((r for r in qstat["running"] if r["id"] == tid), None)
        if q_item:
            task_row["queue_position"] = q_item["queue_position"]
            task_row["ahead_queued_count"] = q_item["ahead_queued_count"]
            task_row["running_count"] = q_item["running_count"]
            task_row["ahead_count"] = q_item["ahead_count"]
            task_row["target_gpu"] = q_item["target_gpu"]
            task_row["estimated_wait_sec"] = q_item["estimated_wait_sec"]
            task_row["estimated_duration_sec"] = q_item["estimated_duration_sec"]
        else:
            task_row["queue_position"] = None
            task_row["ahead_queued_count"] = 0
            task_row["running_count"] = qstat["running_count"]
            task_row["ahead_count"] = 0
            task_row["target_gpu"] = r_item["assigned_gpu"] if r_item else None
            task_row["estimated_wait_sec"] = 0.0
            task_row["estimated_duration_sec"] = None
        if r_item:
            task_row["last_step"] = r_item["last_step"]
            task_row["total_steps"] = r_item["total_steps"]
            task_row["estimated_remaining_sec"] = r_item["estimated_remaining_sec"]
        else:
            task_row["estimated_remaining_sec"] = None
        return task_row

    @staticmethod
    def _sanitize_train_config(cfg: dict, is_privileged: bool) -> dict:
        if not isinstance(cfg, dict):
            raise ValueError(i18n.t("server.invalidTrainParam", field="config"))
        out = dict(cfg)
        for sec_name in ("paths", "train", "model", "io", "color"):
            sec_val = out.get(sec_name)
            if sec_val is None:
                out[sec_name] = {}
            elif isinstance(sec_val, dict):
                out[sec_name] = dict(sec_val)
            else:
                raise ValueError(i18n.t("server.invalidTrainParam", field=sec_name))

        # Never allow raw JSON config to redirect checkpoint/log output outside managed storage/run/<node_name>
        out["paths"].pop("run_dir", None)

        if not is_privileged:
            # Strip host file-probing OCIO path and raw model tensor-size overrides
            out["color"].pop("ocio_config", None)
            mo = out["model"]
            for raw_k in ("n_feats", "n_resblocks", "k_size"):
                mo.pop(raw_k, None)
            msize = str(mo.get("model_size") or "Medium").strip()
            if msize not in ("Small", "Medium", "Large"):
                raise ValueError(i18n.t("server.invalidTrainParam", field="model.model_size"))
            mo["model_size"] = msize

            try:
                in_ch = int(mo.get("in_channels", 3))
                out_ch = int(mo.get("out_channels", in_ch))
            except (TypeError, ValueError):
                raise ValueError(i18n.t("server.invalidTrainParam", field="model.channels"))
            if in_ch not in (1, 3, 4) or out_ch not in (1, 3, 4):
                raise ValueError(i18n.t("server.invalidTrainParam", field="model.channels"))
            mo["in_channels"] = in_ch
            mo["out_channels"] = out_ch

            tr = out["train"]
            try:
                crop_sz = int(tr.get("crop_size", 256) or 256)
            except (TypeError, ValueError):
                raise ValueError(i18n.t("server.invalidTrainParam", field="train.crop_size"))
            if crop_sz < 32 or crop_sz > 512:
                raise ValueError(i18n.t("server.invalidTrainParam", field="train.crop_size"))
            tr["crop_size"] = crop_sz

            raw_bs = tr.get("batch_size")
            if raw_bs is not None and str(raw_bs).strip().lower() != "auto":
                try:
                    bs_i = int(raw_bs)
                except (TypeError, ValueError):
                    raise ValueError(i18n.t("server.invalidTrainParam", field="train.batch_size"))
                if bs_i < 1 or bs_i > 64:
                    raise ValueError(i18n.t("server.invalidTrainParam", field="train.batch_size"))
                tr["batch_size"] = bs_i

            if "lr" in tr and tr["lr"] is not None:
                try:
                    lr_f = float(tr["lr"])
                except (TypeError, ValueError):
                    raise ValueError(i18n.t("server.invalidTrainParam", field="train.lr"))
                if not math.isfinite(lr_f) or lr_f < 1e-7 or lr_f > 1e-2:
                    raise ValueError(i18n.t("server.invalidTrainParam", field="train.lr"))

            if "stop_after_optimal_epochs" in tr and tr["stop_after_optimal_epochs"] is not None:
                try:
                    sao_i = int(tr["stop_after_optimal_epochs"])
                except (TypeError, ValueError):
                    raise ValueError(i18n.t("server.invalidTrainParam", field="train.stop_after_optimal_epochs"))
                if sao_i < 0 or sao_i > 10_000_000:
                    raise ValueError(i18n.t("server.invalidTrainParam", field="train.stop_after_optimal_epochs"))
                tr["stop_after_optimal_epochs"] = sao_i

            io_cfg = out["io"]
            for io_k, min_v in (("checkpoint_interval", 1000), ("preview_interval", 100), ("log_interval", 10)):
                if io_k in io_cfg and io_cfg[io_k] is not None:
                    try:
                        iv = int(io_cfg[io_k])
                    except (TypeError, ValueError):
                        raise ValueError(i18n.t("server.invalidTrainParam", field=f"io.{io_k}"))
                    if iv < min_v or iv > 10_000_000:
                        raise ValueError(i18n.t("server.invalidTrainParam", field=f"io.{io_k}"))
                    io_cfg[io_k] = iv

        return out

    def submit_task(
        self,
        user_id: int,
        username: str,
        node_name: str,
        job_type: str,
        cfg: dict,
        preferred_gpu: str = "auto",
        priority: int = 0,
        is_admin: bool = False,
        client_ip: str = "",
    ) -> dict:
        """Create or update a task node and place it at the back of the Multi-User Task Queue."""
        if not isinstance(cfg, dict):
            raise ValueError(i18n.t("server.invalidTrainParam", field="config"))
        cfg = dict(cfg)
        requeue_extend = bool(cfg.pop("_requeue_extend", False))

        # If extending an actively running task, gracefully stop and checkpoint it first
        stop_tid = None
        with self._lock:
            clean_probe = storage.sanitize_name(node_name, "WildCat1")
            own_probe = db.query_one("SELECT * FROM tasks WHERE user_id = ? AND node_name = ?", (user_id, clean_probe))
            if own_probe and own_probe["status"] == "running" and requeue_extend:
                stop_tid = own_probe["id"]
        if stop_tid is not None:
            self.stop_task(stop_tid, wait_timeout=25.0)

        with self._lock:
            clean_node = storage.sanitize_name(node_name, "WildCat1")
            # Ensure node names are isolated per user if another user or untracked run directory owns clean_node
            own = db.query_one("SELECT * FROM tasks WHERE user_id = ? AND node_name = ?", (user_id, clean_node))
            if own:
                existing = own
            else:
                other = db.query_one("SELECT * FROM tasks WHERE node_name = ? AND user_id != ?", (clean_node, user_id))
                if other or os.path.exists(storage.abs_from_root(f"run/{clean_node}")):
                    clean_node = f"{storage.sanitize_name(username)}_{clean_node}"
                existing = db.query_one("SELECT * FROM tasks WHERE user_id = ? AND node_name = ?", (user_id, clean_node))

            if existing and existing["id"] in self._active:
                self.tick()
                if existing["id"] in self._active:
                    raise RuntimeError(i18n.t("server.taskSavingExit", node=clean_node))
                existing = db.query_one("SELECT * FROM tasks WHERE id = ?", (existing["id"],))

            if existing and existing["status"] in ("running", "queued") and not requeue_extend:
                raise RuntimeError(i18n.t("server.taskAlreadyActive", node=clean_node))

            u_row = db.query_one(
                "SELECT username, password_hash, role, is_active, max_tasks, submitted_task_count, max_epochs, perm_lab, perm_api, created_ip, last_ip FROM users WHERE id = ?",
                (user_id,),
            )
            if u_row and not u_row.get("is_active"):
                raise PermissionError(i18n.t("server.accountDisabled", username=u_row["username"]))

            restrict_on = str(db.get_setting("enable_temp_user_restrictions", "1") or "1").strip() != "0"
            is_privileged = bool(is_admin or (u_row and u_row.get("role") == "admin") or not restrict_on)
            if not is_privileged:
                priority = 0
                preferred_gpu = "auto"

            cfg = self._sanitize_train_config(cfg, is_privileged=is_privileged)
            cfg["run_name"] = clean_node

            run_rel = f"run/{clean_node}"
            is_resume = bool(cfg.get("train", {}).get("resume", False))
            run_abs = storage.abs_from_root(run_rel, for_write=not is_resume)
            paths_cfg = dict(cfg.get("paths") or {})
            cfg["paths"] = paths_cfg
            src_p = str(paths_cfg.get("source") or "").strip()
            tgt_p = str(paths_cfg.get("target") or "").strip()
            eval_seq_p = str(paths_cfg.get("eval_sequence") or "").strip()
            if not eval_seq_p:
                exp_meta = cfg.get("experiment") or {}
                ds_row = None
                for cand_aid in (exp_meta.get("eval_asset_id"), exp_meta.get("dataset_id")):
                    if not cand_aid:
                        continue
                    try:
                        row_cand = db.query_one(
                            "SELECT user_id, kind, source_path, rel_path, seq_path FROM assets WHERE id = ?",
                            (int(cand_aid),),
                        )
                        if row_cand and not is_privileged and int(row_cand.get("user_id") or 0) != int(user_id):
                            row_cand = None
                        if row_cand:
                            cand_seq_probe = (
                                str(row_cand.get("source_path") or row_cand.get("rel_path") or "").strip()
                                if row_cand.get("kind") == "eval_sequence"
                                else str(row_cand.get("seq_path") or "").strip()
                            )
                            if cand_seq_probe and len(storage.expand_sequence_files(cand_seq_probe)) >= 2:
                                ds_row = {"seq_path": cand_seq_probe}
                                break
                    except Exception:
                        pass
                if not ds_row and src_p:
                    ds_row = db.query_one(
                        "SELECT seq_path FROM assets WHERE user_id = ? AND source_path = ? AND seq_path != '' LIMIT 1",
                        (user_id, src_p),
                    )
                if ds_row and ds_row.get("seq_path"):
                    cand_seq = str(ds_row["seq_path"]).strip()
                    if len(storage.expand_sequence_files(cand_seq)) >= 2:
                        eval_seq_p = cand_seq
                        paths_cfg["eval_sequence"] = eval_seq_p
            elif len(storage.expand_sequence_files(eval_seq_p)) < 2:
                eval_seq_p = ""
                paths_cfg.pop("eval_sequence", None)

            if existing and is_resume:
                if not is_privileged and not os.path.isfile(os.path.join(run_abs, "ckpt", "latest.pt")):
                    raise ValueError(i18n.t("server.noCheckpointsYet"))
                if not is_privileged:
                    try:
                        prev_cfg = json.loads(existing.get("config_json") or "{}")
                    except Exception:
                        prev_cfg = {}
                    prev_paths = prev_cfg.get("paths") or {}
                    prev_src = os.path.normpath(str(prev_paths.get("source") or "").strip())
                    prev_tgt = os.path.normpath(str(prev_paths.get("target") or "").strip())
                    if (prev_src and os.path.normpath(src_p) != prev_src) or (
                        prev_tgt and os.path.normpath(tgt_p) != prev_tgt
                    ):
                        raise PermissionError(i18n.t("server.tempUserResumeDatasetLocked"))

            if not existing:
                consume_quota = True
            elif is_resume:
                consume_quota = False
            else:
                never_ran = (
                    existing.get("status") == "cancelled"
                    and int(existing.get("last_step") or 0) == 0
                    and not existing.get("started_at")
                )
                consume_quota = not never_ran

            eff_ip_scope = ""
            if not is_privileged and u_row:
                raw_ip = client_ip or str(u_row.get("last_ip") or u_row.get("created_ip") or "")
                cand_scope = security.normalize_ip_scope(raw_ip) if raw_ip else ""
                if cand_scope and cand_scope not in ("local", "testclient", "localhost", "unknown"):
                    eff_ip_scope = cand_scope

            if consume_quota and not is_privileged and u_row:
                max_t = int(u_row.get("max_tasks") if u_row.get("max_tasks") is not None else 2)
                used_t = int(u_row.get("submitted_task_count") or 0)
                if used_t >= max_t:
                    raise PermissionError(
                        i18n.t("server.tempUserMaxTasksExceeded", max_tasks=max_t, used=used_t)
                    )
                if eff_ip_scope and u_row.get("password_hash") == "temp_algorithmic":
                    def_max_t = max(0, int(db.get_setting("default_user_max_tasks", "2") or 2))
                    ip_cap = max(max_t, def_max_t)
                    ip_row = db.query_one(
                        "SELECT anon_tasks_submitted FROM ip_quota_ledger WHERE ip_scope = ?",
                        (eff_ip_scope,),
                    )
                    ip_used = int((ip_row or {}).get("anon_tasks_submitted") or 0)
                    if ip_used >= ip_cap:
                        raise PermissionError(
                            i18n.t("server.tempUserMaxTasksExceeded", max_tasks=ip_cap, used=ip_used)
                        )
            if not is_privileged:
                allowed_ds_roots = [
                    os.path.abspath(storage.user_root(username)),
                    os.path.abspath(storage.DATASET_ROOT),
                ]
                for p_raw in (src_p, tgt_p) + ((eval_seq_p,) if eval_seq_p else ()):
                    if not p_raw or "\x00" in p_raw:
                        raise PermissionError(i18n.t("server.assetAccessForbidden"))
                    p_dir = os.path.abspath(storage.abs_from_root(p_raw))
                    if not os.path.isdir(p_dir):
                        p_dir = os.path.dirname(p_dir)
                    if not any(storage._is_under_dir(p_dir, base) is not None for base in allowed_ds_roots):
                        raise PermissionError(i18n.t("server.assetAccessForbidden"))
            s_files = storage.expand_sequence_files(src_p)
            t_files = storage.expand_sequence_files(tgt_p)
            if not s_files or not t_files:
                raise ValueError(
                    i18n.t("server.trainSeqMissing", src=src_p, src_n=len(s_files), tgt=tgt_p, tgt_n=len(t_files))
                )
            if not is_resume:
                if os.path.isfile(os.path.join(run_abs, "ckpt", "latest.pt")):
                    raise ValueError(i18n.t("server.taskNameHasHistory", node=clean_node))
                ckpt_dir = os.path.join(run_abs, "ckpt")
                if os.path.isdir(ckpt_dir):
                    shutil.rmtree(ckpt_dir, ignore_errors=True)
                for stale_fn in ("train.log", "recommended_models.json"):
                    old_fp = os.path.join(run_abs, stale_fn)
                    if os.path.isfile(old_fp):
                        try:
                            os.remove(old_fp)
                        except OSError:
                            pass

            # Enforce locked model structure when initial_weights or resume checkpoint is specified
            from wildcat import models

            tr_cfg = dict(cfg.get("train") or {})
            cfg["train"] = tr_cfg
            mo_cfg = dict(cfg.get("model") or {})
            cfg["model"] = mo_cfg
            init_w = tr_cfg.get("initial_weights", "")
            if is_resume:
                cand_pt = os.path.join(run_abs, "ckpt", "latest.pt")
                if os.path.isfile(cand_pt):
                    init_w = cand_pt
            if init_w and init_w != "None":
                if not is_privileged and ("/" in str(init_w) or "\\" in str(init_w) or str(init_w).endswith((".pt", ".cat"))):
                    cand_w_abs = os.path.abspath(storage.abs_from_root(init_w) if not os.path.isabs(init_w) else init_w)
                    allowed_w_roots = [
                        os.path.abspath(os.path.join(ROOT, "weights")),
                        os.path.abspath(storage.user_root(username)),
                        os.path.abspath(run_abs),
                    ]
                    for ur in db.query_all("SELECT run_dir FROM tasks WHERE user_id = ?", (user_id,)):
                        if ur.get("run_dir"):
                            allowed_w_roots.append(os.path.abspath(storage.abs_from_root(ur["run_dir"])))
                    if not any(storage._is_under_dir(cand_w_abs, base) is not None for base in allowed_w_roots):
                        raise PermissionError(i18n.t("server.assetAccessForbidden"))
                abs_w = storage.abs_from_root(init_w) if not os.path.isabs(init_w) else init_w
                w_spec = models.inspect_weight_source(
                    abs_w if os.path.isfile(abs_w) else init_w,
                    fallback_model_size=mo_cfg.get("model_size", "Medium"),
                    fallback_in=int(mo_cfg.get("in_channels", 3)),
                    fallback_out=int(mo_cfg.get("out_channels", mo_cfg.get("in_channels", 3))),
                )
                if w_spec.get("lock_model_size"):
                    mo_cfg["model_size"] = w_spec["model_size"]
                if w_spec.get("lock_channels"):
                    mo_cfg["in_channels"] = w_spec["in_channels"]
                    mo_cfg["out_channels"] = w_spec["out_channels"]

            # Auto-guard against 4-channel config when source/target sequence files only have 3 channels
            if job_type == "train" and (int(mo_cfg.get("in_channels", 3)) > 3 or int(mo_cfg.get("out_channels", 3)) > 3):
                from wildcat import color as _wcolor

                try:
                    s_ch = min(
                        (1 if im.ndim == 2 else int(im.shape[2]))
                        for im in (_wcolor.read_image_float(p, is_data_pass=True) for p in (s_files[:1] + s_files[-1:]))
                    )
                    t_ch = min(
                        (1 if im.ndim == 2 else int(im.shape[2]))
                        for im in (_wcolor.read_image_float(p, is_data_pass=True) for p in (t_files[:1] + t_files[-1:]))
                    )
                    if int(mo_cfg.get("out_channels", 3)) == 1:
                        if s_ch < int(mo_cfg.get("in_channels", 3)):
                            mo_cfg["in_channels"] = max(3, s_ch)
                    elif s_ch < int(mo_cfg.get("in_channels", 3)) or t_ch < int(mo_cfg.get("out_channels", 3)):
                        common_ch = max(3, min(s_ch, t_ch))
                        mo_cfg["in_channels"] = common_ch
                        mo_cfg["out_channels"] = common_ch
                except Exception:
                    pass

            # Compute exact CopyCat total_steps = ceil(epochs * n_frames / batch_size) with no upper cap on epochs
            n_frames = max(1, len(s_files)) if s_files else 1
            info_path = os.path.join(run_abs, "run_info.json")
            saved_info = {}
            if is_resume and os.path.isfile(info_path):
                try:
                    with open(info_path, "r", encoding="utf-8") as f:
                        saved_info = json.load(f) or {}
                except Exception:
                    saved_info = {}

            if tr_cfg.get("batch_size_auto", False) and saved_info.get("batch_size"):
                eff_bs = min(n_frames, max(1, int(saved_info["batch_size"])))
            else:
                eff_bs = min(n_frames, max(1, int(tr_cfg.get("batch_size") or 8)))
            tr_cfg["batch_size"] = eff_bs

            if tr_cfg.get("epochs"):
                epochs = max(1, int(tr_cfg["epochs"]))
                tot_steps = max(1, math.ceil(epochs * n_frames / float(eff_bs)))
            elif tr_cfg.get("total_steps"):
                tot_steps = max(1, int(tr_cfg["total_steps"]))
                epochs = max(1, math.ceil(tot_steps * eff_bs / float(n_frames)))
            else:
                epochs = 10000
                tot_steps = max(1, math.ceil(epochs * n_frames / float(eff_bs)))
            tr_cfg["epochs"] = epochs
            tr_cfg["total_steps"] = tot_steps
            tr_cfg["auto_extend_optimal"] = bool(tr_cfg.get("auto_extend_optimal", False))
            try:
                stop_opt_ep = max(0, min(10_000_000, int(tr_cfg.get("stop_after_optimal_epochs") or 0)))
            except (TypeError, ValueError):
                stop_opt_ep = 0
            tr_cfg["stop_after_optimal_epochs"] = stop_opt_ep
            tr_cfg.pop("max_auto_extend_epochs", None)

            if not is_privileged and u_row:
                max_ep = int(u_row.get("max_epochs") if u_row.get("max_epochs") is not None else 500000)
                if max_ep > 0:
                    if epochs > max_ep:
                        raise PermissionError(
                            i18n.t("server.tempUserMaxEpochsExceeded", max_epochs=max_ep, requested=epochs)
                        )
                    max_steps = max(1_000_000, max_ep * 10)
                    if tot_steps > max_steps:
                        raise PermissionError(
                            i18n.t("server.tempUserStepsExceeded", max_steps=max_steps, requested=tot_steps)
                        )
                    tr_cfg["max_auto_extend_epochs"] = max_ep

            init_last_step = 0
            init_last_loss = None
            if is_resume:
                m_prev = parse_latest_log_metrics(run_abs) if os.path.isdir(run_abs) else {}
                init_last_step = max(
                    int(existing.get("last_step") or 0) if existing else 0,
                    int(m_prev.get("last_step") or 0),
                )
                init_last_loss = m_prev.get("last_loss") if m_prev.get("last_loss") is not None else (existing.get("last_loss") if existing else None)
                if requeue_extend and tot_steps <= init_last_step:
                    raise ValueError(
                        i18n.t(
                            "server.extendEpochsTooSmall",
                            epochs=epochs,
                            tot_steps=tot_steps,
                            last_step=init_last_step,
                        )
                    )

            cfg_rel = f"config/{clean_node}.json"
            cfg_abs = storage.abs_from_root(cfg_rel, for_write=True)
            os.makedirs(os.path.dirname(cfg_abs), exist_ok=True)
            with open(cfg_abs, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)

            os.makedirs(run_abs, exist_ok=True)
            with open(os.path.join(run_abs, f"{clean_node}.json"), "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)

            if os.path.isfile(info_path):
                try:
                    updated_info = dict(saved_info)
                    updated_info["total_steps"] = tot_steps
                    updated_info["epochs"] = epochs
                    updated_info["batch_size"] = eff_bs
                    updated_info["frames"] = n_frames
                    with open(info_path, "w", encoding="utf-8") as f:
                        json.dump(updated_info, f)
                except Exception:
                    pass

            now = time.time()

            with db.transaction() as conn:
                if existing:
                    tid = existing["id"]
                    conn.execute(
                        """
                        UPDATE tasks
                        SET job_type = ?, status = 'queued', priority = ?, preferred_gpu = ?,
                            assigned_gpu = NULL, pid = NULL, config_path = ?, run_dir = ?,
                            config_json = ?, last_step = ?, total_steps = ?, last_loss = ?,
                            error_msg = '', created_at = ?, started_at = NULL, finished_at = NULL,
                            files_pruned = 0, analytics_synced_at = 0.0
                        WHERE id = ?
                        """,
                        (
                            job_type,
                            int(priority),
                            str(preferred_gpu or "auto"),
                            cfg_rel,
                            run_rel,
                            json.dumps(cfg),
                            init_last_step,
                            tot_steps,
                            init_last_loss,
                            now,
                            tid,
                        ),
                    )
                else:
                    cur = conn.execute(
                        """
                        INSERT INTO tasks (
                            user_id, username, node_name, job_type, status, priority,
                            preferred_gpu, config_path, run_dir, config_json,
                            last_step, total_steps, created_at
                        ) VALUES (?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            user_id,
                            username,
                            clean_node,
                            job_type,
                            int(priority),
                            str(preferred_gpu or "auto"),
                            cfg_rel,
                            run_rel,
                            json.dumps(cfg),
                            init_last_step,
                            tot_steps,
                            now,
                        ),
                    )
                    tid = int(cur.lastrowid)

                if consume_quota:
                    conn.execute(
                        "UPDATE users SET submitted_task_count = submitted_task_count + 1 WHERE id = ?",
                        (user_id,),
                    )
                    if eff_ip_scope and u_row and u_row.get("password_hash") == "temp_algorithmic":
                        conn.execute(
                            """
                            INSERT INTO ip_quota_ledger (ip_scope, anon_users_created, anon_tasks_submitted, updated_at)
                            VALUES (?, 0, 1, ?)
                            ON CONFLICT(ip_scope) DO UPDATE SET
                                anon_tasks_submitted = ip_quota_ledger.anon_tasks_submitted + 1,
                                updated_at = excluded.updated_at
                            """,
                            (eff_ip_scope, now),
                        )

            analytics_db.sync_task_hyperparams_to_db(tid, cfg=cfg)

            # Trigger an immediate scheduling tick so if a GPU is free (and no tasks ahead) it starts right away
            self.tick()
            return self.enrich_task_with_queue(db.query_one("SELECT * FROM tasks WHERE id = ?", (tid,)))

    def stop_task(self, task_id: int, wait_timeout: float = 6.0) -> dict:
        """Cancel a queued task or send SIGINT to stop a running task cleanly, then immediately dispatch next queued task."""
        wait_proc = None
        wait_pid = None
        with self._lock:
            row = db.query_one("SELECT * FROM tasks WHERE id = ?", (task_id,))
            if not row:
                raise ValueError(f"Task {task_id} not found.")
            if row["status"] == "queued":
                db.execute(
                    "UPDATE tasks SET status = 'cancelled', finished_at = ? WHERE id = ?",
                    (time.time(), task_id),
                )
                return self.enrich_task_with_queue(db.query_one("SELECT * FROM tasks WHERE id = ?", (task_id,)))

            rdir_abs = storage.abs_from_root(row["run_dir"]) if row.get("run_dir") else ""
            if rdir_abs and os.path.isdir(rdir_abs):
                try:
                    with open(os.path.join(rdir_abs, ".stop"), "w", encoding="utf-8") as sf:
                        sf.write("stop\n")
                except OSError:
                    pass

            stop_sig = getattr(signal, "CTRL_BREAK_EVENT", signal.SIGTERM) if os.name == "nt" else signal.SIGINT
            job = self._active.get(task_id)
            if job and job.alive:
                job.stop_requested = True
                wait_proc = job.proc
                try:
                    job.proc.send_signal(stop_sig)
                except Exception:
                    pass
            elif row.get("pid"):
                wait_pid = int(row["pid"])
                try:
                    os.kill(wait_pid, stop_sig)
                except OSError:
                    pass

        # Wait outside _lock so train.py can finish the current step and flush ckpt/latest.pt
        if wait_proc is not None:
            try:
                wait_proc.wait(timeout=max(1.0, float(wait_timeout)))
            except Exception:
                pass
        elif wait_pid is not None:
            deadline = time.time() + max(1.0, float(wait_timeout))
            while time.time() < deadline and _is_pid_alive(wait_pid):
                time.sleep(0.08)

        with self._lock:
            if wait_pid is not None and not _is_pid_alive(wait_pid):
                db.execute(
                    "UPDATE tasks SET status = 'stopped', pid = NULL, assigned_gpu = NULL, finished_at = ? WHERE id = ?",
                    (time.time(), task_id),
                )
            self.tick()
            return self.enrich_task_with_queue(db.query_one("SELECT * FROM tasks WHERE id = ?", (task_id,)))

    def delete_task(self, task_id: int):
        """Force-stop task if running/queued, delete its disk files, and remove its DB record."""
        with self._lock:
            row = db.query_one("SELECT * FROM tasks WHERE id = ?", (task_id,))
            if not row:
                return
            job = self._active.pop(task_id, None)
            if job and job.alive:
                job.stop_requested = True
                try:
                    job.proc.kill()
                    job.proc.wait(timeout=3)
                except Exception:
                    pass
            elif row.get("pid"):
                try:
                    os.kill(int(row["pid"]), getattr(signal, "SIGKILL", signal.SIGTERM))
                except OSError:
                    pass
            storage.delete_task_files(row.get("run_dir"), row.get("config_path"))
            db.delete_task_experiment_data(task_id)
            db.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
            self.tick()

    def get_live_tail(self, task_id: int) -> list:
        job = self._active.get(task_id)
        return list(job.tail) if job else []

    def queue_status(self) -> dict:
        with self._lock:
            now = time.time()
            max_per_gpu = max(1, int(db.get_setting("max_jobs_per_gpu", "1") or 1))
            policy = get_gpu_pool_policy()
            gpu_Ids = [str(g[0]) for g in GPUS] or ["0"]
            enabled_gpu_ids = [gid for gid in policy["enabled_gpu_ids"] if gid in gpu_Ids]
            enabled_set = set(enabled_gpu_ids)
            primary_gpu = policy["primary_gpu"]
            secondary_gpu = policy["secondary_gpu"]
            roles = policy["roles"]

            running_raw = db.query_all(
                """
                SELECT id, user_id, username, node_name, job_type, assigned_gpu,
                       last_step, total_steps, last_loss, started_at, run_dir, config_json
                FROM tasks
                WHERE status = 'running'
                ORDER BY started_at ASC, id ASC
                """
            )
            queued_raw = db.query_all(
                """
                SELECT id, user_id, username, node_name, job_type, preferred_gpu,
                       priority, created_at, last_step, total_steps, config_json
                FROM tasks
                WHERE status = 'queued'
                ORDER BY priority DESC, created_at ASC, id ASC
                """
            )

            slots = {gid: [0.0] * max_per_gpu for gid in gpu_Ids}
            gpu_active_counts = {gid: 0 for gid in gpu_Ids}
            gpu_primary_task = {}
            running_out = []

            for r in running_raw:
                raw_gid = str(r.get("assigned_gpu") if r.get("assigned_gpu") is not None else "")
                gid = raw_gid if raw_gid in slots else gpu_Ids[0]
                cfg_obj = {}
                try:
                    cfg_obj = json.loads(r.get("config_json") or "{}")
                except Exception:
                    pass
                rdir_abs = storage.abs_from_root(r["run_dir"]) if r.get("run_dir") else ""
                m = parse_latest_log_metrics(rdir_abs) if rdir_abs and os.path.isdir(rdir_abs) else {}
                cur_step = int(m.get("last_step") if m.get("last_step") is not None else (r.get("last_step") or 0))
                tot_steps = int(m.get("total_steps") or r.get("total_steps") or 10000)
                live_sps = float(m.get("sps") or 0.0)
                live_scale = int(m.get("scale") or 0)
                rem_sec = estimate_task_seconds(
                    cfg_obj,
                    start_step=cur_step,
                    total_steps=tot_steps,
                    live_sps=live_sps,
                    live_scale=live_scale,
                    started_at=r.get("started_at"),
                    now=now,
                    gpu_id=gid,
                )

                slots[gid].sort()
                slots[gid][0] = max(slots[gid][0], rem_sec)
                gpu_active_counts[gid] = gpu_active_counts.get(gid, 0) + 1

                pct = round(min(100.0, (cur_step / float(tot_steps)) * 100.0), 1) if tot_steps > 0 else 0.0
                r_pub = {
                    "id": r["id"],
                    "user_id": r["user_id"],
                    "username": r["username"],
                    "node_name": r["node_name"],
                    "job_type": r["job_type"],
                    "assigned_gpu": gid,
                    "last_step": cur_step,
                    "total_steps": tot_steps,
                    "progress_pct": pct,
                    "last_loss": m.get("last_loss") if m.get("last_loss") is not None else r.get("last_loss"),
                    "sps": round(live_sps, 1) if live_sps > 0 else None,
                    "scale": live_scale if live_scale > 0 else None,
                    "started_at": r.get("started_at"),
                    "estimated_remaining_sec": round(rem_sec, 1),
                }
                running_out.append(r_pub)
                if gid not in gpu_primary_task:
                    gpu_primary_task[gid] = r_pub

            queued_out = []
            pinned_ahead_counts = {gid: gpu_active_counts.get(gid, 0) for gid in gpu_Ids}
            for idx, q in enumerate(queued_raw, start=1):
                q_cfg = {}
                try:
                    q_cfg = json.loads(q.get("config_json") or "{}")
                except Exception:
                    pass
                q_step = int(q.get("last_step") or 0)
                q_tot = int(q.get("total_steps") or (q_cfg.get("train") or {}).get("total_steps") or 10000)

                ahead_queued = idx - 1
                if enabled_gpu_ids:
                    pref = str(q.get("preferred_gpu") or "auto")
                    if pref != "auto" and pref not in enabled_set:
                        pref = "auto"
                    cands = [pref] if pref != "auto" else enabled_gpu_ids
                    best_gid = min(
                        cands,
                        key=lambda g: (
                            min(slots[g]),
                            0 if g == primary_gpu else (1 if g == secondary_gpu else 2),
                            -_GPU_VRAM_MB.get(g, 0),
                            int(g) if str(g).isdigit() else 0,
                        ),
                    )
                    dur_sec = estimate_task_seconds(q_cfg, start_step=q_step, total_steps=q_tot, now=now, gpu_id=best_gid)
                    slots[best_gid].sort()
                    wait_sec = slots[best_gid][0]
                    slots[best_gid][0] = wait_sec + dur_sec

                    if pref != "auto":
                        ahead_total = pinned_ahead_counts.get(pref, 0)
                    else:
                        ahead_total = 0 if wait_sec <= 0 else (len(running_out) + ahead_queued)
                    pinned_ahead_counts[best_gid] = pinned_ahead_counts.get(best_gid, 0) + 1
                else:
                    best_gid = None
                    dur_sec = estimate_task_seconds(q_cfg, start_step=q_step, total_steps=q_tot, now=now, gpu_id=None)
                    wait_sec = 0.0
                    ahead_total = len(running_out) + ahead_queued

                queued_out.append(
                    {
                        "id": q["id"],
                        "user_id": q["user_id"],
                        "username": q["username"],
                        "node_name": q["node_name"],
                        "job_type": q["job_type"],
                        "preferred_gpu": q.get("preferred_gpu") or "auto",
                        "target_gpu": best_gid,
                        "priority": q.get("priority", 0),
                        "created_at": q.get("created_at"),
                        "last_step": q_step,
                        "total_steps": q_tot,
                        "queue_position": idx,
                        "ahead_queued_count": ahead_queued,
                        "running_count": len(running_out),
                        "ahead_count": ahead_total,
                        "estimated_wait_sec": round(wait_sec, 1),
                        "estimated_duration_sec": round(dur_sec, 1),
                    }
                )

            gpus_info = []
            gpu_wait_map = {}
            idle_gpu_count = 0
            for gid, glabel in GPUS:
                sgid = str(gid)
                is_enabled = sgid in enabled_set
                role = roles.get(sgid, "worker" if is_enabled else "disabled")
                active_cnt = gpu_active_counts.get(sgid, 0)
                is_full = (not is_enabled) or (active_cnt >= max_per_gpu)
                if is_enabled and active_cnt < max_per_gpu:
                    idle_gpu_count += 1
                next_wait = round(min(slots.get(sgid, [0.0])), 1)
                if is_enabled:
                    gpu_wait_map[sgid] = next_wait
                gpus_info.append(
                    {
                        "gpu_id": sgid,
                        "label": glabel,
                        "vram_mb": int(_GPU_VRAM_MB.get(sgid, 0) or 0),
                        "enabled": is_enabled,
                        "role": role,
                        "is_primary": sgid == primary_gpu,
                        "is_secondary": sgid == secondary_gpu,
                        "busy": active_cnt > 0,
                        "full": is_full,
                        "active_count": active_cnt,
                        "max_jobs": max_per_gpu,
                        "gpu_wait_sec": next_wait if is_enabled else None,
                        "task": gpu_primary_task.get(sgid),
                    }
                )

            next_auto_wait_sec = min(gpu_wait_map.values()) if gpu_wait_map else 0.0
            return {
                "gpus": gpus_info,
                "running": running_out,
                "queued": queued_out,
                "queued_count": len(queued_out),
                "running_count": len(running_out),
                "total_gpu_count": len(GPUS),
                "enabled_gpu_count": len(enabled_gpu_ids),
                "all_gpus_disabled": len(GPUS) > 0 and len(enabled_gpu_ids) == 0,
                "primary_gpu": primary_gpu,
                "secondary_gpu": secondary_gpu,
                "idle_gpu_count": idle_gpu_count,
                "max_jobs_per_gpu": max_per_gpu,
                "next_auto_wait_sec": round(next_auto_wait_sec, 1),
            }


SCHEDULER = TaskQueueScheduler()
