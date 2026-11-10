# -*- coding: utf-8 -*-
"""Multi-User Task Queue & GPU Scheduler for WildCat Core Compute Engine."""
import collections
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time

from . import db, storage, telemetry

ROOT = db.ROOT
PYTHON_BIN = sys.executable if (sys.executable and os.path.isfile(sys.executable)) else os.path.join(ROOT, ".venv", "bin", "python")


def detect_gpus():
    """Return list of (gpu_id_str, label_str)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        gpus = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                idx, name = parts[0], parts[1]
                mem = parts[2] if len(parts) >= 3 else ""
                gpus.append((idx, f"CUDA:{idx} — {name} ({mem})".strip()))
        if gpus:
            return gpus
    except Exception:
        pass
    return [("0", "CUDA:0"), ("1", "CUDA:1")]


GPUS = detect_gpus()


def parse_latest_log_metrics(run_dir_abs: str) -> dict:
    """Quickly parse latest step, total_steps, and loss from train.log."""
    lp = os.path.join(run_dir_abs, "train.log")
    if not os.path.isfile(lp):
        return {}
    parsed = telemetry.parse_train_log(lp, max_bytes=32768)
    latest = parsed.get("latest") or {}
    out = {}
    if "step" in latest:
        out["last_step"] = int(latest["step"])
    if "total" in latest:
        out["last_loss"] = float(latest["total"])
    if "sps" in latest:
        out["sps"] = float(latest["sps"])
    info_p = os.path.join(run_dir_abs, "run_info.json")
    if os.path.isfile(info_p):  # written by the trainer: real total steps for this run
        try:
            with open(info_p) as f:
                info = json.load(f)
            if info.get("total_steps"):
                out["total_steps"] = int(info["total_steps"])
        except Exception:
            pass
    return out


class _ActiveJob:
    def __init__(self, task_id: int, node_name: str, gpu: str, proc: subprocess.Popen):
        self.task_id = task_id
        self.node_name = node_name
        self.gpu = str(gpu)
        self.proc = proc
        self.tail = collections.deque(maxlen=150)
        self.stop_requested = False
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()

    def _read_stdout(self):
        try:
            for line in self.proc.stdout:
                self.tail.append(line.rstrip("\n"))
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

    def _reconcile_on_startup(self):
        """Sync existing disk runs into DB and reset stale 'running' tasks if process is dead."""
        db.init_db()
        admin = db.query_one("SELECT id, username FROM users WHERE username = 'admin'")
        if not admin:
            return
        admin_id = admin["id"]

        # Seed default shared paired datasets into admin's dataset library
        from . import service as _svc

        _svc.seed_demo_assets_for_user(admin_id)

        # Register existing run directories under run/ into tasks table if not yet tracked
        now = time.time()
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
                        with open(cfg_file, "r") as f:
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

        # Reset any orphaned 'running' tasks whose PID is no longer alive
        running_rows = db.query_all("SELECT id, pid, run_dir FROM tasks WHERE status = 'running'")
        for r in running_rows:
            pid = r.get("pid")
            alive = False
            if pid:
                try:
                    os.kill(int(pid), 0)
                    alive = True
                except OSError:
                    alive = False
            if not alive:
                db.execute(
                    "UPDATE tasks SET status = 'stopped', pid = NULL, assigned_gpu = NULL WHERE id = ?",
                    (r["id"],),
                )

    def _loop(self):
        while self._running:
            try:
                self.tick()
            except Exception:
                pass
            time.sleep(1.0)

    def tick(self):
        with self._lock:
            now = time.time()
            # 1. Check active jobs and update live metrics
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
                    rc = job.proc.poll()
                    if job.stop_requested or rc == -signal.SIGINT or rc == 130:
                        new_status = "stopped"
                        err = ""
                    elif rc == 0:
                        new_status = "completed"
                        err = ""
                    else:
                        new_status = "failed"
                        err = "\n".join(list(job.tail)[-10:])
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

            # 2. Determine GPU occupancy
            max_per_gpu = max(1, int(db.get_setting("max_jobs_per_gpu", "1") or 1))
            gpu_Ids = [g[0] for g in GPUS]
            gpu_counts = {gid: 0 for gid in gpu_Ids}
            for job in self._active.values():
                if job.alive and job.gpu in gpu_counts:
                    gpu_counts[job.gpu] += 1

            # 3. Dispatch queued tasks in priority + FIFO order
            queued = db.query_all(
                "SELECT * FROM tasks WHERE status = 'queued' ORDER BY priority DESC, created_at ASC, id ASC"
            )
            for q in queued:
                pref = str(q.get("preferred_gpu") or "auto")
                chosen_gpu = None
                if pref == "auto":
                    # Prefer GPU 1 first if available (leaves GPU 0 lighter), else any available GPU
                    ordered_candidates = sorted(
                        gpu_Ids, key=lambda g: (gpu_counts.get(g, 0), 0 if g == "1" else 1)
                    )
                    for gid in ordered_candidates:
                        if gpu_counts.get(gid, 0) < max_per_gpu:
                            chosen_gpu = gid
                            break
                elif pref in gpu_counts:
                    if gpu_counts[pref] < max_per_gpu:
                        chosen_gpu = pref

                if chosen_gpu is not None:
                    self._launch_queued_task(q, chosen_gpu)
                    gpu_counts[chosen_gpu] = gpu_counts.get(chosen_gpu, 0) + 1

    def _launch_queued_task(self, task_row: dict, gpu_id: str):
        tid = task_row["id"]
        jtype = task_row["job_type"]
        cfg_rel = task_row["config_path"]
        cfg_obj = {}
        try:
            cfg_obj = json.loads(task_row.get("config_json") or "{}")
        except Exception:
            pass

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["PYTHONUNBUFFERED"] = "1"
        # Preview / validation on a secondary GPU: expose it as cuda:1 to the training process.
        # Prefer a GPU that is not running another training job; otherwise stay on the same card.
        if jtype != "infer":
            pv = str(((cfg_obj.get("optimizations") or {}).get("lossless") or {}).get("preview_gpu", "auto"))
            busy = {j.gpu for j in self._active.values() if j.alive}
            others = [g[0] for g in GPUS if g[0] != str(gpu_id)]
            pick = None
            if pv == "auto":
                idle = [g for g in others if g not in busy]
                pick = idle[0] if idle else None
            elif pv not in ("same", "off", "cpu", "") and pv in others:
                pick = pv
            if pick is not None:
                env["CUDA_VISIBLE_DEVICES"] = "%s,%s" % (gpu_id, pick)
                env["WILDCAT_PREVIEW_GPU"] = "1"
            elif pv == "auto":
                env["WILDCAT_PREVIEW_GPU"] = "same"

        if jtype == "infer":
            infer_seq = cfg_obj.get("_infer_seq", "dataset/eval")
            cmd = [PYTHON_BIN, "-u", "infer.py", cfg_rel, infer_seq]
        else:
            cmd = [PYTHON_BIN, "-u", "train.py", cfg_rel]

        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        job = _ActiveJob(tid, task_row["node_name"], str(gpu_id), proc)
        self._active[tid] = job
        db.execute(
            """
            UPDATE tasks
            SET status = 'running', assigned_gpu = ?, pid = ?, started_at = ?, error_msg = ''
            WHERE id = ?
            """,
            (str(gpu_id), proc.pid, time.time(), tid),
        )

    def submit_task(
        self,
        user_id: int,
        username: str,
        node_name: str,
        job_type: str,
        cfg: dict,
        preferred_gpu: str = "auto",
        infer_seq: str = None,
        priority: int = 0,
    ) -> dict:
        """Create or update a task node and place it into the Multi-User Task Queue."""
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

            if existing and existing["status"] in ("running", "queued"):
                raise RuntimeError(f"任务 '{clean_node}' 当前已在运行或排队中。")

            cfg = dict(cfg)
            cfg["run_name"] = clean_node
            if infer_seq:
                cfg["_infer_seq"] = infer_seq

            run_rel = f"run/{clean_node}"
            run_abs = storage.abs_from_root(run_rel)
            is_resume = bool(cfg.get("train", {}).get("resume", False))

            if job_type == "train" and not is_resume and os.path.isfile(os.path.join(run_abs, "ckpt", "latest.pt")):
                raise ValueError(
                    f"任务名称 '{clean_node}' 已有训练记录。新建任务请更换新的任务名；若要继续训练原任务，请在左侧任务列表点击 '{clean_node}' 后选择『继续训练』。"
                )

            # Enforce locked model structure when initial_weights or resume checkpoint is specified
            from wildcat import models

            tr_cfg = cfg.setdefault("train", {})
            mo_cfg = cfg.setdefault("model", {})
            init_w = tr_cfg.get("initial_weights", "")
            if is_resume:
                cand_pt = os.path.join(run_abs, "ckpt", "latest.pt")
                if os.path.isfile(cand_pt):
                    init_w = cand_pt
            if init_w and init_w != "None":
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

            os.makedirs(storage.CONFIG_ROOT, exist_ok=True)
            cfg_rel = f"config/{clean_node}.json"
            cfg_abs = storage.abs_from_root(cfg_rel)
            with open(cfg_abs, "w") as f:
                json.dump(cfg, f, indent=2)

            os.makedirs(run_abs, exist_ok=True)
            with open(os.path.join(run_abs, f"{clean_node}.json"), "w") as f:
                json.dump(cfg, f, indent=2)

            tot_steps = int(cfg.get("train", {}).get("total_steps", 10000))
            now = time.time()

            if existing:
                tid = existing["id"]
                db.execute(
                    """
                    UPDATE tasks
                    SET job_type = ?, status = 'queued', priority = ?, preferred_gpu = ?,
                        assigned_gpu = NULL, pid = NULL, config_path = ?, run_dir = ?,
                        config_json = ?, total_steps = ?, error_msg = '', created_at = ?
                    WHERE id = ?
                    """,
                    (
                        job_type,
                        int(priority),
                        str(preferred_gpu or "auto"),
                        cfg_rel,
                        run_rel,
                        json.dumps(cfg),
                        tot_steps,
                        now,
                        tid,
                    ),
                )
            else:
                tid = db.execute(
                    """
                    INSERT INTO tasks (
                        user_id, username, node_name, job_type, status, priority,
                        preferred_gpu, config_path, run_dir, config_json,
                        last_step, total_steps, created_at
                    ) VALUES (?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?, 0, ?, ?)
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
                        tot_steps,
                        now,
                    ),
                )

            # Trigger an immediate scheduling tick so if a GPU is free it starts right away
            self.tick()
            return db.query_one("SELECT * FROM tasks WHERE id = ?", (tid,))

    def save_draft_node(self, user_id: int, username: str, node_name: str, cfg: dict) -> dict:
        """Save or update a node configuration without queuing execution."""
        with self._lock:
            clean_node = storage.sanitize_name(node_name, "WildCat1")
            own = db.query_one("SELECT * FROM tasks WHERE user_id = ? AND node_name = ?", (user_id, clean_node))
            if own:
                existing = own
            else:
                other = db.query_one("SELECT * FROM tasks WHERE node_name = ? AND user_id != ?", (clean_node, user_id))
                if other or os.path.exists(storage.abs_from_root(f"run/{clean_node}")):
                    clean_node = f"{storage.sanitize_name(username)}_{clean_node}"
                existing = db.query_one("SELECT * FROM tasks WHERE user_id = ? AND node_name = ?", (user_id, clean_node))

            cfg = dict(cfg)
            cfg["run_name"] = clean_node
            os.makedirs(storage.CONFIG_ROOT, exist_ok=True)
            cfg_rel = f"config/{clean_node}.json"
            with open(storage.abs_from_root(cfg_rel), "w") as f:
                json.dump(cfg, f, indent=2)
            run_rel = f"run/{clean_node}"
            run_abs = storage.abs_from_root(run_rel)
            if os.path.isdir(run_abs):
                with open(os.path.join(run_abs, f"{clean_node}.json"), "w") as f:
                    json.dump(cfg, f, indent=2)

            tot_steps = int(cfg.get("train", {}).get("total_steps", 10000))
            now = time.time()
            if existing:
                db.execute(
                    "UPDATE tasks SET config_path = ?, config_json = ?, total_steps = ? WHERE id = ?",
                    (cfg_rel, json.dumps(cfg), tot_steps, existing["id"]),
                )
                return db.query_one("SELECT * FROM tasks WHERE id = ?", (existing["id"],))
            else:
                tid = db.execute(
                    """
                    INSERT INTO tasks (
                        user_id, username, node_name, job_type, status, priority,
                        preferred_gpu, config_path, run_dir, config_json,
                        last_step, total_steps, created_at
                    ) VALUES (?, ?, ?, 'train', 'idle', 0, 'auto', ?, ?, ?, 0, ?, ?)
                    """,
                    (user_id, username, clean_node, cfg_rel, run_rel, json.dumps(cfg), tot_steps, now),
                )
                return db.query_one("SELECT * FROM tasks WHERE id = ?", (tid,))

    def stop_task(self, task_id: int) -> dict:
        """Cancel a queued task or send SIGINT to stop a running task."""
        with self._lock:
            row = db.query_one("SELECT * FROM tasks WHERE id = ?", (task_id,))
            if not row:
                raise ValueError(f"Task {task_id} not found.")
            if row["status"] == "queued":
                db.execute(
                    "UPDATE tasks SET status = 'cancelled', finished_at = ? WHERE id = ?",
                    (time.time(), task_id),
                )
                return db.query_one("SELECT * FROM tasks WHERE id = ?", (task_id,))

            job = self._active.get(task_id)
            if job and job.alive:
                job.stop_requested = True
                try:
                    job.proc.send_signal(signal.SIGINT)
                except Exception:
                    pass
            elif row.get("pid"):
                try:
                    os.kill(int(row["pid"]), signal.SIGINT)
                except OSError:
                    pass
                db.execute(
                    "UPDATE tasks SET status = 'stopped', pid = NULL, assigned_gpu = NULL, finished_at = ? WHERE id = ?",
                    (time.time(), task_id),
                )
            return db.query_one("SELECT * FROM tasks WHERE id = ?", (task_id,))

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
                    os.kill(int(row["pid"]), signal.SIGKILL)
                except OSError:
                    pass
            storage.delete_task_files(row.get("run_dir"), row.get("config_path"))
            db.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
            self.tick()

    def get_live_tail(self, task_id: int) -> list:
        job = self._active.get(task_id)
        return list(job.tail) if job else []

    def queue_status(self) -> dict:
        with self._lock:
            running = db.query_all(
                "SELECT id, user_id, username, node_name, job_type, assigned_gpu, last_step, total_steps, last_loss, started_at FROM tasks WHERE status = 'running' ORDER BY started_at ASC"
            )
            queued = db.query_all(
                "SELECT id, user_id, username, node_name, job_type, preferred_gpu, priority, created_at FROM tasks WHERE status = 'queued' ORDER BY priority DESC, created_at ASC, id ASC"
            )
            for idx, q in enumerate(queued, start=1):
                q["queue_position"] = idx
            gpus_info = []
            busy_map = {}
            for r in running:
                if r.get("assigned_gpu") is not None:
                    busy_map[str(r["assigned_gpu"])] = r
            for gid, glabel in GPUS:
                b = busy_map.get(str(gid))
                gpus_info.append(
                    {
                        "gpu_id": gid,
                        "label": glabel,
                        "busy": b is not None,
                        "task": b,
                    }
                )
            return {
                "gpus": gpus_info,
                "running": running,
                "queued": queued,
                "queued_count": len(queued),
                "running_count": len(running),
            }


SCHEDULER = TaskQueueScheduler()
