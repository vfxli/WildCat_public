# -*- coding: utf-8 -*-
"""
Unified Priority Validation Sequence & Single-Frame Inference Runner (`backend/infer_runner.py`).

All manual inference jobs (single-frame or full-sequence) and post-training auto-inference jobs (`auto_on_finish`)
share the single GPU queue and slot allocator in `backend/scheduler.py`:
  - Ready inference jobs (`status = 'queued'`) have strictly higher dispatch priority than queued training tasks,
    and multiple inference jobs queue in FIFO order without ever colliding on a busy GPU.
  - Post-training auto-inference jobs are created when a training task is submitted (`status = 'waiting_train'`,
    `depends_on_task_id = task.id`), automatically promote to `'queued'` the instant the parent training task
    completes, and automatically cancel if the parent training task is cancelled, stopped, failed, or deleted.
"""
import collections
import glob
import hashlib
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import zipfile

from wildcat import color

from . import analytics_db, db, i18n, scheduler, service, storage, telemetry

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INFER_SUBDIR = "infer_seq"
META_FILE = "infer_meta.json"
VALID_FORMATS = ("exr_float", "exr_half")
_SAFE_OUT_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,180}$")
_JOB_KEEP_SEC = 3600.0


class InferBusyError(RuntimeError):
    """Kept for API exception compatibility; inference jobs now queue cleanly."""


class InferNotFoundError(LookupError):
    """Requested asset, task, job or inference output was not found."""


class ActiveInferSubproc:
    """In-memory handle for a currently running `infer.py --spec` subprocess on an assigned GPU."""

    def __init__(
        self,
        job_id: str,
        task_id: int,
        node_name: str,
        gpu: str,
        gpu_label: str,
        proc: subprocess.Popen,
        tmp_dir: str,
        final_dir: str,
        prog_path: str,
        cancel_path: str,
        eval_out_dir: str = "",
        eval_key: str = "",
    ):
        self.job_id = job_id
        self.task_id = int(task_id)
        self.node_name = node_name
        self.gpu = str(gpu)
        self.gpu_label = gpu_label
        self.proc = proc
        self.tmp_dir = tmp_dir
        self.final_dir = final_dir
        self.prog_path = prog_path
        self.cancel_path = cancel_path
        self.eval_out_dir = eval_out_dir
        self.eval_key = eval_key
        self.stop_requested = False
        self.tail = collections.deque(maxlen=120)
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


def estimate_infer_seconds(total_frames: int, infer_mode: str = "sequence", done_frames: int = 0, fps: float = 0.0) -> float:
    rem = max(0, int(total_frames or 0) - int(done_frames or 0))
    if rem <= 0:
        return 0.5
    if fps and float(fps) > 0.05:
        return max(0.5, rem / float(fps))
    if infer_mode == "single":
        return 1.5
    return max(2.0, 1.5 + rem * 0.08)


def resolve_task_checkpoint(task_row: dict, step: int | None = None) -> tuple[str | None, int]:
    """Resolve `(ckpt_abs_path, resolved_step)` for a trained task; returns `(None, 0)` if untrained."""
    run_dir_rel = str(task_row.get("run_dir") or "").strip()
    if not run_dir_rel:
        return None, 0
    rdir_abs = storage.abs_from_root(run_dir_rel)
    if not os.path.isdir(rdir_abs):
        return None, 0

    ckpt_dir = os.path.join(rdir_abs, "ckpt")
    req_step = int(step) if (step is not None and int(step) > 0) else 0

    if req_step > 0:
        cand_pt = os.path.join(ckpt_dir, f"step_{req_step:06d}.pt")
        if os.path.isfile(cand_pt) and os.path.getsize(cand_pt) > 0:
            return cand_pt, req_step
        for p in sorted(glob.glob(os.path.join(rdir_abs, f"Training_*.{req_step}.pt"))):
            if os.path.isfile(p) and os.path.getsize(p) > 0:
                return p, req_step
        lat = os.path.join(ckpt_dir, "latest.pt")
        if os.path.isfile(lat) and os.path.getsize(lat) > 0:
            sp = telemetry.inspect_ckpt_cached(lat)
            if int(sp.get("step") or 0) == req_step:
                return lat, req_step
        return None, 0

    lat = os.path.join(ckpt_dir, "latest.pt")
    if os.path.isfile(lat) and os.path.getsize(lat) > 0:
        sp = telemetry.inspect_ckpt_cached(lat)
        st = int(sp.get("step") or task_row.get("last_step") or 0)
        if st > 0:
            return lat, st

    step_pts = sorted(glob.glob(os.path.join(ckpt_dir, "step_*.pt")))
    if step_pts:
        sp = telemetry.inspect_ckpt_cached(step_pts[-1])
        st = int(sp.get("step") or 0)
        if st <= 0:
            m = re.search(r"step_0*(\d+)\.pt$", os.path.basename(step_pts[-1]))
            st = int(m.group(1)) if m else 0
        if st > 0:
            return step_pts[-1], st

    cfg_obj = task_row.get("config")
    if not isinstance(cfg_obj, dict):
        try:
            cfg_obj = json.loads(task_row.get("config_json") or "{}")
        except Exception:
            cfg_obj = {}
    parsed = telemetry.parse_train_log_full_cached(os.path.join(rdir_abs, "train.log"))
    inv = telemetry.list_task_models(run_dir_rel, cfg_obj or {}, parsed_log=parsed)
    for it in inv.get("items") or []:
        st = int(it.get("step") or 0)
        if st <= 0:
            continue
        for p in sorted(glob.glob(os.path.join(rdir_abs, f"Training_*.{st}.pt"))):
            if os.path.isfile(p) and os.path.getsize(p) > 0:
                return p, st

    return None, 0


def _resolve_infer_asset(
    user_id: int,
    is_admin: bool,
    asset_id: int,
    allow_paired_source: bool = False,
) -> tuple[dict, str, list[str]]:
    row = db.query_one("SELECT * FROM assets WHERE id = ?", (int(asset_id),))
    if not row:
        raise InferNotFoundError(i18n.t("server.assetNotFound"))
    if not is_admin and int(row["user_id"]) != int(user_id):
        raise PermissionError(i18n.t("server.assetAccessForbidden"))

    if row.get("kind") == "eval_sequence":
        seq_rel = str(row.get("rel_path") or "").strip()
    else:
        seq_rel = str(row.get("seq_path") or "").strip()
        if (not seq_rel or not storage.expand_sequence_files(seq_rel)) and allow_paired_source:
            seq_rel = str(row.get("source_path") or "").strip()

    if not seq_rel:
        raise ValueError(i18n.t("server.inferNoEvalSeq", name=row["name"]))
    seq_files = storage.expand_sequence_files(seq_rel)
    if not seq_files:
        raise ValueError(i18n.t("server.inferNoEvalSeq", name=row["name"]))
    return row, seq_rel, seq_files


def _task_infer_primary_root(task_row: dict) -> str:
    rdir_abs = storage.abs_from_root(task_row["run_dir"])
    task_sub = storage.sanitize_name(os.path.basename(rdir_abs) or task_row["node_name"], f"task_{task_row['id']}")
    return os.path.join(storage.get_infer_dir(), task_sub)


def _task_infer_candidate_roots(task_row: dict) -> list[str]:
    rdir_abs = storage.abs_from_root(task_row["run_dir"])
    task_sub = storage.sanitize_name(os.path.basename(rdir_abs) or task_row["node_name"], f"task_{task_row['id']}")
    roots = []
    for cand in (
        os.path.join(storage.get_infer_dir(), task_sub),
        os.path.join(storage.DEFAULT_INFER_DIR, task_sub),
        os.path.join(rdir_abs, INFER_SUBDIR),
    ):
        ap = os.path.abspath(cand)
        if ap not in roots:
            roots.append(ap)
    return roots


def _locate_output_dir(task_row: dict, out_id: str) -> str | None:
    if not out_id or not _SAFE_OUT_ID_RE.match(out_id) or out_id.startswith("."):
        return None
    for root in _task_infer_candidate_roots(task_row):
        cand = os.path.abspath(os.path.join(root, out_id))
        if not os.path.normcase(cand).startswith(os.path.normcase(root + os.sep)):
            continue
        if os.path.isfile(os.path.join(cand, META_FILE)) and os.path.isdir(os.path.join(cand, "frames")):
            return cand
    return None


def _format_short_tag(output_format: str) -> str:
    if output_format == "exr_half":
        return "exr16f"
    return "exr32f"


def _compute_seq_eval_key(step: int, asset_id: int | None, seq_rel: str, seq_files: list[str]) -> str:
    if not seq_files:
        return f"step{int(step)}_sup_only"
    try:
        mt = int(os.path.getmtime(seq_files[0]))
    except OSError:
        mt = 0
    sig = hashlib.sha1(f"{seq_rel}|{len(seq_files)}|{mt}".encode("utf-8")).hexdigest()[:8]
    aid = int(asset_id) if asset_id is not None else 0
    return f"step{int(step)}_seq{aid}_{sig}"


def _read_output_meta(task_row: dict, out_id: str) -> dict | None:
    out_abs = _locate_output_dir(task_row, out_id)
    if not out_abs:
        return None
    meta_path = os.path.join(out_abs, META_FILE)
    frames_dir = os.path.join(out_abs, "frames")
    try:
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
    except Exception:
        return None
    files = storage.expand_sequence_files(frames_dir)
    if not files:
        return None
    size_bytes = sum(os.path.getsize(p) for p in files if os.path.isfile(p))
    return {
        "out_id": out_id,
        "task_id": int(task_row["id"]),
        "node_name": task_row["node_name"],
        "model_size": str(meta.get("model_size") or ""),
        "in_channels": int(meta.get("in_channels") or 3),
        "out_channels": int(meta.get("out_channels") or 3),
        "step": int(meta.get("step") or 0),
        "infer_mode": str(meta.get("infer_mode") or ("single" if len(files) == 1 and "_single_" in out_id else "sequence")),
        "frame_index": meta.get("frame_index"),
        "asset_id": meta.get("asset_id"),
        "asset_name": str(meta.get("asset_name") or ""),
        "input_rel": str(meta.get("input_rel") or ""),
        "output_rel": storage.rel_to_root(frames_dir),
        "output_format": str(meta.get("output_format") or "exr_float"),
        "frame_count": len(files),
        "width": int(meta.get("width") or 0),
        "height": int(meta.get("height") or 0),
        "seconds": round(float(meta.get("seconds") or 0.0), 2),
        "fps": round(float(meta.get("fps") or 0.0), 2),
        "size_mb": round(size_bytes / (1024.0 * 1024.0), 2),
        "created_at": float(meta.get("created_at") or os.path.getmtime(meta_path)),
    }


def _public_job_from_row(row: dict, q_lookup: dict | None = None) -> dict:
    done = int(row.get("done_frames") or 0)
    tot = int(row.get("total_frames") or 0)
    st = str(row.get("status") or "queued")
    started = row.get("started_at")
    finished = row.get("finished_at")
    out_obj = None
    raw_out = str(row.get("output_json") or "").strip()
    if raw_out:
        try:
            out_obj = json.loads(raw_out)
        except Exception:
            out_obj = None
    if out_obj is None and st == "done" and row.get("out_id") and row.get("task_id"):
        t_row = db.query_one("SELECT * FROM tasks WHERE id = ?", (int(row["task_id"]),))
        if t_row:
            out_obj = _read_output_meta(t_row, str(row["out_id"]))

    q_meta = (q_lookup or {}).get(str(row["job_id"])) or {}
    fps_val = float(row.get("fps") or 0.0)
    rem_sec = (
        round(estimate_infer_seconds(tot, str(row.get("infer_mode") or "sequence"), done_frames=done, fps=fps_val), 1)
        if st == "running"
        else None
    )
    return {
        "job_id": row["job_id"],
        "user_id": int(row["user_id"]),
        "username": str(row.get("username") or ""),
        "task_id": int(row["task_id"]),
        "node_name": str(row["node_name"]),
        "depends_on_task_id": row.get("depends_on_task_id"),
        "auto_on_finish": bool(row.get("auto_on_finish")),
        "asset_id": row.get("asset_id"),
        "asset_name": str(row.get("asset_name") or ""),
        "step": int(row.get("resolved_step") or row.get("req_step") or 0),
        "infer_mode": str(row.get("infer_mode") or "sequence"),
        "frame_index": row.get("frame_index"),
        "output_format": str(row.get("output_format") or "exr_float"),
        "state": st,
        "done": done,
        "total": tot,
        "progress": round(done / float(max(1, tot)), 4) if tot > 0 else 0.0,
        "fps": round(fps_val, 2),
        "device": str(row.get("device") or (f"CUDA:{row['assigned_gpu']}" if row.get("assigned_gpu") is not None else "")),
        "assigned_gpu": row.get("assigned_gpu"),
        "preferred_gpu": str(row.get("preferred_gpu") or "auto"),
        "queue_position": q_meta.get("queue_position"),
        "ahead_count": q_meta.get("ahead_count", 0),
        "target_gpu": q_meta.get("target_gpu"),
        "estimated_wait_sec": q_meta.get("estimated_wait_sec", 0.0),
        "estimated_remaining_sec": rem_sec,
        "seconds": round((float(finished) if finished else time.time()) - float(started), 1) if started else 0.0,
        "created_at": row.get("created_at"),
        "error": str(row.get("error_msg") or ""),
        "output": out_obj,
    }


def _prune_old_finished_jobs(now: float | None = None) -> None:
    ts = float(now if now is not None else time.time())
    db.execute(
        "DELETE FROM infer_jobs WHERE status IN ('done', 'cancelled', 'error') AND finished_at IS NOT NULL AND (? - finished_at) > ?",
        (ts, _JOB_KEEP_SEC),
    )


def enqueue_auto_infer_for_task(
    task_id: int,
    user_id: int,
    username: str,
    node_name: str,
    cfg: dict,
    matched_ds: dict | None,
    eval_seq_rel: str,
    tot_steps: int,
    preferred_gpu: str = "auto",
    now: float | None = None,
) -> dict | None:
    """Bind or cancel a `'waiting_train'` post-training sequence inference job when submitting a training task."""
    ts = float(now if now is not None else time.time())
    tid = int(task_id)
    # Cancel any stale waiting_train auto-infer job previously bound to this task
    db.execute(
        "UPDATE infer_jobs SET status = 'cancelled', finished_at = ? WHERE depends_on_task_id = ? AND status = 'waiting_train'",
        (ts, tid),
    )

    io_cfg = (cfg or {}).get("io") or {}
    if io_cfg.get("infer_on_finish", True) in (False, "false", "0", "off", 0):
        return None
    seq_rel = str(eval_seq_rel or "").strip()
    if not seq_rel:
        return None
    seq_files = storage.expand_sequence_files(seq_rel)
    if len(seq_files) < 2:
        return None

    exp_meta = (cfg or {}).get("experiment") or {}
    asset_id = None
    asset_name = ""
    if matched_ds:
        asset_id = int(matched_ds["id"])
        asset_name = str(matched_ds.get("name") or "")
    else:
        raw_aid = exp_meta.get("eval_asset_id") or exp_meta.get("dataset_id")
        if raw_aid is not None and str(raw_aid).strip() != "":
            try:
                asset_id = int(raw_aid)
            except (TypeError, ValueError):
                asset_id = None
        asset_name = str(exp_meta.get("dataset_name") or "").strip()
    if not asset_name:
        parent_dir = os.path.basename(os.path.dirname(seq_rel.rstrip("/\\")))
        asset_name = f"{parent_dir}_dataset" if parent_dir and not parent_dir.endswith("_dataset") else (parent_dir or "seq")

    clean_asset = storage.sanitize_name(asset_name, "seq")[:80]
    est_step = max(1, int(tot_steps or 1))
    out_id = f"{clean_asset}_step{est_step}_exr32f"
    job_id = secrets.token_hex(8)
    pref = str(preferred_gpu or "auto")

    db.execute(
        """
        INSERT INTO infer_jobs (
            job_id, user_id, username, task_id, node_name, depends_on_task_id, auto_on_finish,
            asset_id, asset_name, seq_rel, req_step, resolved_step, infer_mode, frame_index,
            output_format, out_id, status, preferred_gpu, total_frames, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?, NULL, ?, 'sequence', NULL, 'exr_float', ?, 'waiting_train', ?, ?, ?)
        """,
        (
            job_id,
            int(user_id),
            str(username or ""),
            tid,
            str(node_name),
            tid,
            asset_id,
            asset_name,
            seq_rel,
            est_step,
            out_id,
            pref,
            len(seq_files),
            ts,
        ),
    )
    row = db.query_one("SELECT * FROM infer_jobs WHERE job_id = ?", (job_id,))
    return _public_job_from_row(row) if row else None


def on_train_task_terminal(task_id: int, final_status: str, now: float | None = None) -> None:
    """Promote bound `'waiting_train'` inference jobs to `'queued'` on completion, or cancel on stop/fail/cancel."""
    ts = float(now if now is not None else time.time())
    tid = int(task_id)
    waiting_rows = db.query_all(
        "SELECT * FROM infer_jobs WHERE depends_on_task_id = ? AND status = 'waiting_train'",
        (tid,),
    )
    if not waiting_rows:
        return

    if final_status != "completed":
        db.execute(
            "UPDATE infer_jobs SET status = 'cancelled', finished_at = ? WHERE depends_on_task_id = ? AND status = 'waiting_train'",
            (ts, tid),
        )
        return

    task_row = db.query_one("SELECT * FROM tasks WHERE id = ?", (tid,))
    if not task_row:
        db.execute(
            "UPDATE infer_jobs SET status = 'cancelled', finished_at = ? WHERE depends_on_task_id = ? AND status = 'waiting_train'",
            (ts, tid),
        )
        return

    ckpt_abs, actual_step = resolve_task_checkpoint(task_row, step=None)
    if not ckpt_abs or actual_step <= 0:
        err = i18n.t("server.inferModelUntrained", name=task_row.get("node_name", str(tid)))
        db.execute(
            "UPDATE infer_jobs SET status = 'error', error_msg = ?, finished_at = ? WHERE depends_on_task_id = ? AND status = 'waiting_train'",
            (err, ts, tid),
        )
        return

    for w in waiting_rows:
        clean_asset = storage.sanitize_name(w.get("asset_name") or "seq", "seq")[:80]
        fmt_tag = _format_short_tag(str(w.get("output_format") or "exr_float"))
        out_id = f"{clean_asset}_step{actual_step}_{fmt_tag}"
        db.execute(
            """
            UPDATE infer_jobs
            SET status = 'queued', ready_at = ?, resolved_step = ?, out_id = ?
            WHERE job_id = ?
            """,
            (ts, int(actual_step), out_id, w["job_id"]),
        )


def reconcile_infer_jobs_on_startup(now: float | None = None) -> None:
    """Reset interrupted `'running'` inference jobs back to `'queued'` and reconcile `'waiting_train'` dependencies."""
    ts = float(now if now is not None else time.time())
    # 1. Interrupted running inference jobs -> clean tmp dir and re-queue
    running_rows = db.query_all("SELECT * FROM infer_jobs WHERE status = 'running'")
    for r in running_rows:
        if not scheduler._is_pid_alive(r.get("pid")):
            t_row = db.query_one("SELECT * FROM tasks WHERE id = ?", (int(r["task_id"]),))
            if t_row:
                tmp_dir = os.path.join(_task_infer_primary_root(t_row), f".tmp_{r['job_id']}")
                if os.path.exists(tmp_dir):
                    shutil.rmtree(tmp_dir, ignore_errors=True)
            db.execute(
                """
                UPDATE infer_jobs
                SET status = 'queued', assigned_gpu = NULL, pid = NULL, device = '',
                    done_frames = 0, fps = 0.0, started_at = NULL, ready_at = COALESCE(ready_at, ?)
                WHERE job_id = ?
                """,
                (ts, r["job_id"]),
            )

    # 2. Reconcile waiting_train rows whose parent training task already finished while offline
    waiting_rows = db.query_all("SELECT DISTINCT depends_on_task_id FROM infer_jobs WHERE status = 'waiting_train' AND depends_on_task_id IS NOT NULL")
    for w in waiting_rows:
        dep_tid = int(w["depends_on_task_id"])
        dep_task = db.query_one("SELECT status FROM tasks WHERE id = ?", (dep_tid,))
        if not dep_task:
            on_train_task_terminal(dep_tid, "cancelled", now=ts)
        elif dep_task["status"] not in ("queued", "running"):
            on_train_task_terminal(dep_tid, str(dep_task["status"]), now=ts)


def spawn_queued_infer_job(job_row: dict, gpu_id: str) -> ActiveInferSubproc:
    """Spawn `infer.py --spec` on `gpu_id` (isolated via `CUDA_VISIBLE_DEVICES`) for a `'queued'` inference job."""
    job_id = str(job_row["job_id"])
    tid = int(job_row["task_id"])
    task_row = db.query_one("SELECT * FROM tasks WHERE id = ?", (tid,))
    if not task_row:
        raise RuntimeError(i18n.t("server.taskNotFound"))

    try:
        task_cfg = json.loads(task_row.get("config_json") or "{}")
    except Exception:
        task_cfg = {}

    req_step = job_row.get("req_step")
    ckpt_abs, actual_step = resolve_task_checkpoint(task_row, step=req_step)
    if not ckpt_abs or actual_step <= 0:
        raise RuntimeError(i18n.t("server.inferModelUntrained", name=task_row["node_name"]))

    seq_rel = str(job_row.get("seq_rel") or "").strip()
    seq_files = storage.expand_sequence_files(seq_rel)
    if not seq_files:
        raise RuntimeError(i18n.t("server.inferNoEvalSeq", name=job_row.get("asset_name") or seq_rel))

    mode = str(job_row.get("infer_mode") or "sequence")
    fmt = str(job_row.get("output_format") or "exr_float")
    fidx = job_row.get("frame_index")
    clean_asset = storage.sanitize_name(job_row.get("asset_name") or "seq", "seq")[:80]
    if mode == "single":
        idx_clamped = 0 if fidx is None else max(0, min(len(seq_files) - 1, int(fidx)))
        real_fnum = storage._extract_frame_num(os.path.basename(seq_files[idx_clamped]))
        ftag = f"f{real_fnum:04d}" if real_fnum is not None else f"f{idx_clamped + 1:04d}"
        out_id = f"{clean_asset}_step{actual_step}_single_{ftag}_{_format_short_tag(fmt)}"
        total_frames = 1
    else:
        idx_clamped = None
        out_id = f"{clean_asset}_step{actual_step}_{_format_short_tag(fmt)}"
        total_frames = len(seq_files)

    infer_root = _task_infer_primary_root(task_row)
    os.makedirs(infer_root, exist_ok=True)
    tmp_dir = os.path.join(infer_root, f".tmp_{job_id}")
    final_dir = os.path.join(infer_root, out_id)
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)
    frames_tmp = os.path.join(tmp_dir, "frames")
    os.makedirs(frames_tmp, exist_ok=True)

    prog_path = os.path.join(tmp_dir, "progress.json")
    cancel_path = os.path.join(tmp_dir, ".cancel")
    spec_path = os.path.join(tmp_dir, "spec.json")

    is_cpu = str(gpu_id).lower() == "cpu"
    gpu_label = "CPU" if is_cpu else f"CUDA:{gpu_id}"
    subproc_dev = "cpu" if is_cpu else "cuda:0"

    asset_work_cs = ""
    if job_row.get("asset_id") is not None:
        arow = db.query_one("SELECT working_colorspace FROM assets WHERE id = ?", (int(job_row["asset_id"]),))
        if arow and arow.get("working_colorspace"):
            asset_work_cs = str(arow["working_colorspace"])

    eval_out_dir = ""
    eval_key = ""
    eval_spec = None
    if mode == "sequence" and total_frames >= 2:
        eval_key = _compute_seq_eval_key(actual_step, job_row.get("asset_id"), seq_rel, seq_files)
        rdir_abs = storage.abs_from_root(task_row["run_dir"], for_write=True)
        eval_out_dir = os.path.join(rdir_abs, "eval", eval_key)
        paths_cfg = task_cfg.get("paths") or {}
        src_rel = str(paths_cfg.get("source") or "").strip()
        tgt_rel = str(paths_cfg.get("target") or "").strip()
        eval_spec = {
            "eval_out_dir": eval_out_dir,
            "eval_key": eval_key,
            "eval_asset_id": job_row.get("asset_id"),
            "eval_asset_name": str(job_row.get("asset_name") or ""),
            "ckpt_mode": "latest",
            "sup_source_dir": storage.abs_from_root(src_rel) if src_rel else "",
            "sup_target_dir": storage.abs_from_root(tgt_rel) if tgt_rel else "",
        }

    spec = {
        "ckpt_path": ckpt_abs,
        "ckpt_step": int(actual_step),
        "task_config": task_cfg,
        "input_dir": storage.abs_from_root(seq_rel),
        "input_colorspace": asset_work_cs,
        "output_dir": frames_tmp,
        "output_format": fmt,
        "device": subproc_dev,
        "gpu_label": gpu_label,
        "progress_path": prog_path,
        "cancel_path": cancel_path,
    }
    if mode == "single" and idx_clamped is not None:
        spec["frame_index"] = int(idx_clamped)
    if eval_spec is not None:
        spec["eval_spec"] = eval_spec

    with open(spec_path, "w", encoding="utf-8") as fh:
        json.dump(spec, fh, ensure_ascii=False, indent=2)

    py_bin = scheduler._resolve_python_bin()
    cmd = [py_bin, os.path.join(ROOT, "infer.py"), "--spec", spec_path]
    env = storage.apply_subprocess_storage_env(os.environ.copy())
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if is_cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    popen_kw = {
        "cwd": ROOT,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "env": env,
    }
    if os.name == "nt":
        popen_kw["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    proc = subprocess.Popen(cmd, **popen_kw)

    now = time.time()
    db.execute(
        """
        UPDATE infer_jobs
        SET status = 'running', assigned_gpu = ?, pid = ?, device = ?,
            resolved_step = ?, out_id = ?, eval_key = ?, total_frames = ?,
            done_frames = 0, fps = 0.0, started_at = ?, finished_at = NULL, error_msg = ''
        WHERE job_id = ?
        """,
        (
            None if is_cpu else str(gpu_id),
            int(proc.pid),
            gpu_label,
            int(actual_step),
            out_id,
            eval_key,
            int(total_frames),
            now,
            job_id,
        ),
    )

    return ActiveInferSubproc(
        job_id=job_id,
        task_id=tid,
        node_name=task_row["node_name"],
        gpu=str(gpu_id),
        gpu_label=gpu_label,
        proc=proc,
        tmp_dir=tmp_dir,
        final_dir=final_dir,
        prog_path=prog_path,
        cancel_path=cancel_path,
        eval_out_dir=eval_out_dir,
        eval_key=eval_key,
    )


def poll_or_finalize_active_infer(active: ActiveInferSubproc, now: float | None = None) -> bool:
    """Update live progress if still running; finalize EXR output & Compare eval DB if exited. Returns True if finished."""
    ts = float(now if now is not None else time.time())
    job_id = active.job_id

    if active.alive:
        if os.path.isfile(active.prog_path):
            try:
                with open(active.prog_path, "r", encoding="utf-8") as pf:
                    pr = json.load(pf)
                db.execute(
                    """
                    UPDATE infer_jobs
                    SET done_frames = COALESCE(?, done_frames),
                        total_frames = COALESCE(?, total_frames),
                        fps = COALESCE(?, fps),
                        device = COALESCE(?, device)
                    WHERE job_id = ?
                    """,
                    (
                        int(pr["done"]) if pr.get("done") is not None else None,
                        int(pr["total"]) if pr.get("total") is not None else None,
                        float(pr["fps"]) if pr.get("fps") is not None else None,
                        str(pr["device"]) if pr.get("device") else None,
                        job_id,
                    ),
                )
            except Exception:
                pass
        return False

    try:
        active._reader.join(timeout=2.0)
    except Exception:
        pass
    rc = active.proc.poll()
    row = db.query_one("SELECT * FROM infer_jobs WHERE job_id = ?", (job_id,))
    if not row:
        if os.path.exists(active.tmp_dir):
            shutil.rmtree(active.tmp_dir, ignore_errors=True)
        return True

    if active.stop_requested or rc == 130 or rc == -getattr(signal, "SIGINT", 2) or rc == -getattr(signal, "SIGTERM", 15):
        if os.path.exists(active.tmp_dir):
            shutil.rmtree(active.tmp_dir, ignore_errors=True)
        db.execute(
            "UPDATE infer_jobs SET status = 'cancelled', assigned_gpu = NULL, pid = NULL, finished_at = ? WHERE job_id = ?",
            (ts, job_id),
        )
        return True

    if rc != 0:
        tail = "\n".join([ln for ln in list(active.tail) if ln.strip()][-8:])
        sys.stderr.write(f"[WildCat InferSeq ERROR] job={job_id} rc={rc}\n" + "\n".join(active.tail) + "\n")
        sys.stderr.flush()
        if os.path.exists(active.tmp_dir):
            shutil.rmtree(active.tmp_dir, ignore_errors=True)
        db.execute(
            """
            UPDATE infer_jobs
            SET status = 'error', assigned_gpu = NULL, pid = NULL, finished_at = ?, error_msg = ?
            WHERE job_id = ?
            """,
            (ts, tail or i18n.t("server.requestFailed"), job_id),
        )
        return True

    # Completed cleanly (rc == 0): write infer_meta.json, move tmp_dir -> final_dir, sync Compare eval metrics
    pr_final = {}
    if os.path.isfile(active.prog_path):
        try:
            with open(active.prog_path, "r", encoding="utf-8") as pf:
                pr_final = json.load(pf)
        except Exception:
            pr_final = {}

    task_row = db.query_one("SELECT * FROM tasks WHERE id = ?", (int(row["task_id"]),))
    task_cfg = {}
    if task_row:
        try:
            task_cfg = json.loads(task_row.get("config_json") or "{}")
        except Exception:
            task_cfg = {}
    mo_meta = task_cfg.get("model") or {}
    ms_meta = str(mo_meta.get("model_size") or "Medium")
    if ms_meta == "WildCatLargeV1" and bool(mo_meta.get("global_attention", False)):
        ms_meta = "WildCatLargeV1+GlobalAttn"

    started_at = float(row.get("started_at") or ts)
    meta = {
        "out_id": row["out_id"],
        "task_id": int(row["task_id"]),
        "node_name": row["node_name"],
        "model_size": ms_meta,
        "in_channels": int(mo_meta.get("in_channels") or 3),
        "out_channels": int(pr_final.get("out_channels") or mo_meta.get("out_channels") or 3),
        "step": int(row["resolved_step"]),
        "infer_mode": str(row.get("infer_mode") or "sequence"),
        "frame_index": row.get("frame_index"),
        "asset_id": row.get("asset_id"),
        "asset_name": str(row.get("asset_name") or ""),
        "input_rel": str(row.get("seq_rel") or ""),
        "output_format": str(row.get("output_format") or "exr_float"),
        "width": int(pr_final.get("width") or 0),
        "height": int(pr_final.get("height") or 0),
        "seconds": round(float(pr_final.get("seconds") or max(0.01, ts - started_at)), 2),
        "fps": round(float(pr_final.get("fps") or 0.0), 2),
        "auto_on_finish": bool(row.get("auto_on_finish")),
        "created_at": ts,
    }
    for aux in (active.prog_path, active.cancel_path, os.path.join(active.tmp_dir, "spec.json")):
        if os.path.exists(aux):
            try:
                os.remove(aux)
            except OSError:
                pass
    try:
        with open(os.path.join(active.tmp_dir, META_FILE), "w", encoding="utf-8") as mf:
            json.dump(meta, mf, ensure_ascii=False, indent=2)
        os.makedirs(os.path.dirname(active.final_dir), exist_ok=True)
        if os.path.exists(active.final_dir):
            shutil.rmtree(active.final_dir, ignore_errors=True)
        os.replace(active.tmp_dir, active.final_dir)
    except Exception as exc:
        if os.path.exists(active.tmp_dir):
            shutil.rmtree(active.tmp_dir, ignore_errors=True)
        db.execute(
            "UPDATE infer_jobs SET status = 'error', assigned_gpu = NULL, pid = NULL, finished_at = ?, error_msg = ? WHERE job_id = ?",
            (ts, str(exc), job_id),
        )
        return True

    # Sync Compare evaluation metrics written by infer.py into SQLite
    if active.eval_out_dir and active.eval_key:
        m_path = os.path.join(active.eval_out_dir, "metrics.json")
        if os.path.isfile(m_path):
            try:
                mt = round(os.path.getmtime(m_path), 3)
                with open(m_path, "r", encoding="utf-8") as mfh:
                    m_data = json.load(mfh)
                analytics_db.sync_eval_result_to_db(
                    int(row["task_id"]), int(row["user_id"]), str(active.eval_key), m_data, file_mtime=mt
                )
            except Exception:
                pass

    out_meta = _read_output_meta(task_row, str(row["out_id"])) if task_row else None
    db.execute(
        """
        UPDATE infer_jobs
        SET status = 'done', assigned_gpu = NULL, pid = NULL,
            done_frames = total_frames, fps = ?, output_json = ?, finished_at = ?
        WHERE job_id = ?
        """,
        (meta["fps"], json.dumps(out_meta, ensure_ascii=False) if out_meta else "", ts, job_id),
    )
    return True


def start_infer_job(
    user_id: int,
    is_admin: bool,
    username: str,
    asset_id: int,
    task_id: int,
    step: int | None = None,
    output_format: str = "exr_float",
    infer_mode: str = "sequence",
    frame_index: int | None = None,
    preferred_gpu: str = "auto",
) -> dict:
    fmt = str(output_format or "exr_float").strip().lower()
    if fmt not in VALID_FORMATS:
        fmt = "exr_float"
    mode = "single" if str(infer_mode or "").strip().lower() == "single" else "sequence"

    task_row = service.get_task_for_user(user_id, is_admin, task_id=int(task_id))
    if not task_row:
        raise InferNotFoundError(i18n.t("server.taskNotFound"))

    ckpt_abs, resolved_step = resolve_task_checkpoint(task_row, step=step)
    if not ckpt_abs or resolved_step <= 0:
        raise ValueError(i18n.t("server.inferModelUntrained", name=task_row["node_name"]))

    asset_row, seq_rel, seq_files = _resolve_infer_asset(
        user_id, is_admin, int(asset_id), allow_paired_source=(mode == "single")
    )
    if service.is_user_upload_restricted(user_id, is_admin=is_admin):
        user_rel = storage.rel_to_root(storage.user_root(username))
        if storage.dir_total_bytes(user_rel) >= service.TEMP_USER_MAX_STORAGE_BYTES:
            raise PermissionError(i18n.t("server.tempUserUploadQuotaExceeded", max_assets=4))

    if mode == "single":
        fidx = 0 if frame_index is None else max(0, min(len(seq_files) - 1, int(frame_index)))
        probe_file = seq_files[fidx]
        total_frames = 1
    else:
        fidx = None
        probe_file = seq_files[0]
        total_frames = len(seq_files)

    task_cfg = task_row.get("config") or {}
    model_in = int((task_cfg.get("model") or {}).get("in_channels") or 3)
    asset_work_cs = str(asset_row.get("working_colorspace") or "ACEScg")
    try:
        im0 = color.read_image_float(probe_file, input_colorspace=asset_work_cs, working_colorspace=asset_work_cs)
        seq_ch = 1 if im0.ndim == 2 else int(im0.shape[2])
        del im0
    except Exception as exc:
        raise ValueError(f"Cannot read sequence frame {os.path.basename(probe_file)}: {exc}")

    if seq_ch < model_in:
        raise ValueError(i18n.t("server.inferChannelMismatch", seq_ch=seq_ch, model_in=model_in))

    now = time.time()
    job_id = secrets.token_hex(8)
    clean_asset = storage.sanitize_name(asset_row["name"], "seq")[:80]
    if mode == "single":
        real_fnum = storage._extract_frame_num(os.path.basename(probe_file))
        ftag = f"f{real_fnum:04d}" if real_fnum is not None else f"f{fidx + 1:04d}"
        out_id = f"{clean_asset}_step{resolved_step}_single_{ftag}_{_format_short_tag(fmt)}"
    else:
        out_id = f"{clean_asset}_step{resolved_step}_{_format_short_tag(fmt)}"

    req_step_db = int(step) if (step is not None and int(step) > 0) else None
    pref_gpu = str(preferred_gpu or "auto")

    with scheduler.SCHEDULER._lock:
        _prune_old_finished_jobs(now)
        db.execute(
            """
            INSERT INTO infer_jobs (
                job_id, user_id, username, task_id, node_name, depends_on_task_id, auto_on_finish,
                asset_id, asset_name, seq_rel, req_step, resolved_step, infer_mode, frame_index,
                output_format, out_id, status, preferred_gpu, total_frames, created_at, ready_at
            ) VALUES (?, ?, ?, ?, ?, NULL, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?)
            """,
            (
                job_id,
                int(user_id),
                str(username or ""),
                int(task_row["id"]),
                str(task_row["node_name"]),
                int(asset_row["id"]),
                str(asset_row["name"]),
                seq_rel,
                req_step_db,
                int(resolved_step),
                mode,
                fidx,
                fmt,
                out_id,
                pref_gpu,
                int(total_frames),
                now,
                now,
            ),
        )
        scheduler.SCHEDULER.tick()
        qstat = scheduler.SCHEDULER.queue_status()
        q_lookup = {j["job_id"]: j for j in (qstat.get("infer_jobs") or [])}
        row = db.query_one("SELECT * FROM infer_jobs WHERE job_id = ?", (job_id,))
        return _public_job_from_row(row, q_lookup=q_lookup)


def get_infer_job(user_id: int, is_admin: bool, job_id: str) -> dict:
    with scheduler.SCHEDULER._lock:
        row = db.query_one("SELECT * FROM infer_jobs WHERE job_id = ?", (str(job_id or ""),))
        if not row or (not is_admin and int(row["user_id"]) != int(user_id)):
            raise InferNotFoundError(i18n.t("server.inferJobNotFound"))
        qstat = scheduler.SCHEDULER.queue_status()
        q_lookup = {j["job_id"]: j for j in (qstat.get("infer_jobs") or [])}
        return _public_job_from_row(row, q_lookup=q_lookup)


def list_infer_jobs(user_id: int, is_admin: bool) -> list[dict]:
    with scheduler.SCHEDULER._lock:
        _prune_old_finished_jobs()
        qstat = scheduler.SCHEDULER.queue_status()
        q_lookup = {j["job_id"]: j for j in (qstat.get("infer_jobs") or [])}
        if is_admin:
            rows = db.query_all(
                """
                SELECT * FROM infer_jobs
                ORDER BY
                    CASE status
                        WHEN 'running' THEN 0
                        WHEN 'queued' THEN 1
                        WHEN 'waiting_train' THEN 2
                        ELSE 3
                    END ASC,
                    COALESCE(ready_at, created_at) ASC,
                    created_at DESC
                LIMIT 60
                """
            )
        else:
            rows = db.query_all(
                """
                SELECT * FROM infer_jobs
                WHERE user_id = ?
                ORDER BY
                    CASE status
                        WHEN 'running' THEN 0
                        WHEN 'queued' THEN 1
                        WHEN 'waiting_train' THEN 2
                        ELSE 3
                    END ASC,
                    COALESCE(ready_at, created_at) ASC,
                    created_at DESC
                LIMIT 40
                """,
                (int(user_id),),
            )
        return [_public_job_from_row(r, q_lookup=q_lookup) for r in rows]


def cancel_infer_job(user_id: int, is_admin: bool, job_id: str) -> dict:
    clean_id = str(job_id or "")
    wait_proc = None
    with scheduler.SCHEDULER._lock:
        row = db.query_one("SELECT * FROM infer_jobs WHERE job_id = ?", (clean_id,))
        if not row or (not is_admin and int(row["user_id"]) != int(user_id)):
            raise InferNotFoundError(i18n.t("server.inferJobNotFound"))

        if row["status"] in ("waiting_train", "queued"):
            db.execute(
                "UPDATE infer_jobs SET status = 'cancelled', finished_at = ? WHERE job_id = ?",
                (time.time(), clean_id),
            )
            qstat = scheduler.SCHEDULER.queue_status()
            q_lookup = {j["job_id"]: j for j in (qstat.get("infer_jobs") or [])}
            updated = db.query_one("SELECT * FROM infer_jobs WHERE job_id = ?", (clean_id,))
            return _public_job_from_row(updated, q_lookup=q_lookup)

        if row["status"] == "running":
            act = scheduler.SCHEDULER._active_infer.get(clean_id)
            if act is not None and act.alive:
                act.stop_requested = True
                wait_proc = act.proc
                try:
                    with open(act.cancel_path, "w", encoding="utf-8") as cf:
                        cf.write("1")
                except OSError:
                    pass
                try:
                    act.proc.terminate()
                except Exception:
                    pass
            elif row.get("pid"):
                try:
                    os.kill(int(row["pid"]), getattr(signal, "SIGTERM", 15))
                except OSError:
                    pass

    if wait_proc is not None:
        try:
            wait_proc.wait(timeout=4.0)
        except Exception:
            try:
                wait_proc.kill()
            except Exception:
                pass

    with scheduler.SCHEDULER._lock:
        scheduler.SCHEDULER.tick()
        qstat = scheduler.SCHEDULER.queue_status()
        q_lookup = {j["job_id"]: j for j in (qstat.get("infer_jobs") or [])}
        updated = db.query_one("SELECT * FROM infer_jobs WHERE job_id = ?", (clean_id,))
        if not updated:
            raise InferNotFoundError(i18n.t("server.inferJobNotFound"))
        return _public_job_from_row(updated, q_lookup=q_lookup)


def cancel_infer_jobs_for_task(task_id: int) -> None:
    """Force-cancel all active/waiting inference jobs associated with `task_id` (used when deleting a task)."""
    tid = int(task_id)
    rows = db.query_all(
        "SELECT job_id FROM infer_jobs WHERE (task_id = ? OR depends_on_task_id = ?) AND status IN ('waiting_train', 'queued', 'running')",
        (tid, tid),
    )
    for r in rows:
        jid = r["job_id"]
        act = scheduler.SCHEDULER._active_infer.pop(jid, None)
        if act and act.alive:
            act.stop_requested = True
            try:
                act.proc.kill()
                act.proc.wait(timeout=2.0)
            except Exception:
                pass
            if os.path.exists(act.tmp_dir):
                shutil.rmtree(act.tmp_dir, ignore_errors=True)
    db.execute("DELETE FROM infer_jobs WHERE task_id = ? OR depends_on_task_id = ?", (tid, tid))


def list_infer_outputs(user_id: int, is_admin: bool, task_id: int | None = None) -> list[dict]:
    if task_id is not None:
        t = service.get_task_for_user(user_id, is_admin, task_id=int(task_id))
        tasks = [t] if t else []
    else:
        tasks = service.list_tasks(user_id, all_users=bool(is_admin))

    results = []
    for t in tasks:
        seen_ids = set()
        for infer_root in _task_infer_candidate_roots(t):
            if not os.path.isdir(infer_root):
                continue
            try:
                entries = sorted(os.listdir(infer_root))
            except OSError:
                continue
            for name in entries:
                if name.startswith(".") or name in seen_ids:
                    continue
                item = _read_output_meta(t, name)
                if item is not None:
                    seen_ids.add(name)
                    results.append(item)
    results.sort(key=lambda x: x["created_at"], reverse=True)
    return results


def _resolve_owned_output(user_id: int, is_admin: bool, task_id: int, out_id: str) -> tuple[dict, dict, str, list[str]]:
    if not out_id or not _SAFE_OUT_ID_RE.match(out_id) or out_id.startswith("."):
        raise InferNotFoundError(i18n.t("server.assetNotFound"))
    task_row = service.get_task_for_user(user_id, is_admin, task_id=int(task_id))
    if not task_row:
        raise InferNotFoundError(i18n.t("server.taskNotFound"))
    out_abs = _locate_output_dir(task_row, out_id)
    if not out_abs:
        raise InferNotFoundError(i18n.t("server.assetNotFound"))
    meta = _read_output_meta(task_row, out_id)
    if not meta:
        raise InferNotFoundError(i18n.t("server.assetNotFound"))
    frames_abs = os.path.join(out_abs, "frames")
    files = storage.expand_sequence_files(frames_abs)
    if not files:
        raise InferNotFoundError(i18n.t("server.assetNotFound"))
    return task_row, meta, frames_abs, files


def get_infer_output_frames(user_id: int, is_admin: bool, task_id: int, out_id: str) -> dict:
    _, meta, _, out_files = _resolve_owned_output(user_id, is_admin, int(task_id), out_id)
    out_urls = telemetry.batch_to_jpg_urls(out_files)

    in_files = storage.expand_sequence_files(meta["input_rel"]) if meta.get("input_rel") else []
    in_by_stem = {os.path.splitext(os.path.basename(p))[0]: p for p in in_files}
    matched_in = [in_by_stem.get(os.path.splitext(os.path.basename(op))[0], "") for op in out_files]
    valid_in = [p for p in matched_in if p]
    in_url_map = dict(zip(valid_in, telemetry.batch_to_jpg_urls(valid_in))) if valid_in else {}

    frames = []
    for idx, (op, ou) in enumerate(zip(out_files, out_urls)):
        bn = os.path.basename(op)
        ip = matched_in[idx]
        frames.append(
            {
                "idx": idx,
                "name": bn,
                "out_url": ou,
                "in_url": in_url_map.get(ip, ""),
                "download_url": f"/api/v1/infer-seq/outputs/{meta['task_id']}/{meta['out_id']}/file/{bn}",
            }
        )
    return {"output": meta, "frames": frames}


def build_infer_output_zip(user_id: int, is_admin: bool, task_id: int, out_id: str) -> tuple[tempfile.SpooledTemporaryFile, str]:
    task_row, meta, _, out_files = _resolve_owned_output(user_id, is_admin, int(task_id), out_id)
    zip_name = f"{task_row['node_name']}_{meta['out_id']}.zip"
    spool = tempfile.SpooledTemporaryFile(max_size=16 * 1024 * 1024, mode="w+b")
    with zipfile.ZipFile(spool, mode="w", compression=zipfile.ZIP_STORED) as zf:
        for p in out_files:
            if os.path.isfile(p):
                zf.write(p, arcname=os.path.basename(p))
    spool.seek(0)
    return spool, zip_name


def resolve_single_infer_frame(user_id: int, is_admin: bool, task_id: int, out_id: str, filename: str) -> str:
    _, _, frames_abs, _ = _resolve_owned_output(user_id, is_admin, int(task_id), out_id)
    clean_fn = os.path.basename(str(filename or ""))
    if not clean_fn or clean_fn != filename or clean_fn.startswith("."):
        raise InferNotFoundError(i18n.t("server.assetNotFound"))
    fpath = os.path.abspath(os.path.join(frames_abs, clean_fn))
    if not fpath.startswith(os.path.abspath(frames_abs) + os.sep) or not os.path.isfile(fpath):
        raise InferNotFoundError(i18n.t("server.assetNotFound"))
    return fpath


def delete_infer_output(user_id: int, is_admin: bool, task_id: int, out_id: str) -> None:
    task_row, _, _, _ = _resolve_owned_output(user_id, is_admin, int(task_id), out_id)
    for root in _task_infer_candidate_roots(task_row):
        target_dir = os.path.abspath(os.path.join(root, out_id))
        if os.path.normcase(target_dir).startswith(os.path.normcase(root + os.sep)) and os.path.isdir(target_dir):
            shutil.rmtree(target_dir, ignore_errors=True)
