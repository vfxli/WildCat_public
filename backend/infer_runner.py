# -*- coding: utf-8 -*-
"""
Validation Sequence Inference Runner (`backend/infer_runner.py`).

Executes full-frame inference of a trained task checkpoint over a dataset's validation sequence (`eval/` or
standalone `eval_sequence`) inside an isolated `infer.py --spec` subprocess (0 persistent FastAPI VRAM, O(1) RAM).
Untrained tasks (without saved checkpoints) are strictly rejected.
"""
import glob
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import zipfile

from wildcat import color

from . import db, i18n, scheduler, service, storage, telemetry

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INFER_SUBDIR = "infer_seq"
META_FILE = "infer_meta.json"
VALID_FORMATS = ("exr_float", "exr_half")
_SAFE_OUT_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,180}$")

_INFER_LOCK = threading.Lock()
_INFER_RUN_LOCK = threading.Lock()
_INFER_JOBS: dict[str, dict] = {}
_JOB_KEEP_SEC = 3600.0


class InferBusyError(RuntimeError):
    """The user already has a validation-sequence inference job queued or running."""


class InferNotFoundError(LookupError):
    """Requested asset, task, job or inference output was not found."""


def _pick_infer_device() -> str:
    try:
        q = scheduler.SCHEDULER.queue_status()
        gpus = [g for g in (q.get("gpus") or []) if g.get("enabled", True)]
        if not gpus:
            return "cpu"
        has_running = int(q.get("running_count") or 0) > 0
        idle = [g for g in gpus if not g.get("busy")]
        if idle:
            if has_running:
                best = min(
                    idle,
                    key=lambda g: (
                        0 if g.get("is_secondary") else (2 if g.get("is_primary") else 1),
                        -int(g.get("vram_mb") or scheduler._GPU_VRAM_MB.get(str(g["gpu_id"]), 0)),
                        int(g["gpu_id"]) if str(g["gpu_id"]).isdigit() else 0,
                    ),
                )
            else:
                best = min(
                    idle,
                    key=lambda g: (
                        0 if g.get("is_primary") else (1 if g.get("is_secondary") else 2),
                        -int(g.get("vram_mb") or scheduler._GPU_VRAM_MB.get(str(g["gpu_id"]), 0)),
                        int(g["gpu_id"]) if str(g["gpu_id"]).isdigit() else 0,
                    ),
                )
            return f"cuda:{best['gpu_id']}"
        best = min(
            gpus,
            key=lambda g: (
                0 if g.get("is_secondary") else (2 if g.get("is_primary") else 1),
                int(g.get("active_count") or 0),
                -int(g.get("vram_mb") or scheduler._GPU_VRAM_MB.get(str(g["gpu_id"]), 0)),
            ),
        )
        return f"cuda:{best['gpu_id']}"
    except Exception:
        pass
    return "cuda:0"


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

    parsed = telemetry.parse_train_log_full_cached(os.path.join(rdir_abs, "train.log"))
    inv = telemetry.list_task_models(run_dir_rel, task_row.get("config") or {}, parsed_log=parsed)
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


def _public_job(job: dict) -> dict:
    return {
        "job_id": job["job_id"],
        "user_id": job["user_id"],
        "task_id": job["task_id"],
        "node_name": job["node_name"],
        "asset_id": job["asset_id"],
        "asset_name": job["asset_name"],
        "step": job["step"],
        "infer_mode": job.get("infer_mode", "sequence"),
        "frame_index": job.get("frame_index"),
        "output_format": job["output_format"],
        "state": job["state"],
        "done": job["done"],
        "total": job["total"],
        "progress": round(job["done"] / float(max(1, job["total"])), 4) if job["total"] > 0 else 0.0,
        "fps": job["fps"],
        "device": job["device"],
        "seconds": round((job["finished_at"] or time.time()) - job["started_at"], 1) if job["started_at"] else 0.0,
        "error": job["error"],
        "output": job["output"],
    }


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


def _worker(job_id: str) -> None:
    job = _INFER_JOBS[job_id]
    lang_tok = i18n.set_current_lang(job.get("lang"))
    tmp_dir = job["tmp_dir"]
    frames_tmp = os.path.join(tmp_dir, "frames")
    prog_path = os.path.join(tmp_dir, "progress.json")
    cancel_path = os.path.join(tmp_dir, ".cancel")
    spec_path = os.path.join(tmp_dir, "spec.json")
    proc = None
    try:
        with _INFER_RUN_LOCK:
            if job["cancel"].is_set():
                job["state"] = "cancelled"
                return
            os.makedirs(frames_tmp, exist_ok=True)
            dev = _pick_infer_device()
            job["device"] = dev
            job["state"] = "running"
            job["started_at"] = time.time()

            spec = {
                "ckpt_path": job["ckpt_path"],
                "ckpt_step": job["step"],
                "task_config": job["task_cfg"],
                "input_dir": storage.abs_from_root(job["seq_rel"]),
                "output_dir": frames_tmp,
                "output_format": job["output_format"],
                "device": dev,
                "progress_path": prog_path,
                "cancel_path": cancel_path,
            }
            if job.get("infer_mode") == "single" and job.get("frame_index") is not None:
                spec["frame_index"] = int(job["frame_index"])
            with open(spec_path, "w", encoding="utf-8") as fh:
                json.dump(spec, fh, ensure_ascii=False, indent=2)

            py_bin = scheduler._resolve_python_bin()
            cmd = [py_bin, os.path.join(ROOT, "infer.py"), "--spec", spec_path]
            env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", OPENCV_IO_ENABLE_OPENEXR="1")
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

            while proc.poll() is None:
                if job["cancel"].is_set():
                    try:
                        with open(cancel_path, "w", encoding="utf-8") as cf:
                            cf.write("1")
                    except OSError:
                        pass
                    time.sleep(0.3)
                    if proc.poll() is None:
                        proc.terminate()
                        try:
                            proc.wait(timeout=5.0)
                        except Exception:
                            proc.kill()
                    break
                if os.path.isfile(prog_path):
                    try:
                        with open(prog_path, "r", encoding="utf-8") as pf:
                            pr = json.load(pf)
                        job["done"] = int(pr.get("done") or job["done"])
                        job["total"] = int(pr.get("total") or job["total"])
                        job["fps"] = float(pr.get("fps") or job["fps"])
                        job["device"] = str(pr.get("device") or job["device"])
                    except Exception:
                        pass
                time.sleep(0.25)

            out_txt = ""
            try:
                out_txt = proc.stdout.read() if proc.stdout else ""
            except Exception:
                pass

            if job["cancel"].is_set() or proc.returncode == 130:
                job["state"] = "cancelled"
                return

            if proc.returncode != 0:
                tail = "\n".join([ln for ln in out_txt.splitlines() if ln.strip()][-8:])
                sys.stderr.write(f"[WildCat InferSeq ERROR] job={job_id} rc={proc.returncode}\n{out_txt}\n")
                sys.stderr.flush()
                job["error"] = tail or i18n.t("server.requestFailed")
                job["state"] = "error"
                return

            pr_final = {}
            if os.path.isfile(prog_path):
                try:
                    with open(prog_path, "r", encoding="utf-8") as pf:
                        pr_final = json.load(pf)
                except Exception:
                    pr_final = {}

            mo_meta = job["task_cfg"].get("model") or {}
            ms_meta = str(mo_meta.get("model_size") or "Medium")
            if ms_meta == "WildCatLargeV1" and bool(mo_meta.get("global_attention", False)):
                ms_meta = "WildCatLargeV1+GlobalAttn"
            meta = {
                "out_id": job["out_id"],
                "task_id": job["task_id"],
                "node_name": job["node_name"],
                "model_size": ms_meta,
                "in_channels": int((job["task_cfg"].get("model") or {}).get("in_channels") or 3),
                "out_channels": int(pr_final.get("out_channels") or (job["task_cfg"].get("model") or {}).get("out_channels") or 3),
                "step": job["step"],
                "infer_mode": job.get("infer_mode", "sequence"),
                "frame_index": job.get("frame_index"),
                "asset_id": job["asset_id"],
                "asset_name": job["asset_name"],
                "input_rel": job["seq_rel"],
                "output_format": job["output_format"],
                "width": int(pr_final.get("width") or 0),
                "height": int(pr_final.get("height") or 0),
                "seconds": round(float(pr_final.get("seconds") or (time.time() - job["started_at"])), 2),
                "fps": round(float(pr_final.get("fps") or 0.0), 2),
                "created_at": time.time(),
            }
            for aux in (prog_path, cancel_path, spec_path):
                if os.path.exists(aux):
                    try:
                        os.remove(aux)
                    except OSError:
                        pass
            with open(os.path.join(tmp_dir, META_FILE), "w", encoding="utf-8") as mf:
                json.dump(meta, mf, ensure_ascii=False, indent=2)

            final_dir = job["final_dir"]
            os.makedirs(os.path.dirname(final_dir), exist_ok=True)
            if os.path.exists(final_dir):
                shutil.rmtree(final_dir, ignore_errors=True)
            os.replace(tmp_dir, final_dir)

            job["done"] = job["total"]
            job["fps"] = meta["fps"]
            job["output"] = _read_output_meta(job["task_row"], job["out_id"])
            job["state"] = "done"
    except BaseException as exc:
        sys.stderr.write(f"[WildCat InferSeq EXCEPTION] job={job_id}: {exc}\n{traceback.format_exc()}\n")
        sys.stderr.flush()
        job["error"] = str(exc)
        job["state"] = "error"
    finally:
        i18n.reset_current_lang(lang_tok)
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
        job["finished_at"] = time.time()
        for k in ("task_row", "task_cfg", "ckpt_path"):
            job.pop(k, None)


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
    infer_root = _task_infer_primary_root(task_row)
    os.makedirs(infer_root, exist_ok=True)
    clean_asset = storage.sanitize_name(asset_row["name"], "seq")[:80]
    if mode == "single":
        real_fnum = storage._extract_frame_num(os.path.basename(probe_file))
        ftag = f"f{real_fnum:04d}" if real_fnum is not None else f"f{fidx + 1:04d}"
        out_id = f"{clean_asset}_step{resolved_step}_single_{ftag}_{_format_short_tag(fmt)}"
    else:
        out_id = f"{clean_asset}_step{resolved_step}_{_format_short_tag(fmt)}"
    tmp_dir = os.path.join(infer_root, f".tmp_{job_id}")
    final_dir = os.path.join(infer_root, out_id)

    job = {
        "job_id": job_id,
        "user_id": int(user_id),
        "task_id": int(task_row["id"]),
        "node_name": task_row["node_name"],
        "asset_id": int(asset_row["id"]),
        "asset_name": asset_row["name"],
        "seq_rel": seq_rel,
        "step": int(resolved_step),
        "infer_mode": mode,
        "frame_index": fidx,
        "output_format": fmt,
        "out_id": out_id,
        "tmp_dir": tmp_dir,
        "final_dir": final_dir,
        "ckpt_path": ckpt_abs,
        "task_row": task_row,
        "task_cfg": task_cfg,
        "state": "queued",
        "done": 0,
        "total": total_frames,
        "fps": 0.0,
        "device": "",
        "error": "",
        "output": None,
        "created_at": now,
        "started_at": None,
        "finished_at": None,
        "lang": i18n.get_current_lang(),
        "cancel": threading.Event(),
    }

    with _INFER_LOCK:
        for jid in [k for k, v in _INFER_JOBS.items() if v["finished_at"] and (now - v["finished_at"]) > _JOB_KEEP_SEC]:
            _INFER_JOBS.pop(jid, None)
        if any(v["user_id"] == int(user_id) and v["state"] in ("queued", "running") for v in _INFER_JOBS.values()):
            raise InferBusyError(i18n.t("server.inferBusy"))
        _INFER_JOBS[job_id] = job

    threading.Thread(target=_worker, args=(job_id,), name="wc-infer-" + job_id, daemon=True).start()
    return _public_job(job)


def get_infer_job(user_id: int, is_admin: bool, job_id: str) -> dict:
    with _INFER_LOCK:
        job = _INFER_JOBS.get(str(job_id or ""))
        if not job or (not is_admin and job["user_id"] != int(user_id)):
            raise InferNotFoundError(i18n.t("server.inferJobNotFound"))
        return _public_job(job)


def cancel_infer_job(user_id: int, is_admin: bool, job_id: str) -> dict:
    with _INFER_LOCK:
        job = _INFER_JOBS.get(str(job_id or ""))
        if not job or (not is_admin and job["user_id"] != int(user_id)):
            raise InferNotFoundError(i18n.t("server.inferJobNotFound"))
        job["cancel"].set()
        return _public_job(job)


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
