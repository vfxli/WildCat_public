# -*- coding: utf-8 -*-
"""Decoupled REST API v1 for WildCat Core Service, Multi-User Queue, and Future Plugin Integration."""
import os
import subprocess
import sys
import traceback
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from wildcat.version import BUILD_DATE, BUILD_ID, BUILD_LABEL, BUILD_VERSION

from . import analytics_db, db, experiments, gpu_speed_db, i18n, scheduler, security, service, storage, telemetry

router = APIRouter(prefix="/api/v1", tags=["WildCat API v1"])


def _log_api_exc(where: str, exc: Exception) -> str:
    """Always write full Python traceback to wildcat.log (stderr) while returning a sanitized client message."""
    tb = traceback.format_exc()
    sys.stderr.write(f"[WildCat API ERROR] {where}: {type(exc).__name__}: {exc}\n{tb}\n")
    sys.stderr.flush()
    msg = str(exc).strip()
    if isinstance(exc, (ValueError, PermissionError, FileNotFoundError, RuntimeError)) and msg:
        # Strip any accidental embedded Python tracebacks from subprocess or internal messages
        if "Traceback (most recent call last):" in msg:
            lines = [ln.strip() for ln in msg.splitlines() if ln.strip()]
            return lines[-1][:300] if lines else i18n.t("server.requestFailed")
        return msg[:600]
    return i18n.t("server.requestFailed")


def _mask_user_for_client(user: dict) -> dict:
    out = dict(user)
    if not out.get("perm_api"):
        out["api_key"] = ""
    return out


async def _read_upload_bounded(uf: UploadFile, budget: list[int], default_name: str = "frame.exr") -> tuple[str, bytes]:
    fname = os.path.basename((uf.filename or default_name).replace("\\", "/"))
    ext = os.path.splitext(fname)[1].lower()
    if ext not in security.ALLOWED_UPLOAD_EXTS:
        raise ValueError(i18n.t("server.uploadUnsupportedExt", filename=fname))
    chunks = []
    size = 0
    while True:
        chunk = await uf.read(1024 * 1024)
        if not chunk:
            break
        size += len(chunk)
        budget[0] += len(chunk)
        if size > security.MAX_SINGLE_UPLOAD_BYTES or budget[0] > security.MAX_REQUEST_BODY_BYTES:
            raise ValueError(i18n.t("server.uploadTooLarge", role=fname))
        chunks.append(chunk)
    return fname, b"".join(chunks)


# ─────────────────────── Algorithmic Identity, Permissions & Admin Dependencies ───────────────────────


def get_current_user(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    authorization: Optional[str] = Header(default=None),
    api_key_query: Optional[str] = Query(default=None, alias="api_key"),
    x_admin_token: Optional[str] = Header(default=None, alias="X-Admin-Token"),
    x_client_fp: Optional[str] = Header(default=None, alias="X-Client-FP"),
    x_roam_code: Optional[str] = Header(default=None, alias="X-Roam-Code"),
) -> dict:
    client_ip = security.extract_client_ip(request)
    is_admin = security.verify_admin_header(request, x_admin_token, required=False)
    allow_query_key = request.method in ("GET", "HEAD")
    token = (x_api_key or (api_key_query if allow_query_key else None) or "").strip()
    if not token and authorization:
        if authorization.lower().startswith("bearer "):
            token = authorization[7:].strip()
        else:
            token = authorization.strip()
    if not token and api_key_query and not allow_query_key:
        raise HTTPException(status_code=401, detail=i18n.t("server.invalidApiKey"))
    if token:
        api_bf_key = security.check_persistent_lockout("api_key", client_ip)
        user = service.get_user_by_api_key(token, is_admin=is_admin)
        if not user:
            security.record_persistent_failure(api_bf_key, security.API_KEY_TIERS, reset_window_sec=900.0)
            raise HTTPException(status_code=401, detail=i18n.t("server.invalidApiKey"))
        security.clear_persistent_lockout(api_bf_key)
        if not user.get("perm_api"):
            raise HTTPException(status_code=403, detail=i18n.t("server.tempUserApiLocked"))
        return user

    cookie_tok = request.cookies.get(security.SESSION_COOKIE_NAME) or ""
    cookie_uid = security.verify_session_token(cookie_tok) if cookie_tok else None
    try:
        user, should_set_cookie = service.resolve_or_create_session_user(
            cookie_uid=cookie_uid,
            roam_code=(x_roam_code or "").strip(),
            client_ip=client_ip,
            raw_fp=(x_client_fp or "").strip(),
            is_admin=is_admin,
        )
        if should_set_cookie:
            request.state.set_session_uid = int(user["id"])
        return user
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


def require_perm_lab(user: dict) -> None:
    if not user.get("perm_lab"):
        raise HTTPException(status_code=403, detail=i18n.t("server.tempUserLabLocked"))


def require_admin(
    request: Request,
    x_admin_token: Optional[str] = Header(default=None, alias="X-Admin-Token"),
) -> dict:
    security.verify_admin_header(request, x_admin_token, required=True)
    return {"is_admin": True}


# ─────────────────────── Request Schemas ───────────────────────


class AuthRecoverRequest(BaseModel):
    recovery_code: str


class AdminLoginRequest(BaseModel):
    password: str


class AdminPasswordChangeRequest(BaseModel):
    old_password: str = ""
    new_password: str


class AdminCreateMachineRequest(BaseModel):
    username: str = ""
    note: str = ""
    max_tasks: Optional[int] = None
    max_epochs: Optional[int] = None
    perm_lab: bool = False
    perm_api: bool = False
    perm_converter: bool = False


class AdminUpdateUserRequest(BaseModel):
    is_active: Optional[bool] = None
    note: Optional[str] = None
    max_tasks: Optional[int] = None
    reset_used_tasks: bool = False
    submitted_task_count: Optional[int] = None
    max_epochs: Optional[int] = None
    perm_lab: Optional[bool] = None
    perm_api: Optional[bool] = None
    perm_converter: Optional[bool] = None
    regenerate_recovery_code: bool = False


class AdminSettingsRequest(BaseModel):
    max_jobs_per_gpu: Optional[int] = None
    disabled_gpus: Optional[List[str]] = None
    primary_gpu: Optional[str] = None
    secondary_gpu: Optional[str] = None
    enable_temp_user_restrictions: Optional[bool] = None
    default_user_max_tasks: Optional[int] = None
    default_user_max_epochs: Optional[int] = None
    apply_defaults_to_temp_users: bool = False
    data_cache_root: Optional[str] = None
    migrate_existing: bool = False


class ExperimentArchiveFilesRequest(BaseModel):
    task_ids: List[int]
    keep_final_model: bool = True


class TaskSubmitRequest(BaseModel):
    node_name: str = "WildCat1"
    job_type: str = "train"
    preferred_gpu: str = "auto"  # auto | 0 | 1
    priority: int = 0
    config: dict


class ExperimentBatchItem(BaseModel):
    node_name: str = "Exp_Run"
    preferred_gpu: str = "auto"
    priority: int = 0
    label: str = ""
    note: str = ""
    config: dict


class ExperimentBatchSubmitRequest(BaseModel):
    batch_name: str = ""
    target_batch_id: Optional[str] = None
    eval_asset_id: Optional[int] = None
    items: List[ExperimentBatchItem]


class ExperimentGroupCreateRequest(BaseModel):
    batch_name: str
    task_ids: Optional[List[int]] = None


class ExperimentGroupRenameRequest(BaseModel):
    batch_id: str
    batch_name: str


class ExperimentGroupMoveTasksRequest(BaseModel):
    task_ids: List[int]
    target_batch_id: str = ""
    target_batch_name: str = ""


class ExperimentGroupReorderRequest(BaseModel):
    group_order: Optional[List[str]] = None
    task_ids: Optional[List[int]] = None


class ExperimentGroupDissolveRequest(BaseModel):
    batch_id: str


class ExperimentCompareRequest(BaseModel):
    task_ids: List[int]
    eval_asset_id: Optional[int] = None
    ckpt_mode: str = "best"  # best | latest


class ExperimentEvaluateRequest(BaseModel):
    task_ids: List[int]
    eval_asset_id: Optional[int] = None
    ckpt_mode: str = "best"
    force: bool = False




# ─────────────────────── Public / Session / i18n Endpoints ───────────────────────


@router.get("/health")
def api_health():
    q = scheduler.SCHEDULER.queue_status()
    return {
        "status": "ok",
        "service": "WildCat",
        "version": BUILD_VERSION,
        "build_date": BUILD_DATE,
        "build_id": BUILD_ID,
        "build_label": BUILD_LABEL,
        "queue": {
            "running_count": q["running_count"],
            "queued_count": q["queued_count"],
        },
    }


@router.get("/locales")
def api_list_locales():
    return {
        "status": "ok",
        "default": i18n.DEFAULT_LOCALE,
        "locales": i18n.list_locales(),
    }


@router.get("/locales/{lang_code}")
def api_get_locale_bundle(lang_code: str):
    return {
        "status": "ok",
        "code": i18n.resolve_locale_code(lang_code),
        "messages": i18n.get_locale_bundle(lang_code),
    }


@router.get("/auth/me")
def api_me(user: dict = Depends(get_current_user)):
    return {
        "status": "ok",
        "user": _mask_user_for_client(user),
        "version": {
            "version": BUILD_VERSION,
            "build_date": BUILD_DATE,
            "build_id": BUILD_ID,
            "build_label": BUILD_LABEL,
        },
    }


@router.post("/auth/recover")
def api_recover_session(
    req: AuthRecoverRequest,
    request: Request,
    response: Response,
    x_client_fp: Optional[str] = Header(default=None, alias="X-Client-FP"),
    x_admin_token: Optional[str] = Header(default=None, alias="X-Admin-Token"),
):
    client_ip = security.extract_client_ip(request)
    bf_key = security.check_persistent_lockout("recover", client_ip)
    is_admin = security.verify_admin_header(request, x_admin_token, required=False)
    try:
        user = service.recover_user_by_code(
            req.recovery_code,
            client_ip=client_ip,
            raw_fp=(x_client_fp or "").strip(),
            is_admin=is_admin,
        )
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        _, retry_sec = security.record_persistent_failure(bf_key, security.RECOVER_TIERS, reset_window_sec=1800.0)
        if retry_sec > 0:
            raise HTTPException(
                status_code=429,
                detail=i18n.t("server.bruteForceLocked", seconds=retry_sec),
                headers={"Retry-After": str(retry_sec)},
            )
        raise HTTPException(status_code=400, detail=str(e))
    security.clear_persistent_lockout(bf_key)
    security.set_session_cookie(response, request, int(user["id"]))
    return {"status": "ok", "user": _mask_user_for_client(user)}


@router.post("/auth/regenerate-key")
def api_regen_key(user: dict = Depends(get_current_user)):
    if not user.get("perm_api"):
        raise HTTPException(status_code=403, detail=i18n.t("server.tempUserApiLocked"))
    new_key = service.regenerate_user_api_key(user["id"])
    return {"status": "ok", "api_key": new_key}


# ─────────────────────── User Assets Endpoints ───────────────────────


@router.get("/color/ocio")
def api_get_ocio_info():
    from wildcat import color

    return {"status": "ok", "ocio": color.get_ocio_info()}


@router.get("/assets")
def api_list_assets(all_users: bool = False, user: dict = Depends(get_current_user)):
    inc_all = bool(all_users and user.get("is_admin"))
    return {"status": "ok", "assets": service.list_assets(user["id"], all_users=inc_all)}


@router.get("/assets/{asset_id}/frames")
def api_get_asset_frames(asset_id: int, user: dict = Depends(get_current_user)):
    try:
        data = service.get_asset_frames(user["id"], bool(user.get("is_admin")), asset_id)
        return {"status": "ok", **data}
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/assets/upload-pair")
async def api_upload_paired_dataset(
    request: Request,
    name: str = Form(...),
    strategy: str = Form("standard"),
    task_type: str = Form(""),
    max_error_pct: float = Form(35.0),
    enable_check: bool = Form(True),
    input_colorspace: str = Form("auto"),
    working_colorspace: str = Form("ACEScg"),
    source_files: List[UploadFile] = File(...),
    target_files: List[UploadFile] = File(...),
    seq_files: List[UploadFile] = File(default=[]),
    user: dict = Depends(get_current_user),
):
    security.enforce_rate_limit("upload", security.extract_client_ip(request), limit=20, window_sec=60.0)
    try:
        total_files = len(source_files) + len(target_files) + len(seq_files or [])
        if total_files > security.MAX_UPLOAD_FILES_TOTAL * 3:
            raise ValueError(i18n.t("server.uploadTooManyFiles", role="upload", limit=security.MAX_UPLOAD_FILES_TOTAL))
        budget = [0]
        src_tuples = []
        for uf in source_files:
            src_tuples.append(await _read_upload_bounded(uf, budget, "src.exr"))
        tgt_tuples = []
        for uf in target_files:
            tgt_tuples.append(await _read_upload_bounded(uf, budget, "tgt.exr"))
        seq_tuples = []
        for uf in seq_files or []:
            if uf and uf.filename:
                seq_tuples.append(await _read_upload_bounded(uf, budget, "seq.exr"))
        asset = service.create_paired_dataset(
            user_id=user["id"],
            username=user["username"],
            dataset_name=name,
            source_tuples=src_tuples,
            target_tuples=tgt_tuples,
            max_error_pct=float(max_error_pct),
            strategy=strategy,
            enable_check=bool(enable_check),
            input_colorspace=input_colorspace,
            working_colorspace=working_colorspace,
            seq_tuples=seq_tuples or None,
            task_type=task_type,
            is_admin=bool(user.get("is_admin")),
        )
        return {"status": "ok", "asset": asset}
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=_log_api_exc("assets/upload-pair", e))


class AssetTaskTypeUpdatePayload(BaseModel):
    task_type: str = ""


@router.patch("/assets/{asset_id}")
def api_update_asset_task_type(
    asset_id: int,
    payload: AssetTaskTypeUpdatePayload,
    user: dict = Depends(get_current_user),
):
    try:
        asset = service.update_asset_task_type(
            user_id=user["id"],
            is_admin=bool(user.get("is_admin")),
            asset_id=asset_id,
            task_type=payload.task_type,
        )
        return {"status": "ok", "asset": asset}
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/assets/{asset_id}/eval-sequence")
async def api_attach_eval_sequence(
    request: Request,
    asset_id: int,
    input_colorspace: str = Form("auto"),
    working_colorspace: str = Form("ACEScg"),
    seq_files: List[UploadFile] = File(...),
    user: dict = Depends(get_current_user),
):
    security.enforce_rate_limit("upload", security.extract_client_ip(request), limit=20, window_sec=60.0)
    try:
        if len(seq_files) > security.MAX_UPLOAD_FILES_TOTAL:
            raise ValueError(i18n.t("server.uploadTooManyFiles", role="eval", limit=security.MAX_UPLOAD_FILES_TOTAL))
        budget = [0]
        seq_tuples = []
        for uf in seq_files:
            seq_tuples.append(await _read_upload_bounded(uf, budget, "seq.exr"))
        asset = service.attach_asset_eval_sequence(
            user_id=user["id"],
            is_admin=bool(user.get("is_admin")),
            username=user["username"],
            asset_id=asset_id,
            seq_tuples=seq_tuples,
            input_colorspace=input_colorspace,
            working_colorspace=working_colorspace,
        )
        return {"status": "ok", "asset": asset}
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=_log_api_exc("assets/eval-sequence", e))


@router.delete("/assets/{asset_id}/eval-sequence")
def api_remove_eval_sequence(asset_id: int, user: dict = Depends(get_current_user)):
    try:
        asset = service.remove_asset_eval_sequence(user["id"], bool(user.get("is_admin")), asset_id)
        return {"status": "ok", "asset": asset}
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/assets/import-demos")
def api_import_demos(user: dict = Depends(get_current_user)):
    service.seed_demo_assets_for_user(user["id"], force=True)
    return {"status": "ok", "assets": service.list_assets(user["id"])}


@router.delete("/assets/{asset_id}")
def api_delete_asset(asset_id: int, user: dict = Depends(get_current_user)):
    try:
        service.delete_asset(user["id"], bool(user.get("is_admin")), asset_id)
        return {"status": "ok", "deleted_asset_id": asset_id}
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ─────────────────────── Multi-User Tasks & Queue Endpoints ───────────────────────


@router.get("/queue")
def api_queue_status(user: dict = Depends(get_current_user)):
    return {"status": "ok", "queue": scheduler.SCHEDULER.queue_status()}


@router.get("/gpu-benchmarks")
def api_gpu_benchmarks():
    return {
        "status": "ok",
        **gpu_speed_db.get_frontend_speed_payload(scheduler._GPUS_META),
    }


@router.get("/checkpoints")
def api_checkpoints(user: dict = Depends(get_current_user)):
    return {
        "status": "ok",
        "checkpoints": telemetry.list_all_checkpoints(user_id=user["id"], all_users=bool(user.get("is_admin"))),
    }


@router.get("/tasks")
def api_list_tasks(all_users: bool = False, user: dict = Depends(get_current_user)):
    inc_all = bool(all_users and user.get("is_admin"))
    return {
        "status": "ok",
        "tasks": service.list_tasks(user["id"], all_users=inc_all),
        "lab_groups": experiments.get_user_group_meta(user["id"]),
        "db_summary": analytics_db.get_db_analytics_summary(user["id"], all_users=inc_all),
    }


@router.post("/tasks")
def api_submit_task(req: TaskSubmitRequest, request: Request, user: dict = Depends(get_current_user)):
    client_ip = security.extract_client_ip(request)
    security.enforce_rate_limit("task_submit", client_ip, limit=30, window_sec=60.0)
    if req.job_type != "train":
        raise HTTPException(
            status_code=400,
            detail=i18n.t("server.inferRemoved"),
        )
    try:
        t = scheduler.SCHEDULER.submit_task(
            user_id=user["id"],
            username=user["username"],
            node_name=req.node_name,
            job_type="train",
            cfg=req.config,
            preferred_gpu=req.preferred_gpu,
            priority=req.priority,
            is_admin=bool(user.get("is_admin")),
            client_ip=client_ip,
        )
        return {"status": "ok", "task": t}
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=_log_api_exc("tasks/submit", e))


@router.get("/tasks/{task_id}")
def api_get_task(task_id: int, user: dict = Depends(get_current_user)):
    t = service.get_task_for_user(user["id"], bool(user.get("is_admin")), task_id=task_id)
    if not t:
        raise HTTPException(status_code=404, detail=i18n.t("server.taskNotFound"))
    scheduler.SCHEDULER.enrich_task_with_queue(t)
    return {"status": "ok", "task": t}


@router.get("/tasks/{task_id}/telemetry")
def api_get_task_telemetry(
    task_id: int,
    mode: str = "preview",
    pair: Optional[str] = None,
    user: dict = Depends(get_current_user),
):
    t = service.get_task_for_user(user["id"], bool(user.get("is_admin")), task_id=task_id)
    if not t:
        raise HTTPException(status_code=404, detail=i18n.t("server.taskNotFound"))
    rdir_abs = storage.abs_from_root(t["run_dir"])
    lp = os.path.join(rdir_abs, "train.log")
    parsed = telemetry.parse_train_log_full_cached(lp)
    if not parsed["steps"] and not parsed["vals"]:
        db_curves = analytics_db.load_task_curves_from_db(task_id)
        if db_curves:
            parsed["steps"] = db_curves["steps"]
            parsed["vals"] = db_curves["vals"]
            if db_curves["steps"]:
                parsed["latest"] = db_curves["steps"][-1]
    if not parsed["tail"]:
        parsed["tail"] = scheduler.SCHEDULER.get_live_tail(task_id)
    err_msg = (t.get("error_msg") or "").strip()
    if err_msg:
        tail_joined = "\n".join(parsed["tail"][-40:]) if parsed["tail"] else ""
        err_last = err_msg.splitlines()[-1] if err_msg.splitlines() else err_msg
        if err_last not in tail_joined:
            parsed["tail"] = (parsed["tail"] or []) + err_msg.splitlines()

    cfg = t.get("config") or {}
    viewer = telemetry.get_viewer_frames(t["run_dir"], cfg, mode=mode, pair=pair)
    contact_sheet = telemetry.get_contact_sheet_frames(t["run_dir"], cfg)
    models_inv = telemetry.list_task_models(t["run_dir"], cfg, parsed_log=parsed)

    qstat = scheduler.SCHEDULER.queue_status()
    scheduler.SCHEDULER.enrich_task_with_queue(t, qstat=qstat)
    t["has_ckpt"] = os.path.isfile(os.path.join(rdir_abs, "ckpt", "latest.pt")) or (models_inv["count"] > 0)

    return {
        "status": "ok",
        "task": t,
        "steps": parsed["steps"],
        "vals": parsed["vals"],
        "tail": parsed["tail"],
        "latest": parsed["latest"],
        "viewer": viewer,
        "contact_sheet": contact_sheet,
        "models": models_inv,
    }


@router.get("/tasks/{task_id}/models")
def api_list_task_models(task_id: int, user: dict = Depends(get_current_user)):
    t = service.get_task_for_user(user["id"], bool(user.get("is_admin")), task_id=task_id)
    if not t:
        raise HTTPException(status_code=404, detail=i18n.t("server.taskNotFound"))
    rdir_abs = storage.abs_from_root(t["run_dir"])
    parsed = telemetry.parse_train_log_full_cached(os.path.join(rdir_abs, "train.log"))
    return {
        "status": "ok",
        "models": telemetry.list_task_models(t["run_dir"], t.get("config") or {}, parsed_log=parsed),
    }


@router.post("/tasks/{task_id}/recommend-models")
def api_recommend_task_models(
    task_id: int,
    request: Request,
    force: bool = Query(default=False),
    user: dict = Depends(get_current_user),
):
    security.enforce_rate_limit("export_dl", security.extract_client_ip(request), limit=12, window_sec=60.0)
    try:
        models_inv = service.recommend_task_models(user["id"], bool(user.get("is_admin")), task_id, force=bool(force))
        return {"status": "ok", "models": models_inv}
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tasks/{task_id}/stop")
def api_stop_task(task_id: int, user: dict = Depends(get_current_user)):
    try:
        t = service.stop_user_task(user["id"], bool(user.get("is_admin")), task_id)
        return {"status": "ok", "task": t}
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/tasks/{task_id}")
def api_delete_task(task_id: int, user: dict = Depends(get_current_user)):
    try:
        service.delete_user_task(user["id"], bool(user.get("is_admin")), task_id)
        return {"status": "ok", "deleted_task_id": task_id}
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/tasks/{task_id}/export-cat")
def api_export_task_cat(
    task_id: int,
    request: Request,
    step: Optional[int] = Query(default=None),
    user: dict = Depends(get_current_user),
):
    security.enforce_rate_limit("export_dl", security.extract_client_ip(request), limit=30, window_sec=60.0)
    try:
        res = service.export_task_cat(user["id"], bool(user.get("is_admin")), task_id, step=step)
        return {"status": "ok", **res}
    except Exception as e:
        raise HTTPException(status_code=400, detail=_log_api_exc("tasks/export-cat", e))


@router.get("/tasks/{task_id}/download/{ext}")
def api_download_model(
    task_id: int,
    ext: str,
    request: Request,
    step: Optional[int] = Query(default=None),
    user: dict = Depends(get_current_user),
):
    security.enforce_rate_limit("export_dl", security.extract_client_ip(request), limit=60, window_sec=60.0)
    if ext != "pt":
        raise HTTPException(status_code=400, detail="ext must be 'pt' (convert to .cat with Nuke's CatFileCreator)")
    try:
        fpath, dl_name = service.ensure_nuke_model_file(
            user["id"], bool(user.get("is_admin")), task_id, ext, step=step
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=_log_api_exc("tasks/download", e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=_log_api_exc("tasks/download", e))
    return FileResponse(fpath, filename=dl_name, media_type="application/octet-stream")


@router.get("/tasks/{task_id}/download-all/{ext}")
def api_download_all_models(
    task_id: int,
    ext: str,
    request: Request,
    user: dict = Depends(get_current_user),
):
    security.enforce_rate_limit("export_dl", security.extract_client_ip(request), limit=20, window_sec=60.0)
    if ext != "pt":
        raise HTTPException(status_code=400, detail="ext must be 'pt' (convert to .cat with Nuke's CatFileCreator)")
    try:
        zip_abs, dl_name = service.build_all_models_zip(
            user["id"], bool(user.get("is_admin")), task_id, ext
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=_log_api_exc("tasks/download-all", e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=_log_api_exc("tasks/download-all", e))
    return FileResponse(zip_abs, filename=dl_name, media_type="application/zip")




# ─────────────────────── Experiment Lab Endpoints ───────────────────────


@router.post("/experiments/batch-submit")
def api_experiments_batch_submit(req: ExperimentBatchSubmitRequest, request: Request, user: dict = Depends(get_current_user)):
    require_perm_lab(user)
    client_ip = security.extract_client_ip(request)
    security.enforce_rate_limit("task_submit", client_ip, limit=20, window_sec=60.0)
    try:
        res = experiments.batch_submit_experiments(
            user_id=user["id"],
            username=user["username"],
            batch_name=req.batch_name,
            eval_asset_id=req.eval_asset_id,
            items=[it.model_dump() if hasattr(it, "model_dump") else it.dict() for it in req.items],
            target_batch_id=req.target_batch_id,
            is_admin=bool(user.get("is_admin")),
            client_ip=client_ip,
        )
        return {"status": "ok", **res}
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=_log_api_exc("experiments/batch-submit", e))


@router.get("/experiments/groups")
def api_experiments_list_groups(user: dict = Depends(get_current_user)):
    require_perm_lab(user)
    return {"status": "ok", "lab_groups": experiments.get_user_group_meta(user["id"])}


@router.post("/experiments/groups/create")
def api_experiments_create_group(req: ExperimentGroupCreateRequest, user: dict = Depends(get_current_user)):
    require_perm_lab(user)
    try:
        res = experiments.create_experiment_group(
            user_id=user["id"],
            is_admin=bool(user.get("is_admin")),
            batch_name=req.batch_name,
            task_ids=req.task_ids,
        )
        return {"status": "ok", **res}
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=_log_api_exc("experiments/groups/create", e))


@router.put("/experiments/groups/rename")
def api_experiments_rename_group(req: ExperimentGroupRenameRequest, user: dict = Depends(get_current_user)):
    require_perm_lab(user)
    try:
        res = experiments.rename_experiment_group(
            user_id=user["id"],
            is_admin=bool(user.get("is_admin")),
            batch_id=req.batch_id,
            batch_name=req.batch_name,
        )
        return {"status": "ok", **res}
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=_log_api_exc("experiments/groups/rename", e))


@router.post("/experiments/groups/move-tasks")
def api_experiments_move_tasks(req: ExperimentGroupMoveTasksRequest, user: dict = Depends(get_current_user)):
    require_perm_lab(user)
    try:
        res = experiments.move_tasks_to_group(
            user_id=user["id"],
            is_admin=bool(user.get("is_admin")),
            task_ids=req.task_ids,
            target_batch_id=req.target_batch_id,
            target_batch_name=req.target_batch_name,
        )
        return {"status": "ok", **res}
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=_log_api_exc("experiments/groups/move-tasks", e))


@router.put("/experiments/groups/reorder")
def api_experiments_reorder(req: ExperimentGroupReorderRequest, user: dict = Depends(get_current_user)):
    require_perm_lab(user)
    try:
        res = experiments.reorder_experiment_groups_or_tasks(
            user_id=user["id"],
            is_admin=bool(user.get("is_admin")),
            group_order=req.group_order,
            task_ids=req.task_ids,
        )
        return {"status": "ok", **res}
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=_log_api_exc("experiments/groups/reorder", e))


@router.post("/experiments/groups/dissolve")
def api_experiments_dissolve_group(req: ExperimentGroupDissolveRequest, user: dict = Depends(get_current_user)):
    require_perm_lab(user)
    try:
        res = experiments.dissolve_experiment_group(
            user_id=user["id"],
            is_admin=bool(user.get("is_admin")),
            batch_id=req.batch_id,
        )
        return {"status": "ok", **res}
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=_log_api_exc("experiments/groups/dissolve", e))


@router.post("/experiments/compare")
def api_experiments_compare(req: ExperimentCompareRequest, user: dict = Depends(get_current_user)):
    require_perm_lab(user)
    try:
        res = experiments.compare_tasks(
            user_id=user["id"],
            is_admin=bool(user.get("is_admin")),
            task_ids=req.task_ids,
            eval_asset_id=req.eval_asset_id,
            ckpt_mode=req.ckpt_mode,
        )
        return {"status": "ok", **res}
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=_log_api_exc("experiments/compare", e))


@router.post("/experiments/evaluate")
def api_experiments_evaluate(req: ExperimentEvaluateRequest, request: Request, user: dict = Depends(get_current_user)):
    require_perm_lab(user)
    security.enforce_rate_limit("exp_eval", security.extract_client_ip(request), limit=15, window_sec=60.0)
    try:
        st = experiments.EVAL_RUNNER.enqueue(
            user_id=user["id"],
            is_admin=bool(user.get("is_admin")),
            task_ids=req.task_ids,
            eval_asset_id=req.eval_asset_id,
            ckpt_mode=req.ckpt_mode,
            force=bool(req.force),
        )
        return {"status": "ok", "eval_runner": st}
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=_log_api_exc("experiments/evaluate", e))


@router.get("/experiments/eval-status")
def api_experiments_eval_status(user: dict = Depends(get_current_user)):
    require_perm_lab(user)
    return {"status": "ok", "eval_runner": experiments.EVAL_RUNNER.status()}


@router.post("/experiments/eval-cancel")
def api_experiments_eval_cancel(user: dict = Depends(get_current_user)):
    require_perm_lab(user)
    return {"status": "ok", "eval_runner": experiments.EVAL_RUNNER.cancel()}


@router.get("/experiments/analytics-summary")
def api_experiments_analytics_summary(all_users: bool = False, user: dict = Depends(get_current_user)):
    require_perm_lab(user)
    inc_all = bool(all_users and user.get("is_admin"))
    return {
        "status": "ok",
        "db_summary": analytics_db.get_db_analytics_summary(user["id"], all_users=inc_all),
    }


@router.get("/experiments/export-db")
def api_experiments_export_db(
    fmt: str = Query(default="csv"),
    all_users: bool = Query(default=False),
    user: dict = Depends(get_current_user),
):
    require_perm_lab(user)
    inc_all = bool(all_users and user.get("is_admin"))
    content_bytes, fname, media_type = analytics_db.export_tasks_analytics(
        user["id"], all_users=inc_all, fmt=fmt
    )
    return Response(
        content=content_bytes,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@router.post("/experiments/archive-files")
def api_experiments_archive_files(req: ExperimentArchiveFilesRequest, user: dict = Depends(get_current_user)):
    require_perm_lab(user)
    try:
        res = analytics_db.archive_tasks_heavy_files(
            user_id=user["id"],
            is_admin=bool(user.get("is_admin")),
            task_ids=req.task_ids,
            keep_final_model=bool(req.keep_final_model),
        )
        return {
            "status": "ok",
            **res,
            "db_summary": analytics_db.get_db_analytics_summary(user["id"]),
        }
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=_log_api_exc("experiments/archive-files", e))


# ─────────────────────── Admin Endpoints (Password-Protected) ───────────────────────


@router.post("/admin/login")
def api_admin_login(req: AdminLoginRequest, request: Request):
    client_ip = security.extract_client_ip(request)
    glob_key = security.check_persistent_lockout("admin_global", is_global=True)
    ip_key = security.check_persistent_lockout("admin_login", client_ip)
    if not db.verify_admin_password(req.password):
        security.record_persistent_failure(glob_key, security.ADMIN_GLOBAL_TIERS, reset_window_sec=600.0)
        _, retry_sec = security.record_persistent_failure(ip_key, security.ADMIN_LOGIN_TIERS, reset_window_sec=3600.0)
        if retry_sec > 0:
            raise HTTPException(
                status_code=429,
                detail=i18n.t("server.bruteForceLocked", seconds=retry_sec),
                headers={"Retry-After": str(retry_sec)},
            )
        raise HTTPException(status_code=401, detail=i18n.t("server.adminBadPassword"))
    security.clear_persistent_lockout(ip_key)
    return {"status": "ok", "admin_token": db.issue_admin_token()}


@router.put("/admin/password")
def api_admin_change_password(
    req: AdminPasswordChangeRequest,
    request: Request,
    admin: dict = Depends(require_admin),
):
    client_ip = security.extract_client_ip(request)
    glob_key = security.check_persistent_lockout("admin_global", is_global=True)
    ip_key = security.check_persistent_lockout("admin_login", client_ip)
    if not db.verify_admin_password(req.old_password):
        security.record_persistent_failure(glob_key, security.ADMIN_GLOBAL_TIERS, reset_window_sec=600.0)
        _, retry_sec = security.record_persistent_failure(ip_key, security.ADMIN_LOGIN_TIERS, reset_window_sec=3600.0)
        if retry_sec > 0:
            raise HTTPException(
                status_code=429,
                detail=i18n.t("server.bruteForceLocked", seconds=retry_sec),
                headers={"Retry-After": str(retry_sec)},
            )
        raise HTTPException(status_code=401, detail=i18n.t("server.adminOldPasswordMismatch"))
    if len(req.new_password) < 8:
        raise HTTPException(status_code=400, detail=i18n.t("server.adminPasswordMinLen"))
    try:
        db.update_admin_password(req.new_password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    security.clear_persistent_lockout(ip_key)
    return {
        "status": "ok",
        "message": i18n.t("server.adminPasswordUpdated"),
        "admin_token": db.issue_admin_token(),
    }


@router.get("/admin/settings")
def api_admin_get_settings(admin: dict = Depends(require_admin)):
    st_info = storage.get_storage_disk_stats()
    q_st = scheduler.SCHEDULER.queue_status()
    pol = scheduler.get_gpu_pool_policy()
    return {
        "status": "ok",
        "settings": {
            "max_jobs_per_gpu": int(db.get_setting("max_jobs_per_gpu", "1") or 1),
            "disabled_gpus": pol["disabled_gpus"],
            "primary_gpu": pol["primary_gpu"],
            "secondary_gpu": pol["secondary_gpu"],
            "enable_temp_user_restrictions": service.is_temp_restrictions_enabled(),
            "default_user_max_tasks": int(db.get_setting("default_user_max_tasks", "2") or 2),
            "default_user_max_epochs": int(db.get_setting("default_user_max_epochs", "500000") or 500000),
            "data_cache_root": st_info["data_cache_root"],
        },
        "gpu_pool": {
            "gpus": q_st.get("gpus") or [],
            "total_gpu_count": int(q_st.get("total_gpu_count") or 0),
            "enabled_gpu_count": int(q_st.get("enabled_gpu_count") or 0),
            "all_gpus_disabled": bool(q_st.get("all_gpus_disabled")),
            "primary_gpu": pol["primary_gpu"],
            "secondary_gpu": pol["secondary_gpu"],
            "disabled_gpus": pol["disabled_gpus"],
        },
        "storage": st_info,
        "db_summary": analytics_db.get_db_analytics_summary(all_users=True),
    }


@router.put("/admin/settings")
def api_admin_put_settings(req: AdminSettingsRequest, admin: dict = Depends(require_admin)):
    migrated_items = 0
    updated_temp_users = 0
    try:
        if req.max_jobs_per_gpu is not None:
            db.set_setting("max_jobs_per_gpu", str(max(1, min(4, req.max_jobs_per_gpu))))
        if req.disabled_gpus is not None or req.primary_gpu is not None or req.secondary_gpu is not None:
            scheduler.save_gpu_pool_policy(
                disabled_gpus=req.disabled_gpus,
                primary_gpu=req.primary_gpu,
                secondary_gpu=req.secondary_gpu,
            )
        if req.enable_temp_user_restrictions is not None:
            db.set_setting("enable_temp_user_restrictions", "1" if req.enable_temp_user_restrictions else "0")
        if req.default_user_max_tasks is not None:
            db.set_setting("default_user_max_tasks", str(max(0, min(999999, int(req.default_user_max_tasks)))))
        if req.default_user_max_epochs is not None:
            db.set_setting(
                "default_user_max_epochs",
                str(max(100, min(100_000_000, int(req.default_user_max_epochs)))),
            )
        if req.apply_defaults_to_temp_users:
            mt = int(db.get_setting("default_user_max_tasks", "2") or 2)
            me = int(db.get_setting("default_user_max_epochs", "500000") or 500000)
            updated_temp_users = service.apply_default_quotas_to_temp_users(mt, me)
        if req.data_cache_root is not None:
            q_st = scheduler.SCHEDULER.queue_status()
            ev_st = experiments.EVAL_RUNNER.status()
            has_active = bool((q_st.get("running_count") or 0) > 0 or ev_st.get("busy"))
            c_res = storage.configure_data_cache_root(
                new_path=req.data_cache_root,
                migrate_existing=bool(req.migrate_existing),
                has_active_jobs=has_active,
            )
            migrated_items = int(c_res.get("migrated_items") or 0)
        scheduler.SCHEDULER.tick()
    except Exception as e:
        raise HTTPException(status_code=400, detail=_log_api_exc("admin/settings", e))
    out = api_admin_get_settings(admin)
    out["migrated_items"] = migrated_items
    out["updated_temp_users"] = updated_temp_users
    return out


@router.get("/admin/users")
def api_admin_list_users(admin: dict = Depends(require_admin)):
    return {"status": "ok", "users": service.list_all_users_with_stats()}


@router.post("/admin/users")
def api_admin_create_user(req: AdminCreateMachineRequest, admin: dict = Depends(require_admin)):
    try:
        u = service.create_user_account(
            req.username,
            note=req.note,
            max_tasks=req.max_tasks,
            max_epochs=req.max_epochs,
            perm_lab=bool(req.perm_lab),
            perm_api=bool(req.perm_api),
            perm_converter=bool(req.perm_converter),
        )
        return {"status": "ok", "user": u}
    except Exception as e:
        raise HTTPException(status_code=400, detail=_log_api_exc("admin/users/create", e))


@router.put("/admin/users/{target_uid}")
def api_admin_update_user(
    target_uid: int, req: AdminUpdateUserRequest, admin: dict = Depends(require_admin)
):
    try:
        u = service.admin_update_user(
            target_uid,
            is_active=req.is_active,
            note=req.note,
            max_tasks=req.max_tasks,
            reset_used_tasks=bool(req.reset_used_tasks),
            submitted_task_count=req.submitted_task_count,
            max_epochs=req.max_epochs,
            perm_lab=req.perm_lab,
            perm_api=req.perm_api,
            perm_converter=req.perm_converter,
            regenerate_recovery_code=bool(req.regenerate_recovery_code),
        )
        return {"status": "ok", "user": u}
    except Exception as e:
        raise HTTPException(status_code=400, detail=_log_api_exc("admin/users/update", e))


@router.post("/admin/impersonate/{target_uid}")
def api_admin_impersonate_user(
    target_uid: int,
    request: Request,
    response: Response,
    x_client_fp: Optional[str] = Header(default=None, alias="X-Client-FP"),
    admin: dict = Depends(require_admin),
):
    client_ip = security.extract_client_ip(request)
    u = service.admin_bind_browser_to_user(
        target_uid,
        client_ip=client_ip,
        raw_fp=(x_client_fp or "").strip(),
    )
    if not u:
        raise HTTPException(status_code=404, detail=i18n.t("server.userNotFound"))
    security.set_session_cookie(response, request, int(u["id"]))
    return {"status": "ok", "user": _mask_user_for_client(u)}


@router.delete("/admin/users/{target_uid}")
def api_admin_delete_user(target_uid: int, admin: dict = Depends(require_admin)):
    try:
        service.admin_delete_user(target_uid)
        return {"status": "ok", "deleted_user_id": target_uid}
    except Exception as e:
        raise HTTPException(status_code=400, detail=_log_api_exc("admin/users/delete", e))
