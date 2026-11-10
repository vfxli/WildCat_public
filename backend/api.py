# -*- coding: utf-8 -*-
"""Decoupled REST API v1 for WildCat Core Service, Multi-User Queue, and Future Plugin Integration."""
import os
import subprocess
import sys
import traceback
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from wildcat.version import BUILD_DATE, BUILD_ID, BUILD_LABEL, BUILD_VERSION

from . import db, scheduler, service, storage, telemetry

router = APIRouter(prefix="/api/v1", tags=["WildCat API v1"])


def _log_api_exc(where: str, exc: Exception) -> str:
    """Always write full Python traceback to wildcat.log (stderr) and format detail message."""
    tb = traceback.format_exc()
    sys.stderr.write(f"[WildCat API ERROR] {where}: {type(exc).__name__}: {exc}\n{tb}\n")
    sys.stderr.flush()
    if isinstance(exc, (ValueError, PermissionError)):
        return str(exc) or f"{type(exc).__name__}"
    return f"{type(exc).__name__}: {exc}\n\n{tb}"


# ─────────────────────── Machine Identity & Admin Dependencies ───────────────────────


def get_current_user(
    request: Request,
    x_machine_name: Optional[str] = Header(default=None, alias="X-Machine-Name"),
    machine_query: Optional[str] = Query(default=None, alias="machine"),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    authorization: Optional[str] = Header(default=None),
    api_key_query: Optional[str] = Query(default=None, alias="api_key"),
    x_admin_token: Optional[str] = Header(default=None, alias="X-Admin-Token"),
) -> dict:
    is_admin = db.verify_admin_token(x_admin_token or "")
    token = x_api_key or api_key_query
    if not token and authorization:
        if authorization.lower().startswith("bearer "):
            token = authorization[7:].strip()
        else:
            token = authorization.strip()
    if token:
        user = service.get_user_by_api_key(token)
        if user:
            user["is_admin"] = is_admin
            return user

    client_ip = request.client.host if request.client else ""
    try:
        user = service.get_or_create_machine_user(
            raw_machine=x_machine_name or machine_query or "",
            client_ip=client_ip,
        )
        user["is_admin"] = is_admin
        return user
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


def require_admin(
    x_admin_token: Optional[str] = Header(default=None, alias="X-Admin-Token"),
) -> dict:
    if not db.verify_admin_token(x_admin_token or ""):
        raise HTTPException(status_code=403, detail="需要管理员密码验证")
    return {"is_admin": True}


# ─────────────────────── Request Schemas ───────────────────────


class AdminLoginRequest(BaseModel):
    password: str


class AdminPasswordChangeRequest(BaseModel):
    old_password: str = ""
    new_password: str


class AdminCreateMachineRequest(BaseModel):
    username: str
    note: str = ""


class AdminUpdateUserRequest(BaseModel):
    is_active: Optional[bool] = None
    note: Optional[str] = None


class AdminSettingsRequest(BaseModel):
    max_jobs_per_gpu: Optional[int] = None


class TaskSubmitRequest(BaseModel):
    node_name: str = "WildCat1"
    job_type: str = "train"
    preferred_gpu: str = "auto"  # auto | 0 | 1
    priority: int = 0
    config: dict


class TaskDraftRequest(BaseModel):
    node_name: str = "WildCat1"
    config: dict




# ─────────────────────── Public / Session Endpoints ───────────────────────


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


@router.get("/auth/me")
def api_me(user: dict = Depends(get_current_user)):
    return {
        "status": "ok",
        "user": user,
        "version": {
            "version": BUILD_VERSION,
            "build_date": BUILD_DATE,
            "build_id": BUILD_ID,
            "build_label": BUILD_LABEL,
        },
    }


@router.post("/auth/regenerate-key")
def api_regen_key(user: dict = Depends(get_current_user)):
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


@router.get("/assets/inspect")
def api_inspect_asset_path(path: str = "", user: dict = Depends(get_current_user)):
    from wildcat import color

    files = storage.expand_sequence_files(path.strip())
    thumb = telemetry.to_jpg_url(files[0]) if files else None
    width, height, channels = 0, 0, 3
    file_ext = os.path.splitext(files[0])[1].lower() if files else ".exr"
    if files:
        try:
            im = color.read_image_float(files[0], input_colorspace="auto", working_colorspace="ACEScg")
            height, width = int(im.shape[0]), int(im.shape[1])
            channels = 1 if im.ndim == 2 else int(im.shape[2])
        except Exception as e:
            _log_api_exc("assets/inspect", e)
    return {
        "status": "ok",
        "path": path.strip(),
        "frame_count": len(files),
        "width": width,
        "height": height,
        "channels": channels,
        "format": file_ext.lstrip("."),
        "thumb_url": thumb,
    }


@router.post("/assets/upload-pair")
async def api_upload_paired_dataset(
    name: str = Form(...),
    strategy: str = Form("standard"),
    max_error_pct: float = Form(35.0),
    enable_check: bool = Form(True),
    input_colorspace: str = Form("ACEScg"),
    working_colorspace: str = Form("ACEScg"),
    source_files: List[UploadFile] = File(...),
    target_files: List[UploadFile] = File(...),
    user: dict = Depends(get_current_user),
):
    try:
        src_tuples = []
        for uf in source_files:
            src_tuples.append((uf.filename or "src.exr", await uf.read()))
        tgt_tuples = []
        for uf in target_files:
            tgt_tuples.append((uf.filename or "tgt.exr", await uf.read()))
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
        )
        return {"status": "ok", "asset": asset}
    except Exception as e:
        raise HTTPException(status_code=400, detail=_log_api_exc("assets/upload-pair", e))


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


@router.get("/checkpoints")
def api_checkpoints(user: dict = Depends(get_current_user)):
    return {
        "status": "ok",
        "checkpoints": telemetry.list_all_checkpoints(user_id=user["id"], all_users=bool(user.get("is_admin"))),
    }


@router.get("/checkpoints/inspect")
def api_inspect_checkpoint(path: str = "", user: dict = Depends(get_current_user)):
    from wildcat import models

    p = (path or "").strip()
    abs_p = storage.abs_from_root(p) if p and not os.path.isabs(p) else p
    spec = models.inspect_weight_source(abs_p if os.path.isfile(abs_p) else p)
    return {"status": "ok", "spec": spec}


@router.get("/tasks")
def api_list_tasks(all_users: bool = False, user: dict = Depends(get_current_user)):
    inc_all = bool(all_users and user.get("is_admin"))
    return {"status": "ok", "tasks": service.list_tasks(user["id"], all_users=inc_all)}


@router.post("/tasks/draft")
def api_save_draft_task(req: TaskDraftRequest, user: dict = Depends(get_current_user)):
    try:
        t = scheduler.SCHEDULER.save_draft_node(
            user["id"], user["username"], req.node_name, req.config
        )
        return {"status": "ok", "task": t}
    except Exception as e:
        raise HTTPException(status_code=400, detail=_log_api_exc("tasks/draft", e))


@router.post("/tasks")
def api_submit_task(req: TaskSubmitRequest, user: dict = Depends(get_current_user)):
    if req.job_type != "train":
        raise HTTPException(
            status_code=400,
            detail="网页端序列推理功能已移除；请下载 Nuke 标准命名模型在 Nuke Inference 节点中使用。",
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
        )
        return {"status": "ok", "task": t}
    except Exception as e:
        raise HTTPException(status_code=400, detail=_log_api_exc("tasks/submit", e))


@router.get("/tasks/{task_id}")
def api_get_task(task_id: int, user: dict = Depends(get_current_user)):
    t = service.get_task_for_user(user["id"], bool(user.get("is_admin")), task_id=task_id)
    if not t:
        raise HTTPException(status_code=404, detail="Task not found.")
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
        raise HTTPException(status_code=404, detail="Task not found.")
    rdir_abs = storage.abs_from_root(t["run_dir"])
    lp = os.path.join(rdir_abs, "train.log")
    parsed = telemetry.parse_train_log(lp, max_bytes=400_000)
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
    qpos = next((q["queue_position"] for q in qstat["queued"] if q["id"] == task_id), None)
    t["queue_position"] = qpos
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
        raise HTTPException(status_code=404, detail="Task not found.")
    rdir_abs = storage.abs_from_root(t["run_dir"])
    parsed = telemetry.parse_train_log(os.path.join(rdir_abs, "train.log"), max_bytes=400_000)
    return {
        "status": "ok",
        "models": telemetry.list_task_models(t["run_dir"], t.get("config") or {}, parsed_log=parsed),
    }


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
    step: Optional[int] = Query(default=None),
    user: dict = Depends(get_current_user),
):
    try:
        res = service.export_task_cat(user["id"], bool(user.get("is_admin")), task_id, step=step)
        return {"status": "ok", **res}
    except Exception as e:
        raise HTTPException(status_code=400, detail=_log_api_exc("tasks/export-cat", e))


@router.get("/tasks/{task_id}/download/{ext}")
def api_download_model(
    task_id: int,
    ext: str,
    step: Optional[int] = Query(default=None),
    user: dict = Depends(get_current_user),
):
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
    user: dict = Depends(get_current_user),
):
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




# ─────────────────────── Admin Endpoints (Password-Protected) ───────────────────────


@router.post("/admin/login")
def api_admin_login(req: AdminLoginRequest):
    if not db.verify_admin_password(req.password):
        raise HTTPException(status_code=401, detail="管理员密码错误")
    return {"status": "ok", "admin_token": db.issue_admin_token()}


@router.put("/admin/password")
def api_admin_change_password(req: AdminPasswordChangeRequest, admin: dict = Depends(require_admin)):
    if not db.verify_admin_password(req.old_password):
        raise HTTPException(status_code=401, detail="原管理员密码验证失败")
    if len(req.new_password) < 4:
        raise HTTPException(status_code=400, detail="新密码长度不能少于 4 位")
    db.set_admin_password(req.new_password)
    return {
        "status": "ok",
        "message": "管理员密码已加密更新",
        "admin_token": db.issue_admin_token(),
    }


@router.get("/admin/settings")
def api_admin_get_settings(admin: dict = Depends(require_admin)):
    return {
        "status": "ok",
        "settings": {
            "allow_self_registration": db.get_setting("allow_self_registration", "false"),
            "max_jobs_per_gpu": int(db.get_setting("max_jobs_per_gpu", "1") or 1),
        },
    }


@router.put("/admin/settings")
def api_admin_put_settings(req: AdminSettingsRequest, admin: dict = Depends(require_admin)):
    if req.max_jobs_per_gpu is not None:
        db.set_setting("max_jobs_per_gpu", str(max(1, min(4, req.max_jobs_per_gpu))))
    scheduler.SCHEDULER.tick()
    return api_admin_get_settings(admin)


@router.get("/admin/users")
def api_admin_list_users(admin: dict = Depends(require_admin)):
    return {"status": "ok", "users": service.list_all_users_with_stats()}


@router.post("/admin/users")
def api_admin_create_user(req: AdminCreateMachineRequest, admin: dict = Depends(require_admin)):
    try:
        u = service.create_user_account(req.username, role="user", note=req.note)
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
        )
        return {"status": "ok", "user": u}
    except Exception as e:
        raise HTTPException(status_code=400, detail=_log_api_exc("admin/users/update", e))


@router.delete("/admin/users/{target_uid}")
def api_admin_delete_user(target_uid: int, admin: dict = Depends(require_admin)):
    try:
        service.admin_delete_user(target_uid)
        return {"status": "ok", "deleted_user_id": target_uid}
    except Exception as e:
        raise HTTPException(status_code=400, detail=_log_api_exc("admin/users/delete", e))
