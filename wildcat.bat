@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul 2>&1

REM ==============================================================================
REM WildCat Windows 服务启停与增量热更新管理脚本 (Windows 10 / 11 / Server)
REM ==============================================================================
REM 用法:
REM   wildcat.bat start | stop | restart | status     服务启停与状态管理
REM   wildcat.bat update [文件名.tar.gz]              从 update\ 目录自动读取更新包执行热更新
REM   wildcat.bat rollback                            一键回滚到上一个代码版本
REM   wildcat.bat pack-patch [输出文件名.tar.gz]      打包轻量级代码热更新包至 update\ 目录
REM ==============================================================================

set "BASE_DIR=%~dp0"
if "%BASE_DIR:~-1%"=="\" set "BASE_DIR=%BASE_DIR:~0,-1%"
if "%WILDCAT_PORT%"=="" (
    set "PORT=8765"
) else (
    set "PORT=%WILDCAT_PORT%"
)

set "OPENCV_IO_ENABLE_OPENEXR=1"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

set "PY_BIN="
for %%P in (
    "%BASE_DIR%\runtime\python\python.exe"
    "%BASE_DIR%\runtime\python\Scripts\python.exe"
    "%BASE_DIR%\.venv\Scripts\python.exe"
    "%BASE_DIR%\.venv\python.exe"
) do (
    if not defined PY_BIN if exist "%%~P" set "PY_BIN=%%~P"
)
if not defined PY_BIN set "PY_BIN=python"

if exist "%BASE_DIR%\app\webui\app.py" (
    set "APP_DIR=%BASE_DIR%\app"
) else (
    set "APP_DIR=%BASE_DIR%"
)

set "PYTHONPATH=%APP_DIR%;%BASE_DIR%;%BASE_DIR%\.venv\Lib\site-packages;%BASE_DIR%\runtime\python\Lib\site-packages;%PYTHONPATH%"

set "WC_BASE_DIR=%BASE_DIR%"
set "WC_APP_DIR=%APP_DIR%"
set "WC_PORT=%PORT%"
set "WC_PY_BIN=%PY_BIN%"

(
    "%PY_BIN%" -c "import datetime, glob, os, shutil, socket, subprocess, sys, tarfile, time, traceback; base = os.environ['WC_BASE_DIR']; app_dir = os.environ['WC_APP_DIR']; port = os.environ['WC_PORT']; py_bin = os.environ['WC_PY_BIN']; update_dir = os.path.join(base, 'update'); pid_file = os.path.join(base, 'wildcat.pid'); log_file = os.path.join(base, 'wildcat.log'); cmd = (sys.argv[1] if len(sys.argv) > 1 else 'status').lower(); arg2 = sys.argv[2] if len(sys.argv) > 2 else ''; exec(open(os.path.join(base, 'wildcat.bat'), encoding='utf-8').read().rsplit('# === PY_MANAGER_BEGIN ===', 1)[1])" %*
    exit /b !ERRORLEVEL!
)

# === PY_MANAGER_BEGIN ===
def _is_alive(pid):
    if not pid or pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, int(pid))
        if not h:
            return False
        try:
            code = ctypes.c_ulong()
            if k32.GetExitCodeProcess(h, ctypes.byref(code)):
                return code.value == 259
            return False
        finally:
            k32.CloseHandle(h)
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def _read_pid():
    if os.path.isfile(pid_file):
        try:
            return int(open(pid_file, encoding="utf-8").read().strip())
        except Exception:
            return None
    return None


def _lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _build_label():
    try:
        r = subprocess.run(
            [py_bin, "-c", "import wildcat.version as v; print(v.BUILD_LABEL)"],
            cwd=app_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _rotate_log_if_needed():
    try:
        if os.path.isfile(log_file) and os.path.getsize(log_file) > 20 * 1024 * 1024:
            os.replace(log_file, log_file + ".old")
    except OSError:
        pass


def _code_items():
    items = [
        "wildcat.sh",
        "wildcat.bat",
        "set_admin_password.sh",
        "set_admin_password.bat",
        "update/README.txt",
        "backend",
        "frontend",
        "webui",
        "wildcat",
        "tools",
        "train.py",
        "infer.py",
        "export_cat.py",
        "config/default.json",
    ]
    return [x for x in items if os.path.exists(os.path.join(app_dir, x))]


def _verify_python_modules():
    print("[WildCat] 正在执行 Python 核心模块与语法预检...")
    code = (
        "import sys, traceback\n"
        "try:\n"
        "    import wildcat.version as v, wildcat.trainer, backend.server\n"
        "    print('[WildCat] ✓ 模块预检通过 -> ' + v.BUILD_LABEL)\n"
        "except BaseException:\n"
        "    sys.stderr.write('[WildCat] ✕ 模块预检失败，完整 Python 异常堆栈如下:\\n')\n"
        "    traceback.print_exc()\n"
        "    sys.exit(1)\n"
    )
    r = subprocess.run(
        [py_bin, "-c", code],
        cwd=app_dir,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    out_txt = (r.stdout or "") + (r.stderr or "")
    if out_txt:
        print(out_txt.rstrip())
        try:
            with open(log_file, "a", encoding="utf-8", errors="replace") as lf:
                lf.write(out_txt + "\n")
        except OSError:
            pass
    if r.returncode != 0:
        print("[WildCat] ✕ 代码预检未通过！可执行 wildcat.bat rollback 一键恢复上一版本。")
        sys.exit(1)


def do_start():
    pid = _read_pid()
    ver = _build_label()
    if _is_alive(pid):
        print(f"[WildCat] 服务已在运行中 (PID={pid}) | 版本: {ver} | 监听端口: {port}")
        return
    for d in ("storage", "run", "config", "backups", "update", os.path.join("update", "applied")):
        os.makedirs(os.path.join(base, d), exist_ok=True)
    _rotate_log_if_needed()
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", encoding="utf-8", errors="replace") as lf:
        lf.write(
            f"\n{'=' * 80}\n"
            f"[WildCat] Startup at {now_str} | Build: {ver} | PY_BIN={py_bin}\n"
            f"{'=' * 80}\n"
        )
    log_fh = open(log_file, "a", encoding="utf-8", errors="replace")
    flags = 0
    if os.name == "nt":
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
    proc = subprocess.Popen(
        [py_bin, os.path.join(app_dir, "webui", "app.py"), "--host", "0.0.0.0", "--port", str(port)],
        cwd=app_dir,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        creationflags=flags,
    )
    with open(pid_file, "w", encoding="utf-8") as pf:
        pf.write(str(proc.pid))
    time.sleep(2.0)
    if not _is_alive(proc.pid):
        print(f"[WildCat] ✕ 服务启动失败！最近日志与错误堆栈 ({log_file}) 如下：")
        if os.path.isfile(log_file):
            lines = open(log_file, encoding="utf-8", errors="replace").readlines()
            print("".join(lines[-80:]))
        if os.path.exists(pid_file):
            os.remove(pid_file)
        sys.exit(1)
    print(f"[WildCat] 服务已启动 (PID={proc.pid}) | 版本: {ver} | Python: {py_bin}")
    print(f"[WildCat] 访问地址: http://{_lan_ip()}:{port}")
    print(f"[WildCat] 日志文件: {log_file} (追加模式，保留完整异常堆栈)")


def do_stop():
    pid = _read_pid()
    if _is_alive(pid):
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
        else:
            try:
                os.kill(pid, 15)
            except OSError:
                pass
        print("[WildCat] 服务已停止")
    else:
        print("[WildCat] 服务未运行")
    if os.path.exists(pid_file):
        try:
            os.remove(pid_file)
        except OSError:
            pass


def do_status():
    pid = _read_pid()
    ver = _build_label()
    if _is_alive(pid):
        print(f"[WildCat] 运行中 (PID={pid}) | 版本: {ver} | 端口: {port} | Python: {py_bin}")
    else:
        print(f"[WildCat] 已停止 | 当前代码版本: {ver}")


def _pack_tar(out_path):
    skip_exact = {"tools/make_public.py", "tools/publish_release.sh"}

    def _filter(ti):
        norm = ti.name.replace("\\", "/")
        if "__pycache__" in norm or norm.endswith(".pyc") or norm in skip_exact:
            return None
        return ti

    with tarfile.open(out_path, "w:gz") as tf:
        for rel in _code_items():
            full = os.path.join(app_dir, rel)
            tf.add(full, arcname=rel, filter=_filter)


def _extract_tar(pkg_path, dest_dir):
    with tarfile.open(pkg_path, "r:gz") as tf:
        members = tf.getmembers()
        strip_prefix = all(m.name == "WildCat" or m.name.startswith("WildCat/") for m in members if m.name)
        for m in members:
            if m.name == "WildCat" and strip_prefix:
                continue
            if strip_prefix and m.name.startswith("WildCat/"):
                m.name = m.name[len("WildCat/"):]
            if not m.name or ".." in m.name.replace("\\", "/").split("/"):
                continue
            tf.extract(m, dest_dir)


def _resolve_update_pkg():
    os.makedirs(update_dir, exist_ok=True)
    os.makedirs(os.path.join(update_dir, "applied"), exist_ok=True)
    req = (arg2 or "").strip()
    if not req:
        cands = sorted(glob.glob(os.path.join(update_dir, "*.tar.gz")), key=os.path.getmtime, reverse=True)
        if not cands:
            print(f"[WildCat] ✕ 未在 {update_dir} 目录中找到任何热更新包 (*.tar.gz)！")
            print(f"[WildCat] 请先将更新包放入固定目录: {update_dir}")
            print("[WildCat] 然后重新执行: wildcat.bat update")
            sys.exit(1)
        return cands[0]

    if os.sep not in req and "/" not in req:
        cand = os.path.join(update_dir, req)
        if os.path.isfile(cand):
            return cand

    if os.path.isfile(req):
        req_parent = os.path.normcase(os.path.abspath(os.path.dirname(req)))
        upd_norm = os.path.normcase(os.path.abspath(update_dir))
        if req_parent == upd_norm:
            return os.path.abspath(req)
        print(f"[WildCat] ✕ 热更新包必须放在项目固定目录: {update_dir}")
        print(f"[WildCat] 当前传入路径不在 update 目录内: {req}")
        sys.exit(1)

    print(f"[WildCat] ✕ 在 {update_dir} 中未找到更新包: {req}")
    sys.exit(1)


def do_pack_patch():
    os.makedirs(update_dir, exist_ok=True)
    req = (arg2 or "").strip()
    if not req:
        out_file = os.path.join(update_dir, f"WildCat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_patch.tar.gz")
    elif os.sep not in req and "/" not in req:
        out_file = os.path.join(update_dir, req)
    else:
        out_file = req
    _pack_tar(out_file)
    sz_kb = os.path.getsize(out_file) / 1024.0
    print(f"[WildCat] 已生成轻量增量更新包: {out_file} ({sz_kb:.1f} KB)")


def do_update():
    pkg = _resolve_update_pkg()
    print(f"[WildCat] 检测到热更新包: {pkg}")
    try:
        with tarfile.open(pkg, "r:gz") as tf:
            _ = tf.getnames()
    except Exception:
        print(f"[WildCat] ✕ 更新包校验失败（文件损坏或未传输完整）: {pkg}")
        traceback.print_exc()
        sys.exit(1)

    print("[WildCat] 正在执行增量热更新 (仅更新代码层，保留 .venv 运行库、weights 与 storage/run 数据)...")
    bak_dir = os.path.join(base, "backups")
    applied_dir = os.path.join(update_dir, "applied")
    os.makedirs(bak_dir, exist_ok=True)
    os.makedirs(applied_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    bak_file = os.path.join(bak_dir, f"app_backup_{ts}.tar.gz")
    if os.path.isdir(os.path.join(app_dir, "backend")):
        _pack_tar(bak_file)
        print(f"[WildCat] 已自动备份当前代码至: backups/app_backup_{ts}.tar.gz")
    do_stop()
    _extract_tar(pkg, app_dir)
    if app_dir != base and os.path.isfile(os.path.join(app_dir, "wildcat.bat")):
        shutil.copy2(os.path.join(app_dir, "wildcat.bat"), os.path.join(base, "wildcat.bat"))
    _verify_python_modules()
    do_start()

    base_name = os.path.basename(pkg)
    stem = base_name[:-7] if base_name.endswith(".tar.gz") else base_name
    archived = os.path.join(applied_dir, f"{stem}_applied_{ts}.tar.gz")
    shutil.move(pkg, archived)
    print(f"[WildCat] 已将本次更新包归档至: update/applied/{os.path.basename(archived)}")
    print(f"[WildCat] ✓ 增量热更新全部完成！当前运行版本: {_build_label()}")


def do_rollback():
    baks = sorted(glob.glob(os.path.join(base, "backups", "app_backup_*.tar.gz")), key=os.path.getmtime, reverse=True)
    if not baks:
        print("[WildCat] 未找到可回滚的备份包 (backups/app_backup_*.tar.gz)")
        sys.exit(1)
    latest = baks[0]
    print(f"[WildCat] 正在回滚至备份版本: {latest} ...")
    do_stop()
    _extract_tar(latest, app_dir)
    if app_dir != base and os.path.isfile(os.path.join(app_dir, "wildcat.bat")):
        shutil.copy2(os.path.join(app_dir, "wildcat.bat"), os.path.join(base, "wildcat.bat"))
    _verify_python_modules()
    do_start()
    print(f"[WildCat] ✓ 已成功回滚并重启！当前版本: {_build_label()}")


if cmd == "start":
    do_start()
elif cmd == "stop":
    do_stop()
elif cmd == "restart":
    do_stop()
    do_start()
elif cmd == "status":
    do_status()
elif cmd == "update":
    do_update()
elif cmd == "pack-patch":
    do_pack_patch()
elif cmd == "rollback":
    do_rollback()
else:
    print("用法: wildcat.bat {start|stop|restart|status|update [文件名.tar.gz]|pack-patch [文件名.tar.gz]|rollback}")
    sys.exit(1)
