@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul 2>&1

REM WildCat 修改管理后台密码（不需要重启服务）
REM 用法：把本文件放到 WildCat 目录下，双击或在命令行运行：  set_admin_password.bat

cd /d "%~dp0"
if not exist "backend" (
    echo 请把本脚本放在 WildCat 目录里（和 wildcat.bat 同一层）再运行。
    exit /b 1
)

set "PY="
for %%P in (
    ".\.venv\Scripts\python.exe"
    ".\.venv\python.exe"
    ".\runtime\python\python.exe"
    ".\runtime\python\Scripts\python.exe"
) do (
    if not defined PY if exist "%%~P" set "PY=%%~P"
)
if not defined PY set "PY=python"

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

"%PY%" -c "import getpass, os, sys; sys.path.insert(0, os.getcwd()); p1 = getpass.getpass('请输入新的管理员密码: '); p2 = getpass.getpass('请再输入一次: '); (sys.exit('两次输入不一致，未修改。') if p1 != p2 else None); (sys.exit('密码至少 4 位，未修改。') if len(p1) < 4 else None); from backend import db; db.init_db(); db.update_admin_password(p1); print('管理员密码已修改。无需重启服务，刷新网页后用新密码登录管理后台即可。')"
exit /b %ERRORLEVEL%
