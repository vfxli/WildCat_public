#!/usr/bin/env bash
# ==============================================================================
# WildCat 离线环境部署与热更新管理脚本 (Rocky Linux 9.8 / Driver 570.181 / CUDA 12.8)
# ==============================================================================
# 用法:
#   ./wildcat.sh start | stop | restart | status   # 服务启停与状态管理
#   ./wildcat.sh update <WildCat_YYYYMMDD_N.tar.gz># 后续秒级增量热更新(保留环境与全部数据)
#   ./wildcat.sh rollback                          # 一键回滚到上一个代码版本
#   ./wildcat.sh pack-patch                        # 在开发机打包轻量级代码热更新包 (~100KB)
# ==============================================================================
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${WILDCAT_PORT:-8765}"
PID_FILE="${BASE_DIR}/wildcat.pid"
LOG_FILE="${BASE_DIR}/wildcat.log"
export OPENCV_IO_ENABLE_OPENEXR=1

# 自动重定向便携式 Python 虚拟环境路径 (支持解压到任意目录/NAS共享目录直接运行，不依赖软链接)
relocate_venv() {
  local venv_dir=""
  if [ -d "${BASE_DIR}/runtime/python/cpython-3.12" ]; then
    venv_dir="${BASE_DIR}/runtime/python"
  elif [ -d "${BASE_DIR}/.venv/cpython-3.12" ]; then
    venv_dir="${BASE_DIR}/.venv"
  fi

  if [ -n "${venv_dir}" ]; then
    cat > "${venv_dir}/pyvenv.cfg" <<EOF
home = ${venv_dir}/cpython-3.12/bin
implementation = CPython
version_info = 3.12
include-system-site-packages = false
seed = true
EOF
    if [ -d "${venv_dir}/cpython-3.12/lib/python3.12/site-packages" ]; then
      echo "../../../../lib/python3.12/site-packages" > "${venv_dir}/cpython-3.12/lib/python3.12/site-packages/wildcat_venv.pth" 2>/dev/null || true
    fi
    ln -sfn ../cpython-3.12/bin/python3.12 "${venv_dir}/bin/python" 2>/dev/null || true
    ln -sfn python "${venv_dir}/bin/python3" 2>/dev/null || true
    ln -sfn python "${venv_dir}/bin/python3.12" 2>/dev/null || true
    chmod +x "${venv_dir}/cpython-3.12/bin/python3.12" "${venv_dir}/bin/python"* 2>/dev/null || true
    export PYTHONPATH="${BASE_DIR}:${venv_dir}/lib/python3.12/site-packages:${PYTHONPATH:-}"
  fi
}

relocate_venv

if [ -f "${BASE_DIR}/runtime/python/cpython-3.12/bin/python3.12" ]; then
  PY_BIN="${BASE_DIR}/runtime/python/cpython-3.12/bin/python3.12"
elif [ -f "${BASE_DIR}/.venv/cpython-3.12/bin/python3.12" ]; then
  PY_BIN="${BASE_DIR}/.venv/cpython-3.12/bin/python3.12"
elif [ -x "${BASE_DIR}/.venv/bin/python" ]; then
  PY_BIN="${BASE_DIR}/.venv/bin/python"
else
  PY_BIN="python3"
fi

if [ -f "${BASE_DIR}/app/webui/app.py" ]; then
  APP_DIR="${BASE_DIR}/app"
else
  APP_DIR="${BASE_DIR}"
fi

link_persistent_data() {
  mkdir -p "${BASE_DIR}/storage" "${BASE_DIR}/run" "${BASE_DIR}/config" "${BASE_DIR}/backups"
  if [ "${APP_DIR}" != "${BASE_DIR}" ]; then
    mkdir -p "${BASE_DIR}/data/storage" "${BASE_DIR}/data/run"
    rm -rf "${APP_DIR}/storage" "${APP_DIR}/run"
    ln -sfn "${BASE_DIR}/data/storage" "${APP_DIR}/storage"
    ln -sfn "${BASE_DIR}/data/run" "${APP_DIR}/run"
  fi
}

is_running() {
  if [ -f "${PID_FILE}" ]; then
    local pid
    pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
    if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
      return 0
    fi
  fi
  return 1
}

do_start() {
  if is_running; then
    echo "[WildCat] 服务已在运行中 (PID=$(cat "${PID_FILE}"))，监听端口: ${PORT}"
    return 0
  fi
  relocate_venv
  link_persistent_data
  cd "${APP_DIR}"
  echo "[WildCat] Starting with PY_BIN=${PY_BIN} ($( "${PY_BIN}" --version 2>&1 || true ))" > "${LOG_FILE}"
  nohup "${PY_BIN}" "${APP_DIR}/webui/app.py" --host 0.0.0.0 --port "${PORT}" >> "${LOG_FILE}" 2>&1 &
  echo $! > "${PID_FILE}"
  sleep 2
  if ! is_running; then
    echo "[WildCat] 服务启动失败！最近日志 (${LOG_FILE}) 如下："
    tail -n 30 "${LOG_FILE}" 2>/dev/null || true
    rm -f "${PID_FILE}"
    exit 1
  fi
  local ip_addr
  ip_addr="$(hostname -I 2>/dev/null | awk '{print $1}' || echo '127.0.0.1')"
  echo "[WildCat] 服务已启动 (PID=$(cat "${PID_FILE}")) | Python: ${PY_BIN}"
  echo "[WildCat] 访问地址: http://${ip_addr}:${PORT}"
  echo "[WildCat] 日志文件: ${LOG_FILE}"
}

do_stop() {
  if is_running; then
    local pid
    pid="$(cat "${PID_FILE}")"
    kill "${pid}" 2>/dev/null || true
    sleep 1
    kill -9 "${pid}" 2>/dev/null || true
    rm -f "${PID_FILE}"
    echo "[WildCat] 服务已停止"
  else
    rm -f "${PID_FILE}"
    echo "[WildCat] 服务未运行"
  fi
}

do_status() {
  if is_running; then
    echo "[WildCat] 运行中 (PID=$(cat "${PID_FILE}")) | 端口: ${PORT} | Python: ${PY_BIN}"
  else
    echo "[WildCat] 已停止"
  fi
}

do_update() {
  local pkg="${1:-}"
  if [ -z "${pkg}" ] || [ ! -f "${pkg}" ]; then
    echo "用法: ./wildcat.sh update <WildCat_YYYYMMDD_N.tar.gz>"
    exit 1
  fi
  echo "[WildCat] 正在执行增量热更新 (仅更新代码层，保留 .venv 运行库、weights 与 storage/run 数据)..."
  mkdir -p "${BASE_DIR}/backups"
  local ts
  ts="$(date +%Y%m%d_%H%M%S)"
  if [ -d "${APP_DIR}/backend" ]; then
    # shellcheck disable=SC2046
    tar -czf "${BASE_DIR}/backups/app_backup_${ts}.tar.gz" -C "${APP_DIR}" $(app_code_files) 2>/dev/null || true
    echo "[WildCat] 已自动备份当前代码至: backups/app_backup_${ts}.tar.gz"
  fi
  do_stop || true
  if tar -tzf "${pkg}" 2>/dev/null | head -n 5 | grep -q '^WildCat/'; then
    tar -xzf "${pkg}" --strip-components=1 -C "${APP_DIR}"
  else
    tar -xzf "${pkg}" -C "${APP_DIR}"
  fi
  do_start
  echo "[WildCat] 增量热更新完成！"
}

# Code files shipped in patches / backed up before updates (only those that exist).
app_code_files() {
  local f out=""
  for f in wildcat.sh backend frontend webui wildcat tools train.py infer.py export_cat.py config/default.json; do
    [ -e "${APP_DIR}/${f}" ] && out="${out} ${f}"
  done
  echo "${out}"
}

do_pack_patch() {
  local out_file="${1:-WildCat_$(date +%Y%m%d)_patch.tar.gz}"
  # shellcheck disable=SC2046
  tar --exclude='__pycache__' --exclude='*.pyc' -czf "${out_file}" -C "${APP_DIR}" $(app_code_files)
  echo "[WildCat] 已生成轻量增量更新包: ${out_file} ($(du -h "${out_file}" | awk '{print $1}'))"
}

do_rollback() {
  local latest_bak
  latest_bak="$(ls -1t "${BASE_DIR}/backups"/app_backup_*.tar.gz 2>/dev/null | head -n 1 || true)"
  if [ -z "${latest_bak}" ]; then
    echo "[WildCat] 未找到可回滚的备份包 (backups/app_backup_*.tar.gz)"
    exit 1
  fi
  echo "[WildCat] 正在回滚至备份版本: ${latest_bak} ..."
  do_stop || true
  tar -xzf "${latest_bak}" -C "${APP_DIR}"
  do_start
  echo "[WildCat] 已成功回滚并重启！"
}

case "${1:-status}" in
  start)      do_start ;;
  stop)       do_stop ;;
  restart)    do_stop; do_start ;;
  status)     do_status ;;
  update)     do_update "${2:-}" ;;
  pack-patch) do_pack_patch "${2:-}" ;;
  rollback)   do_rollback ;;
  *)
    echo "用法: $0 {start|stop|restart|status|update <patch.tar.gz>|pack-patch [out.tar.gz]|rollback}"
    exit 1
    ;;
esac
