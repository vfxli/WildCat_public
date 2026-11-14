#!/usr/bin/env bash
# ==============================================================================
# WildCat 离线环境部署与热更新管理脚本 (Linux / CUDA 12.8)
# ==============================================================================
# 用法:
#   ./wildcat.sh start | stop | restart | status   # 服务启停与状态管理
#   ./wildcat.sh update [文件名.tar.gz]            # 从 update/ 目录自动读取更新包执行热更新
#   ./wildcat.sh rollback                          # 一键回滚到上一个代码版本
#   ./wildcat.sh pack-patch [输出文件名.tar.gz]    # 打包轻量增量热更新包至 update/ 目录
# ==============================================================================

# 使用 main() 包裹全部逻辑，确保热更新覆盖 wildcat.sh 自身时 Bash 不会因文件偏移变化报错
main() {
  set -euo pipefail

  BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  UPDATE_DIR="${BASE_DIR}/update"
  PORT="${WILDCAT_PORT:-8765}"
  PID_FILE="${BASE_DIR}/wildcat.pid"
  LOG_FILE="${BASE_DIR}/wildcat.log"
  export OPENCV_IO_ENABLE_OPENEXR=1
  export PYTHONUTF8=1

  if [ -f "${BASE_DIR}/app/webui/app.py" ]; then
    APP_DIR="${BASE_DIR}/app"
  else
    APP_DIR="${BASE_DIR}"
  fi

  # 自动重定向便携式 Python 虚拟环境路径 (支持解压到任意目录直接运行)
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
      export PYTHONPATH="${APP_DIR}:${BASE_DIR}:${venv_dir}/lib/python3.12/site-packages:${PYTHONPATH:-}"
    else
      export PYTHONPATH="${APP_DIR}:${BASE_DIR}:${PYTHONPATH:-}"
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

  link_persistent_data() {
    mkdir -p "${BASE_DIR}/storage" "${BASE_DIR}/run" "${BASE_DIR}/config" "${BASE_DIR}/backups" "${UPDATE_DIR}" "${UPDATE_DIR}/applied"
    if [ "${APP_DIR}" != "${BASE_DIR}" ]; then
      mkdir -p "${BASE_DIR}/data/storage" "${BASE_DIR}/data/run"
      rm -rf "${APP_DIR}/storage" "${APP_DIR}/run"
      ln -sfn "${BASE_DIR}/data/storage" "${APP_DIR}/storage"
      ln -sfn "${BASE_DIR}/data/run" "${APP_DIR}/run"
    fi
  }

  rotate_log_if_needed() {
    if [ -f "${LOG_FILE}" ]; then
      local sz=0
      sz="$(wc -c < "${LOG_FILE}" 2>/dev/null || echo 0)"
      if [ "${sz}" -gt 20971520 ]; then
        mv -f "${LOG_FILE}" "${LOG_FILE}.old" 2>/dev/null || true
      fi
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

  current_build_label() {
    "${PY_BIN}" -c "import wildcat.version as v; print(v.BUILD_LABEL)" 2>/dev/null || echo "unknown"
  }

  do_start() {
    if is_running; then
      echo "[WildCat] 服务已在运行中 (PID=$(cat "${PID_FILE}")) | 版本: $(current_build_label) | 端口: ${PORT}"
      return 0
    fi
    relocate_venv
    link_persistent_data
    rotate_log_if_needed
    cd "${APP_DIR}"
    local now_str ver_str py_ver
    now_str="$(date '+%Y-%m-%d %H:%M:%S')"
    ver_str="$(current_build_label)"
    py_ver="$("${PY_BIN}" --version 2>&1 || true)"
    {
      echo ""
      echo "================================================================================"
      echo "[WildCat] Startup at ${now_str} | Build: ${ver_str} | PY_BIN=${PY_BIN} (${py_ver})"
      echo "================================================================================"
    } >> "${LOG_FILE}"
    if command -v setsid >/dev/null 2>&1; then
      setsid nohup "${PY_BIN}" "${APP_DIR}/webui/app.py" --host 0.0.0.0 --port "${PORT}" </dev/null >> "${LOG_FILE}" 2>&1 &
    else
      nohup "${PY_BIN}" "${APP_DIR}/webui/app.py" --host 0.0.0.0 --port "${PORT}" </dev/null >> "${LOG_FILE}" 2>&1 &
    fi
    echo $! > "${PID_FILE}"
    local ready=0
    for _ in $(seq 1 60); do
      if ! is_running; then
        break
      fi
      if "${PY_BIN}" -c "import socket, sys; s = socket.socket(); s.settimeout(0.25); ok = (s.connect_ex(('127.0.0.1', int(sys.argv[1]))) == 0); s.close(); sys.exit(0 if ok else 1)" "${PORT}" 2>/dev/null; then
        ready=1
        break
      fi
      sleep 0.5
    done
    if [ "${ready}" -ne 1 ] || ! is_running; then
      echo "[WildCat] ✕ 服务启动失败！最近日志与错误堆栈 (${LOG_FILE}) 如下："
      tail -n 80 "${LOG_FILE}" 2>/dev/null || true
      rm -f "${PID_FILE}"
      exit 1
    fi
    local ip_addr
    ip_addr="$(hostname -I 2>/dev/null | awk '{print $1}' || echo '127.0.0.1')"
    echo "[WildCat] 服务已启动 (PID=$(cat "${PID_FILE}")) | 版本: ${ver_str} | Python: ${PY_BIN}"
    echo "[WildCat] 访问地址: http://${ip_addr}:${PORT}"
    echo "[WildCat] 日志文件: ${LOG_FILE} (追加模式，保留完整异常堆栈)"
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
    pkill -f "webui/app.py.*--port ${PORT}" 2>/dev/null || pkill -f "webui/app.py" 2>/dev/null || true
  }

  do_status() {
    if is_running; then
      echo "[WildCat] 运行中 (PID=$(cat "${PID_FILE}")) | 版本: $(current_build_label) | 端口: ${PORT} | Python: ${PY_BIN}"
    else
      echo "[WildCat] 已停止 | 当前代码版本: $(current_build_label)"
    fi
  }

  # Code files shipped in patches / backed up before updates (only those that exist).
  app_code_files() {
    local f out=""
    for f in wildcat.sh wildcat.bat set_admin_password.sh set_admin_password.bat update/README.txt backend frontend webui wildcat tools data/open_experiments.db data/gpu_benchmarks.db train.py infer.py export_cat.py config/default.json; do
      [ -e "${APP_DIR}/${f}" ] && out="${out} ${f}"
    done
    echo "${out}"
  }

  resolve_update_pkg() {
    mkdir -p "${UPDATE_DIR}" "${UPDATE_DIR}/applied"
    local req="${1:-}"
    local candidate=""
    if [ -z "${req}" ]; then
      candidate="$(ls -1t "${UPDATE_DIR}"/*.tar.gz 2>/dev/null | head -n 1 || true)"
      if [ -z "${candidate}" ]; then
        echo "[WildCat] ✕ 未在 ${UPDATE_DIR}/ 目录中找到任何热更新包 (*.tar.gz)！" >&2
        echo "[WildCat] 请先将更新包放入固定目录: ${UPDATE_DIR}/" >&2
        echo "[WildCat] 然后重新执行: ./wildcat.sh update" >&2
        exit 1
      fi
      echo "${candidate}"
      return 0
    fi

    # 若只传了文件名，直接在 update/ 目录内查找
    if [[ "${req}" != */* ]] && [ -f "${UPDATE_DIR}/${req}" ]; then
      echo "${UPDATE_DIR}/${req}"
      return 0
    fi

    # 若传了路径，强制要求文件必须位于项目 update/ 目录下，彻底杜绝路径混乱
    if [ -f "${req}" ]; then
      local req_dir
      req_dir="$(cd "$(dirname "${req}")" && pwd)"
      if [ "${req_dir}" = "${UPDATE_DIR}" ]; then
        echo "${UPDATE_DIR}/$(basename "${req}")"
        return 0
      else
        echo "[WildCat] ✕ 热更新包必须放在项目固定目录: ${UPDATE_DIR}/" >&2
        echo "[WildCat] 当前传入路径不在 update/ 目录内: ${req}" >&2
        echo "[WildCat] 请先执行: cp \"${req}\" \"${UPDATE_DIR}/\" && ./wildcat.sh update" >&2
        exit 1
      fi
    fi

    echo "[WildCat] ✕ 在 ${UPDATE_DIR}/ 中未找到更新包: ${req}" >&2
    echo "[WildCat] 当前 ${UPDATE_DIR}/ 下可用的更新包:" >&2
    ls -lh "${UPDATE_DIR}"/*.tar.gz 2>/dev/null >&2 || echo "  (空)" >&2
    exit 1
  }

  verify_python_modules() {
    echo "[WildCat] 正在执行 Python 核心模块与语法预检..."
    if ! "${PY_BIN}" -c "import traceback, sys
try:
    import wildcat.version as v
    import wildcat.trainer
    import backend.server
    print('[WildCat] ✓ 模块预检通过 -> ' + v.BUILD_LABEL)
except BaseException:
    sys.stderr.write('[WildCat] ✕ 模块预检失败，完整 Python 异常堆栈如下:\n')
    traceback.print_exc()
    sys.exit(1)
" 2>&1 | tee -a "${LOG_FILE}"; then
      echo "[WildCat] ✕ 代码预检未通过！可执行 ./wildcat.sh rollback 一键恢复上一版本。"
      exit 1
    fi
  }

  do_update() {
    local pkg
    pkg="$(resolve_update_pkg "${1:-}")"
    echo "[WildCat] 检测到热更新包: ${pkg}"

    # 1. 停止服务前先校验 tar.gz 完整性
    if ! tar -tzf "${pkg}" >/dev/null 2>&1; then
      echo "[WildCat] ✕ 更新包校验失败（文件损坏或未传输完整）: ${pkg}"
      exit 1
    fi

    echo "[WildCat] 正在执行增量热更新 (仅更新代码层，保留 .venv 运行库、weights 与 storage/run 数据)..."
    mkdir -p "${BASE_DIR}/backups" "${UPDATE_DIR}/applied"
    local ts
    ts="$(date +%Y%m%d_%H%M%S)"
    if [ -d "${APP_DIR}/backend" ]; then
      # shellcheck disable=SC2046
      tar -czf "${BASE_DIR}/backups/app_backup_${ts}.tar.gz" -C "${APP_DIR}" $(app_code_files) 2>/dev/null || true
      echo "[WildCat] 已自动备份当前代码至: backups/app_backup_${ts}.tar.gz"
    fi

    do_stop || true
    rm -rf "${BASE_DIR}/WildCat" "${APP_DIR}/WildCat" 2>/dev/null || true
    if (set +o pipefail; tar -tzf "${pkg}" 2>/dev/null | head -n 5 | grep -q '^WildCat/'); then
      tar -xzf "${pkg}" --strip-components=1 -C "${APP_DIR}"
    else
      tar -xzf "${pkg}" -C "${APP_DIR}"
    fi
    chmod +x "${APP_DIR}/wildcat.sh" "${APP_DIR}/set_admin_password.sh" 2>/dev/null || true
    if [ "${APP_DIR}" != "${BASE_DIR}" ] && [ -f "${APP_DIR}/wildcat.sh" ]; then
      cp -f "${APP_DIR}/wildcat.sh" "${BASE_DIR}/wildcat.sh"
      chmod +x "${BASE_DIR}/wildcat.sh" 2>/dev/null || true
      [ -f "${APP_DIR}/wildcat.bat" ] && cp -f "${APP_DIR}/wildcat.bat" "${BASE_DIR}/wildcat.bat" || true
    fi

    verify_python_modules
    do_start

    local base_pkg archived_pkg
    base_pkg="$(basename "${pkg}")"
    archived_pkg="${UPDATE_DIR}/applied/${base_pkg%.tar.gz}_applied_${ts}.tar.gz"
    mv -f "${pkg}" "${archived_pkg}"
    echo "[WildCat] 已将本次更新包归档至: update/applied/$(basename "${archived_pkg}")"
    echo "[WildCat] ✓ 增量热更新全部完成！当前运行版本: $(current_build_label)"
  }

  do_pack_patch() {
    mkdir -p "${UPDATE_DIR}"
    local arg="${1:-}"
    local out_file=""
    if [ -z "${arg}" ]; then
      out_file="${UPDATE_DIR}/WildCat_$(date +%Y%m%d_%H%M%S)_patch.tar.gz"
    elif [[ "${arg}" != */* ]]; then
      out_file="${UPDATE_DIR}/${arg}"
    else
      out_file="${arg}"
    fi
    # shellcheck disable=SC2046
    tar --exclude='__pycache__' --exclude='*.pyc' --exclude='tools/make_public.py' --exclude='tools/publish_release.sh' -czf "${out_file}" -C "${APP_DIR}" $(app_code_files)
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
    chmod +x "${APP_DIR}/wildcat.sh" "${APP_DIR}/set_admin_password.sh" 2>/dev/null || true
    if [ "${APP_DIR}" != "${BASE_DIR}" ] && [ -f "${APP_DIR}/wildcat.sh" ]; then
      cp -f "${APP_DIR}/wildcat.sh" "${BASE_DIR}/wildcat.sh"
      chmod +x "${BASE_DIR}/wildcat.sh" 2>/dev/null || true
    fi
    verify_python_modules
    do_start
    echo "[WildCat] ✓ 已成功回滚并重启！当前版本: $(current_build_label)"
  }

  do_foreground() {
    do_stop || true
    relocate_venv
    link_persistent_data
    rotate_log_if_needed
    cd "${APP_DIR}"
    local now_str ver_str py_ver ip_addr
    now_str="$(date '+%Y-%m-%d %H:%M:%S')"
    ver_str="$(current_build_label)"
    py_ver="$("${PY_BIN}" --version 2>&1 || true)"
    ip_addr="$(hostname -I 2>/dev/null | awk '{print $1}' || echo '127.0.0.1')"
    {
      echo ""
      echo "================================================================================"
      echo "[WildCat] Startup (foreground) at ${now_str} | Build: ${ver_str} | PY_BIN=${PY_BIN} (${py_ver})"
      echo "================================================================================"
    } >> "${LOG_FILE}"
    echo "[WildCat] 前台守护模式启动中 (PID=$$) | 版本: ${ver_str} | 地址: http://${ip_addr}:${PORT}"
    echo $$ > "${PID_FILE}"
    exec "${PY_BIN}" "${APP_DIR}/webui/app.py" --host 0.0.0.0 --port "${PORT}" >> "${LOG_FILE}" 2>&1
  }

  case "${1:-status}" in
    start)      do_start ;;
    stop)       do_stop ;;
    restart)    do_stop; do_start ;;
    foreground) do_foreground ;;
    status)     do_status ;;
    update)     do_update "${2:-}" ;;
    pack-patch) do_pack_patch "${2:-}" ;;
    rollback)   do_rollback ;;
    *)
      echo "用法: $0 {start|stop|restart|foreground|status|update [文件名.tar.gz]|pack-patch [文件名.tar.gz]|rollback}"
      exit 1
      ;;
  esac
}

main "$@"
