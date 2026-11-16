#!/usr/bin/env bash
# WildCat 修改管理后台密码（不需要重启服务）
# 用法：把本文件放到 WildCat 目录下，运行：  bash set_admin_password.sh
set -e
cd "$(dirname "$0")"
if [ ! -d backend ]; then
  echo "请把本脚本放在 WildCat 目录里（和 wildcat.sh 同一层）再运行。"; exit 1
fi
PY=""
for c in ./.venv/bin/python ./.venv/cpython-3.12/bin/python3.12 ./runtime/python/cpython-3.12/bin/python3.12; do
  if [ -x "$c" ]; then PY="$c"; break; fi
done
[ -n "$PY" ] || { echo "找不到 WildCat 自带的 Python（.venv）。"; exit 1; }
read -r -s -p "请输入新的管理员密码: " P1; echo
read -r -s -p "请再输入一次: " P2; echo
if [ "$P1" != "$P2" ]; then echo "两次输入不一致，未修改。"; exit 1; fi
if [ ${#P1} -lt 4 ]; then echo "密码至少 4 位，未修改。"; exit 1; fi
WC_NEW_ADMIN_PASSWORD="$P1" "$PY" - <<'PYEOF'
import os, sys
sys.path.insert(0, os.getcwd())
from backend import db
db.init_db()
db.update_admin_password(os.environ["WC_NEW_ADMIN_PASSWORD"])
print("管理员密码已修改。无需重启服务，刷新网页后用新密码登录管理后台即可。")
PYEOF
