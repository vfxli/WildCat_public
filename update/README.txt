WildCat 热更新目录 (update/) 使用说明
=====================================

为避免因路径不一致导致更新失败，所有增量热更新包 (*.tar.gz) 必须放入本目录 (update/) 中执行更新。

【Linux 更新步骤】
1. 将热更新包 (例如 WildCat_20261111_patch.tar.gz) 放入本目录:
     <项目根目录>/update/
2. 在项目根目录执行一键热更新 (自动识别 update/ 下最新的更新包):
     ./wildcat.sh update
   （若 update/ 下有多个更新包，也可指定文件名: ./wildcat.sh update WildCat_20261111_patch.tar.gz）

【Windows 更新步骤】
1. 将热更新包 (例如 WildCat_20261111_patch.tar.gz) 放入本目录:
     <项目根目录>\update\
2. 在项目根目录执行一键热更新:
     wildcat.bat update

【自动化保护机制】
- 更新前自动校验压缩包完整性，并自动备份旧代码至 backups/ 目录。
- 解压后自动执行 Python 核心模块预检；若出现异常会直接打印完整 Python Traceback 堆栈。
- 更新成功并重启服务后，已应用的更新包会自动归档至 update/applied/ 目录，防止下次重复误用旧包。
- 如需回退到上一版本，执行:
     ./wildcat.sh rollback    (Linux)
     wildcat.bat rollback     (Windows)
