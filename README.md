# WildCat

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg)](https://www.python.org/)
[![PyTorch 2.8–2.11](https://img.shields.io/badge/PyTorch-2.8%E2%80%932.11-EE4C2C.svg)](https://pytorch.org/)

WildCat 是一个用于 VFX 序列帧的神经网络训练与推理工具，支持在 **Linux** 与 **Windows** 上以服务形式运行，通过浏览器访问。训练流程参考 Nuke **CopyCat**，与 CopyCat 兼容：

- 训练参数、单位与默认值与 CopyCat 相同（Epochs / Batch Size / Crop Size / Model Size / Multi-Resolution / Initial Weights / Checkpoint Interval / Contact Sheet Interval）；
- 训练得到的模型以 TorchScript（`.pt`）导出，经 Nuke 自带的 **CatFileCreator** 节点转换为 `.cat` 后，可在 **Inference** 节点中使用；
- 执行层做了加速（数据常驻显存、整步 CUDA Graph 等），不改变训练算法。

数据管线使用 ACEScg 线性空间与 OpenEXR（16-bit half），预览经 OCIO 转换为 sRGB 显示。

---

## 与 CopyCat 的对应关系

| CopyCat 控件 | WildCat | 说明 |
| --- | --- | --- |
| Epochs（默认 10000） | Epochs | Total Steps = Epochs × 数据集帧数 / Batch Size |
| Batch Size：Auto / Manual | Auto (8) / 手动 | Auto = 8，显存不足时自动减小；Batch 不大于帧数 |
| Crop Size（默认 256） | Crop Size | 32 / 64 / 128 / 256 / 512 |
| Model Size：Small / Medium / Large | 同 | |
| Use Multi-Resolution Training | Multi-Resolution | 三段等长：1/4 → 1/2 → 1/1 |
| Initial Weights：None / Checkpoint | 同 | 从 WildCat 的 `.pt` 检查点继续训练时自动锁定模型规格与通道 |
| Checkpoint Interval（默认 1000 步） | Checkpoint Interval | 每次保存 `Training_YYMMDD_HHMMSS.<step>.pt`（TorchScript） |
| Contact Sheet Interval（默认 100 步） | Contact Sheet Interval | 每次保存 `Training_YYMMDD_HHMMSS.<step>.png`（输入 \| Ground Truth \| 输出） |
| Automatic Mixed Precision：Adaptive / Disabled / Force Enabled | Mixed Precision | 默认 Adaptive |
| Learning Rate（1e-4） / Seed（0） | Learning Rate / Seed | |

任务目录 `run/<任务名>/`：

```text
run/<任务名>/
├── Training_261110_041929.1000.pt      # 每个 Checkpoint Interval 一个 TorchScript 模型（网页支持单版本或一键打包下载）
├── Training_261110_041929.100.png      # contact sheet
├── Training_261110_041929.summary      # step,loss,step,loss,...
├── ckpt/step_001000.pt  latest.pt      # 断点续训用（含优化器状态）
├── contact_sup/step_000100.png         # 3×3 局部块实时预览（对齐 CopyCat Viewer 左下角 Contact Sheet：crop 1..3 × input|groundtruth|output）
├── preview_sup/<帧名>/step_000100.png  # 整帧预览（Web 端播放器使用）
└── train.log                           # 实时训练日志与完整异常堆栈
```

---

## 在 Nuke 中使用

1. 在 Nuke 中创建 **CatFileCreator** 节点；
2. Torch Script File 选择 WildCat 导出的 `.pt`（任务目录下的 `Training_*.<step>.pt`，或网页「下载模型 (.pt)」得到的文件）；
3. Channels In / Channels Out 填 `rgba.red, rgba.green, rgba.blue`（单通道输出时 Channels Out 填 `rgba.alpha`）；
4. 设置 Cat File 输出路径，点击 Create .cat file；
5. 在 **Inference** 节点的 Model File 中选择生成的 `.cat`。

Nuke 17 内置的 libtorch 无法读取 PyTorch 2.12 及以后版本保存的 TorchScript 文件，导出时请使用 PyTorch 2.11 或更早版本（训练与导出使用同一环境即可）。

---

## 加速选项

训练页「加速选项」面板中的选项只改变执行方式，不改变训练结果，默认全部开启：
数据常驻显存（半精度 EXR 以 fp16 无损存放）、整步 CUDA Graph、融合 Adam、NHWC 布局、cuDNN 自动选算法、异步写盘、预览与验证在第二张显卡上异步推理。
面板右上角为总开关；全部关闭时为不做执行优化的实现，用于对照验证。

---

## 速度

Nuke 17.0v2 CopyCat 与 WildCat（默认设置）在同一台机器、同一张 GPU 上分别训练；
数据为 40 帧 772×855 半精度 EXR，Multi-Resolution 开启，Mixed Precision = Adaptive。
表中为 1/4 · 1/2 · 1/1 三个阶段稳定段的 steps/s；整个训练耗时之比按三段等长计算。

| GPU | 模型 | Batch | Crop | CopyCat | WildCat | 耗时之比 |
| --- | --- | --- | --- | --- | --- | --- |
| RTX 4090 | Small | 4 | 256 | 68.8 · 66.4 · 55.2 | 511 · 344.7 · 164.5 | 4.4 |
| RTX 4090 | Small | 8 | 256 | 68.6 · 64 · 41.6 | 421.3 · 242.8 · 94.5 | 3.2 |
| RTX 4090 | Medium | 4 | 256 | 64.5 · 61.6 · 44.3 | 375.8 · 232.6 · 98.7 | 3.2 |
| RTX 4090 | Medium | 8 | 256 | 64.4 · 51.8 · 30.9 | 304.2 · 155 · 55.5 | 2.4 |
| RTX 4090 | Medium | 8 | 512 | 48.1 · 17.1 · 8.7 | 155 · 55.3 · 14.1 | 2.0 |
| RTX 4090 | Large | 4 | 256 | 30 · 27.6 · 19.4 | 156.7 · 97.9 · 40.9 | 3.0 |
| RTX 4090 | Large | 8 | 256 | 30.4 · 20.4 · 13 | 128.3 · 64.5 · 23 | 2.4 |
| RTX 5090 | Small | 4 | 256 | 67.3 · 63.6 · 55.1 | 585.8 · 409.3 · 198.3 | 5.3 |
| RTX 5090 | Small | 8 | 256 | 64.8 · 61.9 · 42.6 | 507.1 · 299.7 · 119 | 4.0 |
| RTX 5090 | Medium | 4 | 256 | 64.5 · 63.5 · 43.5 | 410.5 · 274.7 · 124.8 | 3.8 |
| RTX 5090 | Medium | 8 | 256 | 60.6 · 55.6 · 36 | 291.6 · 162.4 · 71.8 | 2.6 |
| RTX 5090 | Medium | 8 | 512 | 54.4 · 35.7 · 13.1 | 183.7 · 71.9 · 20 | 1.8 |
| RTX 5090 | Large | 4 | 256 | 27.8 · 27.1 · 20.2 | 174.2 · 116.4 · 52.1 | 3.7 |
| RTX 5090 | Large | 8 | 256 | 29.5 · 23.3 · 15.2 | 141.4 · 76.7 · 29.5 | 2.6 |

### 20 万 Epochs（35 万步）训练精度对照

在相同参数（7 帧 772×855 半精度 EXR，Medium 模型，Batch 4，Crop 256，开启 Multi-Resolution，Seed 0）下运行 200,000 Epochs（共 350,000 步），1/4 → 1/2 → 1/1 三阶段训练 Loss 收敛曲线与 Nuke 17.0v2 CopyCat 全程重合；各检查点在 Nuke Inference 节点中评测的整帧输出 MSE 完全处于同一水平（35 万步最终 MSE：CopyCat `0.000126`，WildCat `0.000109`）：

![200k-Epoch Loss Parity](docs/loss_parity_200k.png)

### 显存

不同显存容量使用同一套自动策略：

1. **Auto Batch**：取 8；启动前实测一步训练的峰值显存，放不下时自动减半。
2. **数据放置**：每个阶段开始时，若该阶段数据集不超过空闲显存的 60%，整体常驻显存；否则改为内存预取（后台线程裁剪到锁页内存，异步拷贝）。实测 200 帧 4K（约 21 GB）走内存预取时，速度与常驻显存相差 0~4%。
3. **预览与验证**：有第二张显卡时在副卡上异步推理；只有一张卡时在同卡副流上异步执行，忙时跳过本次预览。
4. **大分辨率推理**：整帧推理显存不足时自动改为 1024 分块（128 像素重叠）。

| 显存 | 常驻显存的数据量（半精度 EXR） |
| --- | --- |
| 16 GB | 约 9 GB：4K 约 85 帧 / 2K 约 320 帧 |
| 24 GB | 约 14 GB：4K 约 130 帧 |
| 32 GB | 约 18 GB：4K 约 170 帧 |

环境变量 `WILDCAT_VRAM_LIMIT_GB` 可限制单个任务使用的显存。

---

## 安装与启动

支持 **Linux**（Ubuntu 22.04+ / Rocky Linux 9）与 **Windows**（Windows 10 / 11 / Server 2022），NVIDIA 驱动 ≥ 570，Python 3.12。

Linux：

```bash
git clone https://github.com/vfxli/WildCat_public.git WildCat && cd WildCat
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
./wildcat.sh start
```

Windows（CMD / PowerShell）：

```bat
git clone https://github.com/vfxli/WildCat_public.git WildCat && cd WildCat
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
wildcat.bat start
```

浏览器打开 `http://<服务器IP>:8765`。

管理后台密码：首次启动前可通过环境变量 `WC_PASSWORD` 指定；未指定时，首次启动会随机生成一个初始密码并写入 `wildcat.log`，登录管理后台后请修改（也可随时运行 `bash set_admin_password.sh` 或 `set_admin_password.bat` 修改）。

Linux 服务管理：

```bash
./wildcat.sh start | stop | restart | status
./wildcat.sh pack-patch                     # 打包代码更新包（默认输出到 update/ 目录）
./wildcat.sh update                         # 将更新包放入 update/ 目录后一键更新（保留环境与数据）
./wildcat.sh rollback                       # 回滚到上一个版本
```

Windows 服务管理：

```bat
wildcat.bat start | stop | restart | status
wildcat.bat pack-patch                      # 打包代码更新包（默认输出到 update\ 目录）
wildcat.bat update                          # 将更新包放入 update\ 目录后一键更新（保留环境与数据）
wildcat.bat rollback                        # 回滚到上一个版本
```

### 命令行

```bash
python train.py config/default.json                 # 训练
python infer.py config/default.json dataset/eval    # 序列推理，输出 16-bit half EXR
python export_cat.py config/default.json            # 导出 TorchScript 模型 (.pt)
python tools/benchmark.py --src 'a/src.####.exr' --gt 'a/gt.####.exr' --model Medium --batch 8
```

---

## 目录结构

```text
WildCat/
├── backend/        # FastAPI 服务、多 GPU 任务队列、SQLite、遥测
├── frontend/       # Vue 3 + Tailwind 工作台
├── webui/          # Web 服务入口
├── wildcat/
│   ├── version.py  # 版本号与构建标识
│   ├── models.py   # Small / Medium / Large 模型与权重加载
│   ├── data.py     # EXR 序列加载、多分辨率、取帧与裁剪、显存常驻 / 内存预取
│   ├── engine.py   # 单步训练（混合精度、损失缩放、Adam），CUDA Graph 捕获
│   ├── trainer.py  # 训练流程、检查点、TorchScript 导出
│   ├── preview.py  # 异步预览 / 验证 / 3×3 contact sheet
│   ├── color.py    # OpenEXR 读写与 OCIO
│   └── cat_format.py  # TorchScript 读取与导出
├── update/         # 热更新包目录（放入 .tar.gz 后执行 update 一键升级）
├── tools/benchmark.py
├── config/default.json
├── wildcat.sh
└── wildcat.bat
```

Nuke 与 CopyCat 为 Foundry 的产品。

## License

[MIT](LICENSE)
