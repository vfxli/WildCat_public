# WildCat

**[简体中文](README.md)** | **English**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Platform: Linux | Windows](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-0078D4.svg)](#installation--quick-start)
[![UI: 中文 | English](https://img.shields.io/badge/UI-中文%20%7C%20English-10B981.svg)](#key-features)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg)](https://www.python.org/)
[![PyTorch 2.8–2.11](https://img.shields.io/badge/PyTorch-2.8%E2%80%932.11-EE4C2C.svg)](https://pytorch.org/)

**WildCat** is a high-speed sequence neural training, batch ablation, and inference studio built for **VFX compositing workflows**, running natively on **Windows** and **Linux** with a hot-switchable **English / Simplified Chinese** web interface.

WildCat mirrors **Nuke CopyCat**'s training pipeline while remaining 100% compatible with Nuke's inference workflow:

- **100% Default Parity with Nuke 17 CopyCat**: With optional advanced knobs off, parameter names, units, 4px bottom-anchored sampling grids, 3-level pyramid MSE supervision, and `poly(0.3)` LR schedules match CopyCat faithfully;
- **2.1×–5.5× Faster on the Same GPU**: VRAM dataset residency, full-step CUDA Graph capture, fused Adam, `channels_last` layout, and zero-contention secondary-GPU async preview slash training wall-clock time without altering mathematical convergence;
- **Seamless Nuke Integration**: Full ACEScg linear 16-bit Half OpenEXR pipeline with real-time OCIO sRGB browser preview. Exported TorchScript (`.pt`) checkpoints convert directly via Nuke's built-in **CatFileCreator** node for **Inference**;
- **Backed by a 153M-Epoch (750M Shadow-Epoch) Blind-Tested Study**: Ships with an embedded SQLite benchmark & experiment archive (`data/open_experiments.db`) distilled from **206 full runs across 9 real VFX shot families** on a **16× RTX 5090 cluster**, powering one-click VFX task presets and intelligent validation checkpoint selection.

---

## Key Features

1. **Bilingual Web Studio & Native Windows / Linux Support**
   - Instant **English / 简体中文** toggle in the top bar (zero-hardcoded locale architecture; drop any `<lang>.json` into `frontend/locales/` for auto-discovery);
   - Turnkey service scripts for both **Windows** (`wildcat.bat`) and **Linux** (`wildcat.sh`) with hot-update and one-click rollback.
2. **5 Empirically Validated VFX Presets + Optional Advanced Hyperparameters**
   - One-click presets for **Grained Plate Cleanup, Greenscreen Keying & Despill, Bokeh / Focus Pull & Motion Blur, Sensor Degrain & Clean Marker Removal, and Articulated Roto Mattes**;
   - 5 rigorously blind-tested optional knobs (`1px` all-phase crop sampling, Charbonnier + L1 spatial edge gradient + ACEScg highlight damping composite loss, `15/25/60` bokeh stage weighting, `Stage-Reset EMA 0.9995`, and paired horizontal flip), each documented with side-by-side CopyCat comparisons and LaTeX math.
3. **Intelligent Validation Checkpoint Recommender & Auto-Extend**
   - When an evaluation sequence is attached, WildCat gates selection to the `1/1` full-resolution stage, penalizes 2nd-order temporal flicker (`HFI`) creep and training-frame memorization, and recommends **Balanced Best**, **Lowest Playback Flicker**, and **Sharpest ROI Detail** checkpoints (cutting unseen selection regret by **46.3%** across 146 real trajectories), plus optional auto-extension if convergence has not plateaued.
4. **Experiments Lab & Dynamic Multi-GPU Pool**
   - Queue batch hyperparameter sweeps or curated Best-Combo exploration suites, overlay multi-run loss/validation curves, stream full sequences to inspect frame-by-frame temporal flicker heatmaps (`ΔR`, `HFI`, `Static Boil`), and export CSV reports;
   - Dynamically enable/disable any detected GPU (`0..N`) and assign **Primary** (first-choice training) and **Secondary** (parallel overflow + zero-contention preview/eval) roles in Admin, backed by `data/gpu_benchmarks.db` ETA estimation.

---

## Summary of Blind-Test Findings Across 9 VFX Shot Families

All conclusions are judged strictly by **held-out unseen frames (Unseen PSNR ↑)** and **100-frame playback flicker (`HFI` ↓)**:

| VFX Shot Family | CopyCat Default PSNR | WildCat Recommended PSNR | Net Gain / Key Production Benefit | Core Recipe |
| --- | --- | --- | --- | --- |
| Clean Marker Cleanup (SmartVector) | 54.09 dB | **59.02 dB** | **+4.93 dB** (eliminates 4px stride phase blind spot) | `Small · 1px All-Phase` |
| Greenscreen Keying & Despill (3→4 RGBA) | 61.45 dB | **64.40 dB** | **+2.95 dB** (razor-sharp hair · linear Alpha preserved) | `Small · 1px + Grad 0.15 + HDR 0.35` |
| Large Bokeh / Focus Pull Recovery | 33.83 dB | **43.55 dB** | **+9.72 dB** (wide receptive field required) | `Large · 15/25/60 Stage + HDR 0.50` |
| Fast Multi-Frame Motion Blur Recovery | 43.25 dB | **47.13 dB** | **+3.88 dB** (captures full motion trajectory) | `Large · 1px + Grad 0.10 + Stage-EMA` |
| Live-Action Facial Scar Cleanup (Grained) | 30.49 dB | **30.83 / 30.79 dB** | **-25.9% playback flicker · 4.1× cleaner untouched BG** | `Small + HFlip 0.5` or `Medium Combo` |
| 2K High-Res Beauty / Blemish Cleanup | 38.25 dB | **39.59 dB** | **+1.34 dB** (wider 384px context on 2K plates) | `Small/Med · Crop 384 + Composite Loss` |
| Low-Light Sensor Degrain | 41.32 dB | **41.78 dB** | **+0.46 dB** (peak around ~70% steps avoids grain fit) | `Small · 1px + C0.25/G0.05/H0.50` |
| Articulated Dance / Parkour Roto Matte | 15.61 dB | **16.30–17.30 dB** | **10.2× cleaner blackmatte floor** (pose coverage wins) | `Small · 3→1ch · HDR=0 + Stage-EMA` |

---

## Using Exported Models in Nuke

1. Create a **CatFileCreator** node in Nuke;
2. Point `Torch Script File` to the `.pt` model downloaded from WildCat (or `run/<task_name>/Training_*.<step>.pt`);
3. Set `Channels In` and `Channels Out` matching your training channels (e.g., `rgba.red, rgba.green, rgba.blue`; add `rgba.alpha` for 4-channel keying or use `rgba.alpha` alone for 1-channel roto mattes);
4. Choose an output `Cat File` path and click **Create .cat file**;
5. Load the resulting `.cat` file in Nuke's **Inference** node.

> **PyTorch Version Note**: Nuke 17's bundled libtorch cannot load TorchScript archives saved by PyTorch ≥ 2.12. Install `torch>=2.8.0,<2.12` (recommended: `torch==2.11.0+cu128`).

---

## Installation & Quick Start

Supports **Windows 10 / 11 / Server 2022** and **Linux** (Ubuntu 22.04+ / Rocky Linux 9), NVIDIA Driver ≥ 570, Python 3.12.

### Windows (CMD / PowerShell)

```bat
git clone https://github.com/vfxli/WildCat_public.git WildCat && cd WildCat
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
wildcat.bat start
```

### Linux

```bash
git clone https://github.com/vfxli/WildCat_public.git WildCat && cd WildCat
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
./wildcat.sh start
```

Open `http://127.0.0.1:8765` in your browser. On first launch (if `WC_PASSWORD` is unset), a random initial admin password is generated and logged in `wildcat.log`. You can change it anytime via `set_admin_password.bat` or `bash set_admin_password.sh`.

---

Nuke and CopyCat are products of Foundry.

## License

[MIT](LICENSE)
