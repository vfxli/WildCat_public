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
- **About 1.8×–5.3× faster per training run on the same GPU**: VRAM dataset residency, full-step CUDA Graph capture, fused Adam, `channels_last` layout and asynchronous preview on a secondary GPU shorten training without changing its result;
- **Seamless Nuke Integration**: Full ACEScg linear 16-bit Half OpenEXR pipeline with real-time OCIO sRGB browser preview. Exported TorchScript (`.pt`) checkpoints convert directly via Nuke's built-in **CatFileCreator** node for **Inference**;
- **Optional enhancements**: 6 advanced setting modules and 6 task presets (plus one click back to the Nuke defaults), all off by default; with them off, training is the same as CopyCat's. The test method and results of each one are on the "Strategy Evidence (Advanced)" page of the web help center (re-validated in 2026-11: 3 random seeds per configuration, scored only on frames the model never saw, paired against the phase-aligned CopyCat defaults); `data/open_experiments.db` holds the raw records of these tests (752 training runs, see "Experiment Database") and the archive of 206 early single-seed comparison runs;
- **Face editing**: the "Keyframe commonality" module (where hand-painted keyframes disagree, it changes the regions of single keyframes that disagree with the others before training; DIS optical flow), the "Face edit" preset (Medium + 1 px crop grid + Charbonnier 0.25 + keyframe commonality + averaged weights) and the "Training quality (in-between frames)" monitor on the task page;
- **Help next to every setting**: every parameter and metric in the web UI has a "?" - hover for a summary, click for the full entry in the help center.

---

## Key Features

1. **Bilingual Web Studio & Native Windows / Linux Support**
   - Instant **English / 简体中文** toggle in the top bar (zero-hardcoded locale architecture; drop any `<lang>.json` into `frontend/locales/` for auto-discovery);
   - Turnkey service scripts for both **Windows** (`wildcat.bat`) and **Linux** (`wildcat.sh`) with hot-update and one-click rollback.
2. **6 Task Presets + Optional Advanced Settings**
   - One-click presets for **Clean-up, Keying, Defocus, Degrain, Roto and Face edit**: model size, channels and the related switches (set from the 2026-11 re-validation; presets leave Epochs alone);
   - 6 optional advanced modules: crop grid (1 px, with grid anchor), composite loss (Charbonnier / gradient term / highlight damping), stage ratios and transition warm-up, averaged weights (EMA), augmentation (horizontal flip / exposure jitter) and keyframe commonality; each one has its explanation and test evidence in the help center. With all of them off, training follows CopyCat's default behaviour.
3. **Checkpoint Recommender & Auto-Extend**
   - When a validation sequence is attached, WildCat suggests two checkpoints from the `1/1` full-resolution stage: **Balanced** = the last full-resolution checkpoint, **Detail** = the checkpoint that fits the edited area of the keyframes best (a training-frame fit only); optional auto-extend while training-frame accuracy still improves (off by default). The rules were simplified in 2026-11 after the re-validation; `HFI` is not a flicker measure and takes part in no decision.
4. **Experiments Lab & Dynamic Multi-GPU Pool**
   - Queue batch hyperparameter sweeps or parameter-combination suites, overlay multi-run loss/validation curves, stream full sequences to inspect frame-to-frame difference heatmaps (`ΔR`, `HFI`, `Static Boil`), and export CSV reports;
   - Dynamically enable/disable any detected GPU (`0..N`) and assign **Primary** (first-choice training) and **Secondary** (parallel overflow + zero-contention preview/eval) roles in Admin, backed by `data/gpu_benchmarks.db` ETA estimation.

---

## Experiment Database (`data/open_experiments.db`)

The bundled SQLite database holds two sets of experiments, all as raw records:

- **Experiments from 2026-11 on** (`exp2026_*` tables, 752 training runs): the re-validation, 394 runs (7 general datasets; 3 random seeds per configuration, scored only on frames the model never saw, paired against the phase-aligned CopyCat defaults);
  24 supplementary runs, 3 preset checks, 133 runs of the paired test of the Face edit preset with and without Charbonnier (5 public face shots × 4 keyframe conditions × 3 seeds × with / without, plus re-runs of the same jobs on a second platform), and 198 keyframe-commonality research runs.
  Every figure on the help center's "Strategy Evidence (Advanced)" page comes from these records. `exp2026_about` describes the tables; each run carries its settings, GPU model and torch version, speed, the scores at every evaluation point (raw / averaged weights, unseen / training frames) and the per-frame scores at the final step.
  Not included: internal production material, shots made with face-swap models under non-commercial licences, user-supplied material; machine names, paths and code hashes are not written either.
- **Early comparison runs** (`experiment_runs` and the other Lab tables, 347 records, the seed data of the Experiments page): the archive of 206 single-seed experiments. Please keep its limits in mind before using any number from it:

- every configuration was run with a single random seed; repeating the same configuration can move the held-out score by about 0.5 dB;
- each dataset has only 2 held-out frames, and the same frames were used to choose configurations, so they are not an independent test set;
- 6 of the 9 datasets are synthetic;
- on 3 datasets the plate height is not a multiple of 4 and the default-configuration baseline had an evaluation issue; those comparisons have been withdrawn from this page.

The table keeps only the two items that can be recomputed from the raw records and are not affected by that issue (held-out PSNR at the end of training). They are still single-seed results:

| Dataset | Default configuration | With the option(s) on | Difference | Option(s) |
| --- | --- | --- | --- | --- |
| SmartVector marker removal (synthetic: decals composited on a live plate, 960×540) | 54.09 dB | 59.02 dB | +4.93 dB | `Small`, `1px` crop grid only |
| Greenscreen keying & despill (ground truth generated by a formula, 960×540, 3→4 channels) | 61.45 dB | 64.40 dB | +2.95 dB | `Small`, `1px` + composite loss (0.30 / 0.15 / 0.35) |

The 2026-11 re-validation is complete (3 random seeds per configuration, 24–83 unseen frames per dataset, paired against the phase-aligned CopyCat defaults): the conclusions and figures of every strategy are on the "Strategy Evidence (Advanced)" page of the web help center, and the task presets follow them.

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
