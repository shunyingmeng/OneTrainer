# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OneTrainer is a modular training framework for diffusion models. It supports multiple model architectures (SD 1.5/2.x/3.x, SDXL, FLUX, PixArt, Würstchen, Sana, Hunyuan Video, HiDream, Chroma, Qwen, Z-Image), training methods (full fine-tune, LoRA, embeddings, VAE fine-tune), and model formats (diffusers, ckpt, safetensors). It has both a GUI (CustomTkinter) and CLI mode.

## Setup & Commands

```bash
# Install (creates venv automatically)
./install.sh          # Linux/Mac
install.bat           # Windows

# Activate venv manually
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start GUI
python scripts/train_ui.py

# CLI training
python scripts/train.py --config-path <path-to-json-config>

# Other CLI scripts (all in scripts/)
python scripts/convert_model.py --help
python scripts/sample.py --help
python scripts/generate_captions.py --help
python scripts/generate_masks.py --help
python scripts/calculate_loss.py --help
```

## Code Quality

Ruff is used for linting and formatting. Pre-commit hooks enforce style on commits.

```bash
# Install dev dependencies and hooks (run OUTSIDE venv)
pip install -r requirements-dev.txt
pre-commit install

# Run linting manually
ruff check .
ruff check --fix .
```

Key ruff settings (in `pyproject.toml`):
- Line length: 120
- Import order: stdlib → first-party (`modules`) → mgds → torch → hf (diffusers/transformers) → third-party → local
- Python ≥ 3.10

## Architecture

### Module System

The codebase uses a **plugin factory pattern**. Each module type has a base class, and concrete implementations register themselves via `modules/util/factory.py`. The factory registry is populated at import time — `modules/util/create.py` calls `factory.import_dir()` to auto-discover all implementations under each module directory.

The module types (each a subdirectory of `modules/`):

| Module | Purpose | Base Class |
|--------|---------|------------|
| `model` | Holds weights, optimizers, embeddings | `BaseModel` |
| `modelLoader` | Loads models from disk formats | `BaseModelLoader` |
| `modelSaver` | Saves models to disk formats | `BaseModelSaver` |
| `modelSetup` | Configures training (optimizer, device placement, predictions) | `BaseModelSetup` |
| `modelSampler` | Generates samples during/after training | `BaseModelSampler` |
| `dataLoader` | Loads training data using MGDS graph-based pipeline | `BaseDataLoader` |
| `trainer` | Orchestrates the full training loop | `BaseTrainer` |
| `ui` | GUI code (CustomTkinter tabs) | — |

### Adding Support for a New Model

Requires implementing classes in 6 module directories: `model`, `modelLoader`, `modelSaver`, `modelSetup`, `modelSampler`, `dataLoader`. Each implementation registers itself with the factory using `@factory.register(BaseClass, ModelType.X, TrainingMethod.Y)` decorators or explicit registration calls. Then wire it in `modules/util/create.py` and add the enum value to `ModelType` in `modules/util/enum/ModelType.py`.

### Key Files

- **`modules/util/create.py`** — Central factory that instantiates the correct module implementations based on `ModelType` and `TrainingMethod`. Also creates optimizers, LR schedulers, and noise schedulers.
- **`modules/util/config/TrainConfig.py`** — The main configuration dataclass with all training parameters. Serializes to/from JSON. Supports versioned migrations.
- **`modules/util/enum/`** — Enumerations that drive the factory system: `ModelType`, `TrainingMethod`, `Optimizer`, `DataType`, `NoiseScheduler`, `LearningRateScheduler`, etc.
- **`modules/trainer/GenericTrainer.py`** — The main training loop implementation.
- **`modules/util/factory.py`** — The plugin registry (register/get pattern).

### Training Flow

`scripts/train.py` → loads `TrainConfig` from JSON → `create_trainer()` → `GenericTrainer`:
1. `model_loader.load()` — load model weights
2. `model_setup.setup()` — configure optimizer, move to device
3. Training loop: load batch → forward → backward → optimizer step → periodic sampling/backup
4. `model_saver.save()` — save final model

### Configuration

Training is entirely config-driven via JSON files. The `training_presets/` directory contains pre-built configurations for various model types. The GUI produces and consumes these same JSON configs.

### Scripts

All user-facing entry points are in `scripts/`. Scripts contain no extra logic — they delegate entirely to modules.

### External Dependencies

- **MGDS** (`mgds` package) — Custom graph-based dataset library by the same author. Used for all data loading.
- **diffusers** — Installed from git as an editable package for latest features.
- **CustomTkinter** — GUI framework.

## Training Configs (Test Fixtures)

The `training_configs/` folder contains working config JSONs for quick smoke-testing of code changes:

| File | Model | Notes |
|------|-------|-------|
| `chroma.json` | Chroma-HD (CHROMA_1) | dynamic_timestep_shifting=true |
| `klein.json` | FLUX.2 Klein 9B (FLUX_2) | |
| `illustrious.json` | Illustrious XL (SDXL) | PPSF optimizer, trains UNet + TE1 + TE2 |
| `zimage.json` | Z-Image | Regex layer filter |

All use LoRA training with `split_groups: true`. Most use PRODIGY_ADV; `illustrious.json` uses PRODIGY_PLUS_SCHEDULE_FREE.

**IMPORTANT: Any changes to files in `training_configs/` must be reviewed and approved by the user.** Do not modify these configs without explicit confirmation — they are known-good baselines for testing.

## Git Remotes

- `origin` — upstream Nerogar/OneTrainer (HTTPS, read-only)
- `fork` — user's fork at `git@github-sy:shunyingmeng/OneTrainer.git` (SSH, push here)

Always push to `fork`, not `origin`. The SSH host `github-sy` is configured in `~/.ssh/config` for the user's secondary GitHub account.
