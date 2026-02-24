# OneTrainer

Modular training framework for diffusion models. Supports multiple architectures (SD 1.5/2.x/3.x, SDXL, FLUX, PixArt, Wuerstchen, Sana, Hunyuan Video, HiDream, Chroma, Qwen, Z-Image), training methods (full fine-tune, LoRA, embeddings, VAE fine-tune), and model formats (diffusers, ckpt, safetensors). GUI (CustomTkinter) and CLI.

## Setup & Commands

```bash
# Install (creates venv)
./install.sh          # Linux/Mac
install.bat           # Windows

# Activate venv
source venv/bin/activate

# Run GUI
python scripts/train_ui.py

# CLI training
python scripts/train.py --config-path <path-to-json-config>

# Other CLI scripts
python scripts/convert_model.py --help
python scripts/sample.py --help
python scripts/generate_captions.py --help
```

## Code Quality

Ruff for linting/formatting. Pre-commit hooks enforce style.

```bash
pip install -r requirements-dev.txt && pre-commit install   # run OUTSIDE venv
ruff check .         # lint
ruff check --fix .   # auto-fix
```

Key settings (`pyproject.toml`): line length 120, Python >= 3.10, import order: stdlib > first-party (`modules`) > mgds > torch > hf (diffusers/transformers) > third-party > local.

## Architecture

### Plugin Factory Pattern

Each module type has a base class; implementations register via `modules/util/factory.py`. Discovery is automatic via `factory.import_dir()` called from `modules/util/create.py`.

| Module | Purpose | Base Class |
|--------|---------|------------|
| `model` | Weights, optimizers, embeddings | `BaseModel` |
| `modelLoader` | Load from disk | `BaseModelLoader` |
| `modelSaver` | Save to disk | `BaseModelSaver` |
| `modelSetup` | Training config (optimizer, device, predict/loss) | `BaseModelSetup` |
| `modelSampler` | Sample generation | `BaseModelSampler` |
| `dataLoader` | Data pipeline (MGDS graph-based) | `BaseDataLoader` |
| `trainer` | Training loop orchestration | `BaseTrainer` |

### Adding a New Model

Implement classes in 6 module dirs (model, modelLoader, modelSaver, modelSetup, modelSampler, dataLoader), register with factory, add `ModelType` enum in `modules/util/enum/ModelType.py`, wire in `modules/util/create.py`.

### Key Files

- `modules/util/create.py` — Central factory, instantiates modules by `ModelType` + `TrainingMethod`
- `modules/util/config/TrainConfig.py` — Main config dataclass, JSON serialization, versioned migrations
- `modules/util/enum/` — Enums driving the factory: `ModelType`, `TrainingMethod`, `Optimizer`, etc.
- `modules/trainer/GenericTrainer.py` — Main training loop
- `modules/util/factory.py` — Plugin registry

### Training Flow

`scripts/train.py` > `TrainConfig` > `create_trainer()` > `GenericTrainer`:
1. `model_loader.load()` — load weights
2. `model_setup.setup()` — optimizer, device placement
3. Training loop: batch > forward > backward > step > periodic sample/backup
4. `model_saver.save()` — save final model

### External Dependencies

- **MGDS** — Graph-based dataset library (same author)
- **diffusers** — Installed from git (editable) for latest features
- **CustomTkinter** — GUI framework

## Training Configs (Test Fixtures)

`training_configs/` contains known-good configs for smoke-testing:

| File | Model | Notes |
|------|-------|-------|
| `chroma.json` | Chroma-HD (CHROMA_1) | dynamic_timestep_shifting |
| `klein.json` | FLUX.2 Klein 9B (FLUX_2) | |
| `illustrious.json` | Illustrious XL (SDXL) | PPSF optimizer, trains UNet+TE1+TE2 |
| `zimage.json` | Z-Image | Regex layer filter |

All use LoRA with `split_groups: true`. **Do not modify these without explicit user approval.**

## Git Remotes

- `origin` — upstream `Nerogar/OneTrainer` (read-only)
- `fork` — `git@github-sy:shunyingmeng/OneTrainer.git` (push here)

Always push to `fork`. SSH host `github-sy` is configured in `~/.ssh/config`.
