# Branch: dev/unified-prodigy-optimizer

Unified Prodigy optimizer with per-group adaptive learning rates. Based on `master`.

## Goal

Unify `PRODIGY_ADV` and `PRODIGY_PLUS_SCHEDULE_FREE` into a single optimizer by adding per-parameter-group adaptive `d` (learning rate discovery). This branch implements the first step: `split_groups` support for Prodigy ADV.

## Changes from master

### Includes shared fixes (also in dev/enhance-and-fix)
- Dynamic timestep shifting fix for Flux2 and Z-Image
- Sampler timestep shift config support
- CEP validation guard and scaling formula fix
- Multi-level validation (high/mid/low timesteps)
- Configurable deterministic timestep fraction

### Unique to this branch
- **`ProdigyAdvSplitGroups`** (`modules/util/optimizer/prodigy_adv_split.py`): Subclass of `prodigy_optimizer.Prodigy` that gives each parameter group its own adaptive `d` value, allowing UNet, text encoders, etc. to independently discover optimal learning rates
- **Optimizer wiring** (`modules/util/create.py`): Refactored `_create_prodigy_adv_optimizer()` to use `ProdigyAdvSplitGroups` when `split_groups=True`, with cleaner parameter group construction
- **Enum registration** (`modules/util/optimizer_util.py`): Registered new optimizer class

## Files Modified

- `modules/util/optimizer/prodigy_adv_split.py` — new file, 179 lines
- `modules/util/create.py` — refactored optimizer creation (72 lines changed)
- `modules/util/optimizer_util.py` — import registration

## TODO

- [ ] Add schedule-free wrapper option to `ProdigyAdvSplitGroups`
- [ ] Merge PRODIGY_PLUS_SCHEDULE_FREE functionality into the unified optimizer
- [ ] Add UI toggle for split_groups in optimizer settings
- [ ] Test with multi-TE models (SDXL, HiDream) to verify per-group d convergence
- [ ] Consider deprecating separate PRODIGY_PLUS_SCHEDULE_FREE enum once unified

## Architecture Notes

- `ProdigyAdvSplitGroups` overrides `step()` to maintain separate `d` and `d_numerator`/`d_denom` accumulators per group, rather than the single global `d` in base Prodigy
- The `create.py` refactor groups parameters by component (transformer, text_encoder_1, text_encoder_2, etc.) before passing to the optimizer, enabling meaningful per-group adaptation
