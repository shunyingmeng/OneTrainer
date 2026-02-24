# Branch: dev/enhance-and-fix

Training quality enhancements and bug fixes for flow matching models. Based on `master`.

## Changes from master

### Bug Fixes
- **Dynamic timestep shifting** (`Flux2Model`, `ZImageModel`): Replace linear interpolation from scheduler config with BFL empirical mu coefficients (`a=0.00016927, b=0.45666666`), fixing incorrect shift values
- **Sampler timestep shift** (`Flux2Sampler`, `FluxSampler`, `QwenSampler`, `ZImageSampler`): Samplers now respect `dynamic_timestep_shifting` and `timestep_shift` from `SampleConfig` instead of always using dynamic shifting
- **CEP skipped during validation** (all `Base*Setup.py`): `_apply_conditional_embedding_perturbation()` now gated on `not deterministic` to avoid perturbing validation predictions
- **CEP scaling formula** (`ModelSetupNoiseMixin`): Fixed from `sqrt(gamma/d)` to `gamma/sqrt(d)` per paper equation (8); method made `@staticmethod`

### Enhancements
- **Multi-level validation** (`GenericTrainer`): Validation now runs at three timestep levels (high=0.85, mid=0.5, low=0.15) logged as `loss/validation_high/`, `loss/validation_mid/`, `loss/validation_low/` in TensorBoard
- **Configurable deterministic timestep** (`ModelSetupNoiseMixin`): `_deterministic_timestep_fraction` field replaces hardcoded `0.5`, set by trainer per validation level
- **MIN_SNR_GAMMA loss weighting** (`ModelSetupDiffusionLossMixin`): New `LossWeight.MIN_SNR_GAMMA` option for flow matching models — `min(SNR, gamma) / (SNR + 1)` weighting controlled by `loss_weight_strength`
- **SampleConfig additions** (`SampleConfig`): Added `dynamic_timestep_shifting` (bool) and `timestep_shift` (float) fields

### Config Changes
- `illustrious.json`: Switched from PRODIGY_ADV to PRODIGY_PLUS_SCHEDULE_FREE as baseline

## Files Modified

Key files (excluding `training_configs/` and `CLAUDE.md`):
- `modules/model/Flux2Model.py`, `modules/model/ZImageModel.py` — timestep shift fix
- `modules/modelSampler/Flux2Sampler.py`, `FluxSampler.py`, `QwenSampler.py`, `ZImageSampler.py` — sampler shift config
- `modules/modelSetup/Base*Setup.py` (12 files) — CEP validation guard
- `modules/modelSetup/mixin/ModelSetupDiffusionLossMixin.py` — MIN_SNR_GAMMA
- `modules/modelSetup/mixin/ModelSetupNoiseMixin.py` — CEP fix, deterministic timestep fraction
- `modules/trainer/GenericTrainer.py` — multi-level validation
- `modules/util/config/SampleConfig.py` — new fields
- `modules/util/enum/LossWeight.py` — new enum value

## Testing Notes

- Validation changes affect TensorBoard logging structure — existing dashboards may need tag updates
- MIN_SNR_GAMMA requires `loss_weight_strength` to be set (acts as the gamma clamp value)
- Sampler timestep shift changes are backward-compatible (defaults match previous behavior)
