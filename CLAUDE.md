# Branch: dev/wan22-i2v-support

WAN 2.2 I2V (image-to-video) A14B model support for OneTrainer. Based on `master`.

## Status

### Completed: LoRA Training
- `WAN_I2V_A14B` model type enum with `is_wan()`, `is_wan_i2v()`, `is_flow_matching()` helpers
- `transformer_2` config field in `TrainConfig`, `ModelWeightDtypes`, `ModelNames` (config migration v10 > v11)
- `WanModel` class with dual transformer, single T5 text encoder, `AutoencoderKLWan` VAE
- Model loaders: `WanModelLoader`, `WanLoRALoader`, `WanEmbeddingLoader`, `WanLoRAModelLoader`
- Model setup: `BaseWanSetup` (predict/loss with dual-transformer routing, I2V conditioning, WAN VAE normalization), `WanLoRASetup` (independent LoRA for transformer + transformer_2)
- `WanBaseDataLoader` for image datasets
- `WanSampler` with dual-transformer inference and I2V sampling
- Model savers: `WanLoRASaver`, `WanEmbeddingSaver`, `WanModelSaver`, `WanLoRAModelSaver`
- `enable_checkpointing_for_wan_transformer()` in `checkpointing_util.py`
- Model spec files: `wan_i2v.json`, `wan_i2v-lora.json`
- Training preset: `#wan i2v LoRA 96GB.json` (Prodigy ADV, bf16, CPU-offloaded grad checkpointing)

### Completed: LoRA Export
- ComfyUI-compatible LoRA key conversion in `convert_wan_lora.py`
  - Maps between diffusers names (`attn1.to_q`, `attn2.add_k_proj`, `ffn.net.0.proj`) and original WAN names (`self_attn.q`, `cross_attn.k_img`, `ffn.0`)
  - omi prefix: `diffusion_model` / `diffusion_model_2`, diffusers prefix: `lora_transformer` / `lora_transformer_2`
  - Includes T5 text encoder mapping via `map_t5()`
- `WanLoRASaver` saves in omi format (`enable_omi_format=True`) for direct ComfyUI loading

## TODO

### Fine-Tune Training
- [ ] Implement `WanFineTuneSetup.setup_model()` and `predict()` (currently raise `NotImplementedError`)
- [ ] Full fine-tune requires: unfreezing base transformer weights, full optimizer state, saving full checkpoints

### Video Dataset Support
- [ ] Video file loading (mp4, etc.): temporal frame sampling, multi-frame VAE encoding, variable-length sequences
- [ ] Currently image-only (single frame unsqueezed to 1-frame video)

### Other WAN Variants (deferred)
- WAN 2.2 T2V — different pipeline, no I2V conditioning
- WAN 2.2 5B — single transformer, smaller architecture
- WAN 2.1 — older architecture version

### Advanced Features (deferred)
- Quantization support (NF4/INT8) for smaller GPUs
- Embedding training (TI) — `WanEmbeddingSaver` exists but setup is minimal
- kohya/musubi-tuner legacy format import

## Files Added/Modified

New files (all under `modules/`):
- `model/WanModel.py` — dual transformer model class
- `modelLoader/wan/` — `WanModelLoader`, `WanLoRALoader`, `WanEmbeddingLoader`
- `modelLoader/WanFineTuneModelLoader.py`, `WanLoRAModelLoader.py` — factory stubs
- `modelSampler/WanSampler.py` — dual-transformer inference
- `modelSaver/wan/` — `WanLoRASaver`, `WanEmbeddingSaver`, `WanModelSaver`
- `modelSaver/WanFineTuneModelSaver.py`, `WanLoRAModelSaver.py` — factory stubs
- `modelSetup/BaseWanSetup.py`, `WanLoRASetup.py`, `WanFineTuneSetup.py`
- `dataLoader/WanBaseDataLoader.py`
- `util/convert/lora/convert_wan_lora.py` — ComfyUI LoRA key conversion
- `util/checkpointing_util.py` — `enable_checkpointing_for_wan_transformer()`
- `util/enum/ModelType.py` — `WAN_I2V_A14B` enum
- `util/config/TrainConfig.py` — `transformer_2` field, config v11 migration
- `util/ModelNames.py`, `util/ModelWeightDtypes.py` — `transformer_2` support
- `resources/sd_model_spec/wan_i2v.json`, `wan_i2v-lora.json`
- `training_presets/#wan i2v LoRA 96GB.json`
- `training_configs/wan.json` — test fixture

## Architecture Notes

- **Dual Transformer MoE**: `boundary_ratio=0.9` — timestep >= 900 uses `transformer` (high-noise), < 900 uses `transformer_2` (low-noise)
- **I2V Conditioning**: `concat([4ch_temporal_mask, 16ch_first_frame_latent])` > 20ch condition, `concat([16ch_noisy_latent, 20ch_condition])` > 36ch model input
- **VAE Normalization**: `scaled = (latent - latents_mean) * (1/latents_std)` using per-channel arrays from VAE config
- **Pattern References**: Follows `Chroma` pattern (single T5 text encoder) + `HunyuanVideo` pattern (video VAE, flow matching)
