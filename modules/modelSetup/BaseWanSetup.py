from abc import ABCMeta
from random import Random

import modules.util.multi_gpu_util as multi
from modules.model.WanModel import WanModel, WanModelEmbedding
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.mixin.ModelSetupDebugMixin import ModelSetupDebugMixin
from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import ModelSetupDiffusionLossMixin
from modules.modelSetup.mixin.ModelSetupEmbeddingMixin import ModelSetupEmbeddingMixin
from modules.modelSetup.mixin.ModelSetupFlowMatchingMixin import ModelSetupFlowMatchingMixin
from modules.modelSetup.mixin.ModelSetupNoiseMixin import ModelSetupNoiseMixin
from modules.modelSetup.mixin.ModelSetupText2ImageMixin import ModelSetupText2ImageMixin
from modules.module.AdditionalEmbeddingWrapper import AdditionalEmbeddingWrapper
from modules.util.checkpointing_util import (
    enable_checkpointing_for_t5_encoder_layers,
    enable_checkpointing_for_wan_transformer,
)
from modules.util.config.TrainConfig import TrainConfig
from modules.util.conv_util import apply_circular_padding_to_conv2d
from modules.util.dtype_util import create_autocast_context, disable_fp16_autocast_context
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.quantization_util import quantize_layers
from modules.util.torch_util import torch_gc
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor


class BaseWanSetup(
    BaseModelSetup,
    ModelSetupDiffusionLossMixin,
    ModelSetupDebugMixin,
    ModelSetupNoiseMixin,
    ModelSetupFlowMatchingMixin,
    ModelSetupEmbeddingMixin,
    ModelSetupText2ImageMixin,
    metaclass=ABCMeta
):
    LAYER_PRESETS = {
        "attn-mlp": ["attn", "ffn"],
        "attn-only": ["attn"],
        "blocks": ["blocks"],
        "full": [],
    }

    def setup_optimizations(
            self,
            model: WanModel,
            config: TrainConfig,
    ):
        if config.gradient_checkpointing.enabled():
            model.transformer_offload_conductor = \
                enable_checkpointing_for_wan_transformer(model.transformer, config)
            if model.transformer_2 is not None:
                model.transformer_2_offload_conductor = \
                    enable_checkpointing_for_wan_transformer(model.transformer_2, config)
            if model.text_encoder is not None:
                model.text_encoder_offload_conductor = \
                    enable_checkpointing_for_t5_encoder_layers(model.text_encoder, config)

        if config.force_circular_padding:
            apply_circular_padding_to_conv2d(model.vae)
            apply_circular_padding_to_conv2d(model.transformer)
            if model.transformer_2 is not None:
                apply_circular_padding_to_conv2d(model.transformer_2)
            if model.transformer_lora is not None:
                apply_circular_padding_to_conv2d(model.transformer_lora)
            if model.transformer_2_lora is not None:
                apply_circular_padding_to_conv2d(model.transformer_2_lora)

        model.autocast_context, model.train_dtype = create_autocast_context(self.train_device, config.train_dtype, [
            config.weight_dtypes().transformer,
            config.weight_dtypes().transformer_2,
            config.weight_dtypes().text_encoder,
            config.weight_dtypes().vae,
            config.weight_dtypes().lora if config.training_method == TrainingMethod.LORA else None,
            config.weight_dtypes().embedding if config.train_any_embedding() else None,
        ], config.enable_autocast_cache)

        model.text_encoder_autocast_context, model.text_encoder_train_dtype = \
            disable_fp16_autocast_context(
                self.train_device,
                config.train_dtype,
                config.fallback_train_dtype,
                [
                    config.weight_dtypes().text_encoder,
                    config.weight_dtypes().lora if config.training_method == TrainingMethod.LORA else None,
                    config.weight_dtypes().embedding if config.train_any_embedding() else None,
                ],
                config.enable_autocast_cache,
            )

        quantize_layers(model.text_encoder, self.train_device, model.text_encoder_train_dtype, config)
        quantize_layers(model.vae, self.train_device, model.train_dtype, config)
        quantize_layers(model.transformer, self.train_device, model.train_dtype, config)
        if model.transformer_2 is not None:
            quantize_layers(model.transformer_2, self.train_device, model.train_dtype, config)

        # Note: VAE tiling is not enabled here because the WAN VAE's feature caching
        # is not thread-safe with the concurrent DiskCache encoding pipeline.
        # For single-image encoding during training, tiling is unnecessary.

    def _setup_embeddings(
            self,
            model: WanModel,
            config: TrainConfig,
    ):
        additional_embeddings = []
        for embedding_config in config.all_embedding_configs():
            embedding_state = model.embedding_state_dicts.get(embedding_config.uuid, None)
            if embedding_state is None:
                with model.autocast_context:
                    embedding_state = self._create_new_embedding(
                        model,
                        embedding_config,
                        model.tokenizer,
                        model.text_encoder,
                        lambda text: model.encode_text(
                            text=text,
                            train_device=self.temp_device,
                        )[0][0][1:],
                    )
            else:
                embedding_state = embedding_state.get("t5_out", embedding_state.get("t5", None))

            if embedding_state is not None:
                embedding_state = embedding_state.to(
                    dtype=model.text_encoder.get_input_embeddings().weight.dtype,
                    device=self.train_device,
                ).detach()

            embedding = WanModelEmbedding(
                embedding_config.uuid,
                embedding_state,
                embedding_config.placeholder,
                embedding_config.is_output_embedding,
            )
            if embedding_config.uuid == config.embedding.uuid:
                model.embedding = embedding
            else:
                additional_embeddings.append(embedding)

        model.additional_embeddings = additional_embeddings

        if model.tokenizer is not None:
            self._add_embeddings_to_tokenizer(model.tokenizer, model.all_text_encoder_embeddings())

    def _setup_embedding_wrapper(
            self,
            model: WanModel,
            config: TrainConfig,
    ):
        if model.tokenizer is not None and model.text_encoder is not None:
            model.embedding_wrapper = AdditionalEmbeddingWrapper(
                tokenizer=model.tokenizer,
                orig_module=model.text_encoder.encoder.embed_tokens,
                embeddings=model.all_text_encoder_embeddings(),
            )

        if model.embedding_wrapper is not None:
            model.embedding_wrapper.hook_to_module()

    def _setup_embeddings_requires_grad(
            self,
            model: WanModel,
            config: TrainConfig,
    ):
        if model.text_encoder is not None:
            for embedding, embedding_config in zip(model.all_text_encoder_embeddings(),
                                                   config.all_embedding_configs(), strict=True):
                train_embedding = \
                    embedding_config.train \
                    and config.text_encoder.train_embedding \
                    and not self.stop_embedding_training_elapsed(embedding_config, model.train_progress)
                embedding.requires_grad_(train_embedding)

    def _prepare_i2v_conditioning(
            self,
            model: WanModel,
            scaled_latent_image: Tensor,
    ) -> Tensor:
        """
        Prepare I2V conditioning for WAN A14B (expand_timesteps=False).
        Takes the first frame latent and creates [mask, first_frame_latent] conditioning.

        Args:
            model: WanModel with VAE config
            scaled_latent_image: Already normalized latent (B, C, T, H, W) where C=16

        Returns:
            condition: (B, 20, T, H, W) = [4ch_mask, 16ch_first_frame_latent]
        """
        batch_size = scaled_latent_image.shape[0]
        num_channels = scaled_latent_image.shape[1]  # 16
        num_latent_frames = scaled_latent_image.shape[2]
        latent_height = scaled_latent_image.shape[3]
        latent_width = scaled_latent_image.shape[4]

        vae_scale_factor_temporal = model.vae.config.scale_factor_temporal  # typically 4

        # For image-only training, the first frame IS the target image.
        # Extract the first frame as the conditioning latent.
        first_frame_latent = scaled_latent_image[:, :, 0:1, :, :]  # (B, 16, 1, H, W)

        # Expand first frame latent to match temporal dimension with zeros for other frames
        latent_condition = torch.zeros_like(scaled_latent_image)
        latent_condition[:, :, 0:1, :, :] = first_frame_latent

        # Build the temporal mask (following WanImageToVideoPipeline.prepare_latents logic)
        # For expand_timesteps=False: mask is built from pixel-space frame count
        # Number of pixel frames = (num_latent_frames - 1) * vae_scale_factor_temporal + 1
        num_pixel_frames = (num_latent_frames - 1) * vae_scale_factor_temporal + 1

        # mask_lat_size starts as (B, 1, num_pixel_frames, H, W) with 1s everywhere
        mask_lat_size = torch.ones(
            batch_size, 1, num_pixel_frames, latent_height, latent_width,
            device=scaled_latent_image.device, dtype=scaled_latent_image.dtype,
        )
        # Zero out all frames except the first (first frame = known/conditioning)
        mask_lat_size[:, :, 1:] = 0

        # Replicate first frame mask by vae_scale_factor_temporal
        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=vae_scale_factor_temporal)
        mask_lat_size = torch.cat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)

        # Reshape from pixel frames to latent frames: (B, 1, pixel_frames, H, W) -> (B, vae_scale_factor_temporal, latent_frames, H, W)
        mask_lat_size = mask_lat_size.view(
            batch_size, -1, vae_scale_factor_temporal, latent_height, latent_width
        )
        mask_lat_size = mask_lat_size.transpose(1, 2)
        # mask_lat_size is now (B, vae_scale_factor_temporal, num_latent_frames, H, W) = (B, 4, T, H, W)

        # Concatenate mask and latent condition: (B, 4+16, T, H, W) = (B, 20, T, H, W)
        condition = torch.cat([mask_lat_size, latent_condition], dim=1)

        return condition

    def predict(
            self,
            model: WanModel,
            batch: dict,
            config: TrainConfig,
            train_progress: TrainProgress,
            *,
            deterministic: bool = False,
    ) -> dict:
        with model.autocast_context:
            batch_seed = 0 if deterministic else train_progress.global_step * multi.world_size() + multi.rank()
            generator = torch.Generator(device=config.train_device)
            generator.manual_seed(batch_seed)
            rand = Random(batch_seed)

            # VAE normalization parameters
            latents_mean = (
                torch.tensor(model.vae.config.latents_mean)
                .view(1, model.vae.config.z_dim, 1, 1, 1)
                .to(self.train_device, dtype=model.train_dtype.torch_dtype())
            )
            latents_std = (
                1.0 / torch.tensor(model.vae.config.latents_std)
                .view(1, model.vae.config.z_dim, 1, 1, 1)
                .to(self.train_device, dtype=model.train_dtype.torch_dtype())
            )

            text_encoder_output, text_attention_mask = model.encode_text(
                train_device=self.train_device,
                batch_size=batch['latent_image'].shape[0],
                rand=rand,
                tokens=batch.get("tokens"),
                tokens_mask=batch.get("tokens_mask"),
                text_encoder_layer_skip=config.text_encoder_layer_skip,
                text_encoder_output=batch['text_encoder_hidden_state'] \
                    if 'text_encoder_hidden_state' in batch and not config.train_text_encoder_or_embedding() else None,
                text_encoder_dropout_probability=config.text_encoder.dropout_probability,
            )

            if config.cep_enabled:
                text_encoder_output = self._apply_conditional_embedding_perturbation(
                    text_encoder_output, config.cep_gamma, generator
                )

            latent_image = batch['latent_image']

            # Ensure 5D: (B, C, T, H, W)
            if latent_image.ndim == 4:
                latent_image = latent_image.unsqueeze(2)

            # WAN VAE normalization: (latent - mean) * (1/std)
            scaled_latent_image = (latent_image - latents_mean) * latents_std

            # Prepare I2V conditioning (20ch = 4ch mask + 16ch first_frame_latent)
            condition = self._prepare_i2v_conditioning(model, scaled_latent_image)

            latent_noise = self._create_noise(scaled_latent_image, config, generator)

            num_train_timesteps = model.noise_scheduler.config['num_train_timesteps']

            shift = model.calculate_timestep_shift(
                scaled_latent_image.shape[-2], scaled_latent_image.shape[-1],
            )
            timestep = self._get_timestep_discrete(
                num_train_timesteps,
                deterministic,
                generator,
                scaled_latent_image.shape[0],
                config,
                shift=shift if config.dynamic_timestep_shifting else config.timestep_shift,
            )

            scaled_noisy_latent_image, sigma = self._add_noise_discrete(
                scaled_latent_image,
                latent_noise,
                timestep,
                model.noise_scheduler.timesteps,
            )

            # Concatenate noisy latent (16ch) with condition (20ch) -> 36ch input
            latent_input = torch.cat([scaled_noisy_latent_image, condition], dim=1)

            # Route to appropriate transformer based on timestep boundary
            # boundary_ratio=0.9 → boundary_timestep=900
            # t >= 900 → transformer (high-noise stage)
            # t < 900 → transformer_2 (low-noise stage)
            boundary_timestep = 0.9 * num_train_timesteps if model.transformer_2 is not None else None

            if boundary_timestep is not None:
                # Determine which samples go to which transformer
                high_noise_mask = timestep >= boundary_timestep  # -> transformer
                low_noise_mask = ~high_noise_mask  # -> transformer_2

                predicted_flow = torch.zeros_like(scaled_latent_image)

                if high_noise_mask.any():
                    high_indices = high_noise_mask.nonzero(as_tuple=True)[0]
                    high_input = latent_input[high_indices]
                    high_timestep = timestep[high_indices]
                    high_text = text_encoder_output[high_indices]

                    high_output = model.transformer(
                        hidden_states=high_input.to(dtype=model.train_dtype.torch_dtype()),
                        timestep=high_timestep,
                        encoder_hidden_states=high_text.to(dtype=model.train_dtype.torch_dtype()),
                        return_dict=False,
                    )[0]
                    predicted_flow[high_indices] = high_output.to(predicted_flow.dtype)

                if low_noise_mask.any():
                    low_indices = low_noise_mask.nonzero(as_tuple=True)[0]
                    low_input = latent_input[low_indices]
                    low_timestep = timestep[low_indices]
                    low_text = text_encoder_output[low_indices]

                    low_output = model.transformer_2(
                        hidden_states=low_input.to(dtype=model.train_dtype.torch_dtype()),
                        timestep=low_timestep,
                        encoder_hidden_states=low_text.to(dtype=model.train_dtype.torch_dtype()),
                        return_dict=False,
                    )[0]
                    predicted_flow[low_indices] = low_output.to(predicted_flow.dtype)
            else:
                # Single transformer mode
                predicted_flow = model.transformer(
                    hidden_states=latent_input.to(dtype=model.train_dtype.torch_dtype()),
                    timestep=timestep,
                    encoder_hidden_states=text_encoder_output.to(dtype=model.train_dtype.torch_dtype()),
                    return_dict=False,
                )[0]

            flow = latent_noise - scaled_latent_image
            model_output_data = {
                'loss_type': 'target',
                'timestep': timestep,
                'predicted': predicted_flow,
                'target': flow,
            }

            if config.debug_mode:
                with torch.no_grad():
                    predicted_scaled_latent_image = scaled_noisy_latent_image - predicted_flow * sigma
                    self._save_tokens("7-prompt", batch['tokens'], model.tokenizer, config, train_progress)
                    self._save_latent("1-noise", latent_noise, config, train_progress)
                    self._save_latent("2-noisy_image", scaled_noisy_latent_image, config, train_progress)
                    self._save_latent("3-predicted_flow", predicted_flow, config, train_progress)
                    self._save_latent("4-flow", flow, config, train_progress)
                    self._save_latent("5-predicted_image", predicted_scaled_latent_image, config, train_progress)
                    self._save_latent("6-image", scaled_latent_image, config, train_progress)

        return model_output_data

    def calculate_loss(
            self,
            model: WanModel,
            batch: dict,
            data: dict,
            config: TrainConfig,
    ) -> Tensor:
        return self._flow_matching_losses(
            batch=batch,
            data=data,
            config=config,
            train_device=self.train_device,
            sigmas=model.noise_scheduler.sigmas,
        ).mean()

    def prepare_text_caching(self, model: WanModel, config: TrainConfig):
        model.to(self.temp_device)

        if not config.train_text_encoder_or_embedding():
            model.text_encoder_to(self.train_device)

        model.eval()
        torch_gc()
