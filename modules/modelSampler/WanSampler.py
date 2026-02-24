import copy
import inspect
from collections.abc import Callable

from modules.model.WanModel import WanModel
from modules.modelSampler.BaseModelSampler import BaseModelSampler, ModelSamplerOutput
from modules.util import factory
from modules.util.config.SampleConfig import SampleConfig
from modules.util.enum.AudioFormat import AudioFormat
from modules.util.enum.FileType import FileType
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.NoiseScheduler import NoiseScheduler
from modules.util.enum.VideoFormat import VideoFormat
from modules.util.torch_util import torch_gc

import torch

from tqdm import tqdm


class WanSampler(BaseModelSampler):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            model: WanModel,
            model_type: ModelType,
    ):
        super().__init__(train_device, temp_device)

        self.model = model
        self.model_type = model_type
        self.pipeline = model.create_pipeline()

    @torch.no_grad()
    def __sample_base(
            self,
            prompt: str,
            negative_prompt: str,
            height: int,
            width: int,
            seed: int,
            random_seed: bool,
            diffusion_steps: int,
            cfg_scale: float,
            noise_scheduler: NoiseScheduler,
            text_encoder_layer_skip: int = 0,
            on_update_progress: Callable[[int, int], None] = lambda _, __: None,
    ) -> ModelSamplerOutput:
        with self.model.autocast_context:
            generator = torch.Generator(device=self.train_device)
            if random_seed:
                generator.seed()
            else:
                generator.manual_seed(seed)

            noise_scheduler = copy.deepcopy(self.model.noise_scheduler)
            transformer = self.pipeline.transformer
            transformer_2 = self.pipeline.transformer_2
            vae = self.pipeline.vae

            num_channels_latents = 16
            vae_scale_factor_spatial = vae.config.scale_factor_spatial  # 8
            vae_scale_factor_temporal = vae.config.scale_factor_temporal  # 4

            # For image-only sampling, use a single frame
            num_frames = 1
            num_latent_frames = 1

            latent_height = height // vae_scale_factor_spatial
            latent_width = width // vae_scale_factor_spatial

            # VAE normalization params
            latents_mean = (
                torch.tensor(vae.config.latents_mean)
                .view(1, vae.config.z_dim, 1, 1, 1)
                .to(self.train_device, dtype=self.model.train_dtype.torch_dtype())
            )
            latents_std = (
                1.0 / torch.tensor(vae.config.latents_std)
                .view(1, vae.config.z_dim, 1, 1, 1)
                .to(self.train_device, dtype=self.model.train_dtype.torch_dtype())
            )

            # Encode text
            self.model.text_encoder_to(self.train_device)

            text_encoder_output, text_attention_mask = self.model.encode_text(
                text=prompt,
                batch_size=1,
                train_device=self.train_device,
                text_encoder_layer_skip=text_encoder_layer_skip,
            )

            if cfg_scale > 1.0 and negative_prompt:
                negative_text_encoder_output, negative_text_attention_mask = self.model.encode_text(
                    text=negative_prompt,
                    batch_size=1,
                    train_device=self.train_device,
                    text_encoder_layer_skip=text_encoder_layer_skip,
                )
            else:
                negative_text_encoder_output = None

            self.model.text_encoder_to(self.temp_device)
            torch_gc()

            # Prepare latent noise (B, C, T, H, W)
            latent_image = torch.randn(
                size=(1, num_channels_latents, num_latent_frames, latent_height, latent_width),
                generator=generator,
                device=self.train_device,
                dtype=torch.float32,
            )

            # For I2V with a single image, we create a blank conditioning
            # (since there's no input image for sampling, use zeros)
            condition = torch.zeros(
                1, vae_scale_factor_temporal + num_channels_latents,
                num_latent_frames, latent_height, latent_width,
                device=self.train_device,
                dtype=self.model.train_dtype.torch_dtype(),
            )

            noise_scheduler.set_timesteps(diffusion_steps, device=self.train_device)
            timesteps = noise_scheduler.timesteps

            boundary_timestep = None
            if transformer_2 is not None:
                num_train_timesteps = noise_scheduler.config.get('num_train_timesteps', 1000)
                boundary_timestep = 0.9 * num_train_timesteps

            self.model.transformer_to(self.train_device)
            if transformer_2 is not None:
                self.model.transformer_2_to(self.train_device)

            for i, timestep in enumerate(tqdm(timesteps, desc="sampling")):
                # Route to transformer based on timestep
                if boundary_timestep is not None and timestep < boundary_timestep:
                    current_model = transformer_2
                else:
                    current_model = transformer

                # Concatenate latent with condition (36ch input)
                latent_model_input = torch.cat([latent_image, condition], dim=1)

                noise_pred = current_model(
                    hidden_states=latent_model_input.to(dtype=self.model.train_dtype.torch_dtype()),
                    timestep=timestep.expand(1),
                    encoder_hidden_states=text_encoder_output.to(dtype=self.model.train_dtype.torch_dtype()),
                    return_dict=False,
                )[0]

                if negative_text_encoder_output is not None and cfg_scale > 1.0:
                    noise_pred_uncond = current_model(
                        hidden_states=latent_model_input.to(dtype=self.model.train_dtype.torch_dtype()),
                        timestep=timestep.expand(1),
                        encoder_hidden_states=negative_text_encoder_output.to(dtype=self.model.train_dtype.torch_dtype()),
                        return_dict=False,
                    )[0]
                    noise_pred = noise_pred_uncond + cfg_scale * (noise_pred - noise_pred_uncond)

                latent_image = noise_scheduler.step(
                    noise_pred, timestep, latent_image, return_dict=False
                )[0]

                on_update_progress(i + 1, len(timesteps))

            self.model.transformer_to(self.temp_device)
            if transformer_2 is not None:
                self.model.transformer_2_to(self.temp_device)
            torch_gc()

            # Decode
            self.model.vae_to(self.train_device)

            # WAN VAE denormalization: latent / std + mean
            latent_image = latent_image.to(vae.dtype)
            latent_image = latent_image / latents_std.to(latent_image.dtype) + latents_mean.to(latent_image.dtype)

            image = vae.decode(latent_image, return_dict=False)[0]

            # Convert from video format (B, C, T, H, W) to image
            if image.ndim == 5:
                image = image[:, :, 0, :, :]  # Take first frame

            # Normalize to [0, 1]
            image = (image + 1.0) / 2.0
            image = image.clamp(0, 1)

            # Convert to PIL
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            from PIL import Image
            import numpy as np
            image = Image.fromarray((image[0] * 255).astype(np.uint8))

            self.model.vae_to(self.temp_device)
            torch_gc()

            return ModelSamplerOutput(
                file_type=FileType.IMAGE,
                data=image,
            )

    def sample(
            self,
            sample_config: SampleConfig,
            destination: str,
            image_format: ImageFormat | None = None,
            video_format: VideoFormat | None = None,
            audio_format: AudioFormat | None = None,
            on_sample: Callable[[ModelSamplerOutput], None] = lambda _: None,
            on_update_progress: Callable[[int, int], None] = lambda _, __: None,
    ):
        sampler_output = self.__sample_base(
            prompt=sample_config.prompt,
            negative_prompt=sample_config.negative_prompt,
            height=self.quantize_resolution(sample_config.height, 16),
            width=self.quantize_resolution(sample_config.width, 16),
            seed=sample_config.seed,
            random_seed=sample_config.random_seed,
            diffusion_steps=sample_config.diffusion_steps,
            cfg_scale=sample_config.cfg_scale,
            noise_scheduler=sample_config.noise_scheduler,
            text_encoder_layer_skip=sample_config.text_encoder_1_layer_skip,
            on_update_progress=on_update_progress,
        )

        self.save_sampler_output(
            sampler_output, destination,
            image_format, video_format, audio_format,
        )

        on_sample(sampler_output)

factory.register(BaseModelSampler, WanSampler, ModelType.WAN_I2V_A14B)
