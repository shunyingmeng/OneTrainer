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
from modules.util.image_util import load_image
from modules.util.torch_util import torch_gc

import torch
import numpy as np
from PIL import Image


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
            base_image_path: str = "",
            text_encoder_layer_skip: int = 0,
            on_update_progress: Callable[[int, int], None] = lambda _, __: None,
    ) -> ModelSamplerOutput:
        generator = torch.Generator(device=self.train_device)
        if random_seed:
            generator.seed()
        else:
            generator.manual_seed(seed)

        pipe = self.pipeline

        # Load reference image for I2V conditioning
        if base_image_path:
            ref_image = load_image(base_image_path, convert_mode="RGB")
        else:
            print("WARNING: No base_image_path set for WAN I2V sampling. "
                  "Set base_image_path in sample config for proper results.")
            # Create a blank image as fallback
            ref_image = Image.new("RGB", (width, height), (128, 128, 128))

        # Apply correct flow_shift to the scheduler
        flow_shift = self.model.noise_scheduler.config.get('flow_shift', 3.0)
        if hasattr(pipe.scheduler, 'set_shift'):
            pipe.scheduler.set_shift(flow_shift)

        # Move all models to train device for sampling
        self.model.text_encoder_to(self.train_device)
        self.model.transformer_to(self.train_device)
        if self.model.transformer_2 is not None:
            self.model.transformer_2_to(self.train_device)
        self.model.vae_to(self.train_device)

        # Use the official pipeline for sampling
        result = pipe(
            image=ref_image,
            prompt=prompt,
            negative_prompt=negative_prompt if cfg_scale > 1.0 else None,
            height=height,
            width=width,
            num_frames=1,
            num_inference_steps=diffusion_steps,
            guidance_scale=cfg_scale,
            generator=generator,
            output_type="np",
        )

        # Extract the single frame from the video output
        frames = result.frames
        if isinstance(frames, list) and len(frames) > 0:
            if isinstance(frames[0], list):
                frame = frames[0][0]
            else:
                frame = frames[0]
        else:
            frame = frames

        # Convert numpy array to PIL Image
        if isinstance(frame, np.ndarray):
            # Handle video dimensions: (1, H, W, 3) or (H, W, 3)
            while frame.ndim > 3:
                frame = frame[0]
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
            image = Image.fromarray(frame)
        elif isinstance(frame, Image.Image):
            image = frame
        else:
            raise ValueError(f"Unexpected frame type: {type(frame)}")

        # Move models back to temp device
        self.model.text_encoder_to(self.temp_device)
        self.model.transformer_to(self.temp_device)
        if self.model.transformer_2 is not None:
            self.model.transformer_2_to(self.temp_device)
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
            base_image_path=sample_config.base_image_path,
            text_encoder_layer_skip=sample_config.text_encoder_1_layer_skip,
            on_update_progress=on_update_progress,
        )

        self.save_sampler_output(
            sampler_output, destination,
            image_format, video_format, audio_format,
        )

        on_sample(sampler_output)

factory.register(BaseModelSampler, WanSampler, ModelType.WAN_I2V_A14B)
