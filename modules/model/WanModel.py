from contextlib import nullcontext
from random import Random

from modules.model.BaseModel import BaseModel, BaseModelEmbedding
from modules.model.util.t5_util import encode_t5
from modules.module.AdditionalEmbeddingWrapper import AdditionalEmbeddingWrapper
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import ModelType
from modules.util.LayerOffloadConductor import LayerOffloadConductor

import torch
from torch import Tensor

from diffusers import (
    AutoencoderKLWan,
    DiffusionPipeline,
    FlowMatchEulerDiscreteScheduler,
    WanImageToVideoPipeline,
    WanTransformer3DModel,
)
from transformers import AutoTokenizer, UMT5EncoderModel


class WanModelEmbedding:
    def __init__(
            self,
            uuid: str,
            text_encoder_vector: Tensor | None,
            placeholder: str,
            is_output_embedding: bool,
    ):
        self.text_encoder_embedding = BaseModelEmbedding(
            uuid=uuid,
            placeholder=placeholder,
            vector=text_encoder_vector,
            is_output_embedding=is_output_embedding,
        )


class WanModel(BaseModel):
    # WAN 2.2 official reference points for timestep shift:
    #   480p (latent ~60x107 = 6420 area) → sample_shift=3.0
    #   720p (latent ~90x160 = 14400 area) → sample_shift=5.0
    # Source: Wan-Video/Wan2.2 wan/configs/wan_i2v_A14B.py + generate.py
    _SHIFT_BASE_AREA = 6420
    _SHIFT_MAX_AREA = 14400
    _SHIFT_BASE = 3.0
    _SHIFT_MAX = 5.0

    # base model data
    tokenizer: AutoTokenizer | None
    noise_scheduler: FlowMatchEulerDiscreteScheduler | None
    text_encoder: UMT5EncoderModel | None
    vae: AutoencoderKLWan | None
    transformer: WanTransformer3DModel | None
    transformer_2: WanTransformer3DModel | None

    # autocast context
    text_encoder_autocast_context: torch.autocast | nullcontext

    text_encoder_train_dtype: DataType

    text_encoder_offload_conductor: LayerOffloadConductor | None
    transformer_offload_conductor: LayerOffloadConductor | None
    transformer_2_offload_conductor: LayerOffloadConductor | None

    # persistent embedding training data
    embedding: WanModelEmbedding | None
    additional_embeddings: list[WanModelEmbedding] | None
    embedding_wrapper: AdditionalEmbeddingWrapper | None

    # persistent lora training data
    text_encoder_lora: LoRAModuleWrapper | None
    transformer_lora: LoRAModuleWrapper | None
    transformer_2_lora: LoRAModuleWrapper | None
    lora_state_dict: dict | None

    def __init__(
            self,
            model_type: ModelType,
    ):
        super().__init__(
            model_type=model_type,
        )

        self.tokenizer = None
        self.noise_scheduler = None
        self.text_encoder = None
        self.vae = None
        self.transformer = None
        self.transformer_2 = None

        self.text_encoder_autocast_context = nullcontext()

        self.text_encoder_train_dtype = DataType.FLOAT_32

        self.text_encoder_offload_conductor = None
        self.transformer_offload_conductor = None
        self.transformer_2_offload_conductor = None

        self.embedding = None
        self.additional_embeddings = []
        self.embedding_wrapper = None

        self.text_encoder_lora = None
        self.transformer_lora = None
        self.transformer_2_lora = None
        self.lora_state_dict = None

    def adapters(self) -> list[LoRAModuleWrapper]:
        return [a for a in [
            self.text_encoder_lora,
            self.transformer_lora,
            self.transformer_2_lora,
        ] if a is not None]

    def calculate_timestep_shift(self, latent_height: int, latent_width: int) -> float:
        """
        Compute resolution-dependent timestep shift for WAN 2.2.

        Uses linear interpolation on latent spatial area, calibrated to the
        official WAN 2.2 I2V-A14B shift values (480p→3.0, 720p→5.0).
        Extrapolates linearly for resolutions beyond the reference range.
        """
        latent_area = latent_height * latent_width
        shift = self._SHIFT_BASE + (latent_area - self._SHIFT_BASE_AREA) * \
            (self._SHIFT_MAX - self._SHIFT_BASE) / (self._SHIFT_MAX_AREA - self._SHIFT_BASE_AREA)
        return max(shift, 1.0)

    def all_embeddings(self) -> list[WanModelEmbedding]:
        return self.additional_embeddings \
               + ([self.embedding] if self.embedding is not None else [])

    def all_text_encoder_embeddings(self) -> list[BaseModelEmbedding]:
        return [embedding.text_encoder_embedding for embedding in self.additional_embeddings] \
               + ([self.embedding.text_encoder_embedding] if self.embedding is not None else [])

    def vae_to(self, device: torch.device):
        self.vae.to(device=device)

    def text_encoder_to(self, device: torch.device):
        if self.text_encoder is not None:
            if self.text_encoder_offload_conductor is not None and \
                    self.text_encoder_offload_conductor.layer_offload_activated():
                self.text_encoder_offload_conductor.to(device)
            else:
                self.text_encoder.to(device=device)

        if self.text_encoder_lora is not None:
            self.text_encoder_lora.to(device)

    def transformer_to(self, device: torch.device):
        if self.transformer_offload_conductor is not None and \
                self.transformer_offload_conductor.layer_offload_activated():
            self.transformer_offload_conductor.to(device)
        else:
            self.transformer.to(device=device)

        if self.transformer_lora is not None:
            self.transformer_lora.to(device)

    def transformer_2_to(self, device: torch.device):
        if self.transformer_2 is None:
            return

        if self.transformer_2_offload_conductor is not None and \
                self.transformer_2_offload_conductor.layer_offload_activated():
            self.transformer_2_offload_conductor.to(device)
        else:
            self.transformer_2.to(device=device)

        if self.transformer_2_lora is not None:
            self.transformer_2_lora.to(device)

    def to(self, device: torch.device):
        self.vae_to(device)
        self.text_encoder_to(device)
        self.transformer_to(device)
        self.transformer_2_to(device)

    def eval(self):
        self.vae.eval()
        if self.text_encoder is not None:
            self.text_encoder.eval()
        self.transformer.eval()
        if self.transformer_2 is not None:
            self.transformer_2.eval()

    def create_pipeline(self) -> DiffusionPipeline:
        return WanImageToVideoPipeline(
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            vae=self.vae,
            scheduler=self.noise_scheduler,
            transformer=self.transformer,
            transformer_2=self.transformer_2,
            boundary_ratio=0.9 if self.transformer_2 is not None else None,
        )

    def add_text_encoder_embeddings_to_prompt(self, prompt: str) -> str:
        return self._add_embeddings_to_prompt(self.all_text_encoder_embeddings(), prompt)

    def encode_text(
            self,
            train_device: torch.device,
            batch_size: int = 1,
            rand: Random | None = None,
            text: str | list[str] = None,
            tokens: Tensor = None,
            tokens_mask: Tensor = None,
            text_encoder_layer_skip: int = 0,
            text_encoder_dropout_probability: float | None = None,
            text_encoder_output: Tensor = None,
    ) -> tuple[Tensor, Tensor]:
        # tokenize prompt
        if tokens is None and text is not None:
            if isinstance(text, list):
                text = [self.add_text_encoder_embeddings_to_prompt(t) for t in text]
            else:
                text = self.add_text_encoder_embeddings_to_prompt(text)
            tokenizer_output = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                return_tensors="pt",
            )
            tokens = tokenizer_output.input_ids.to(self.text_encoder.device)
            tokens_mask = tokenizer_output.attention_mask.to(self.text_encoder.device)

            seq_lengths = tokens_mask.sum(dim=1)
            mask_indices = torch.arange(tokens_mask.size(1), device=tokens_mask.device).unsqueeze(0).expand(batch_size, -1)
            bool_attention_mask = (mask_indices <= seq_lengths.unsqueeze(1))
        else:
            assert tokens_mask is not None
            bool_attention_mask = tokens_mask.bool()

        with self.text_encoder_autocast_context:
            text_encoder_output = encode_t5(
                text_encoder=self.text_encoder,
                tokens=tokens,
                default_layer=-1,
                layer_skip=text_encoder_layer_skip,
                text_encoder_output=text_encoder_output,
                use_attention_mask=True,
                attention_mask=bool_attention_mask.float(),
            )

        text_encoder_output = self._apply_output_embeddings(
            self.all_text_encoder_embeddings(),
            self.tokenizer,
            tokens,
            text_encoder_output,
        )

        # apply dropout
        if text_encoder_dropout_probability is not None:
            dropout_text_encoder_mask = (torch.tensor(
                [rand.random() > text_encoder_dropout_probability for _ in range(batch_size)],
                device=train_device)).float()
            text_encoder_output = text_encoder_output * dropout_text_encoder_mask[:, None, None]

        # prune tokens that are masked in all batch samples
        seq_lengths = bool_attention_mask.sum(dim=1)
        max_seq_length = seq_lengths.max().item()

        if max_seq_length % 16 > 0 and (seq_lengths != max_seq_length).any():
            max_seq_length += (16 - max_seq_length % 16)

        text_encoder_output = text_encoder_output[:, :max_seq_length, :]
        bool_attention_mask = bool_attention_mask[:, :max_seq_length]

        return (text_encoder_output, bool_attention_mask)
