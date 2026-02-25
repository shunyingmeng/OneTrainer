import os
import traceback

from modules.model.WanModel import WanModel
from modules.modelLoader.mixin.HFModelLoaderMixin import HFModelLoaderMixin
from modules.util.config.TrainConfig import QuantizationConfig
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes

import torch

from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    WanTransformer3DModel,
)
from transformers import AutoTokenizer, UMT5EncoderModel


class WanModelLoader(
    HFModelLoaderMixin,
):
    def __init__(self):
        super().__init__()

    def __load_internal(
            self,
            model: WanModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            vae_model_name: str,
            include_text_encoder: bool,
            quantization: QuantizationConfig,
    ):
        if os.path.isfile(os.path.join(base_model_name, "meta.json")):
            self.__load_diffusers(
                model, model_type, weight_dtypes, base_model_name, vae_model_name, include_text_encoder, quantization,
            )
        else:
            raise Exception("not an internal model")

    def __load_diffusers(
            self,
            model: WanModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            vae_model_name: str,
            include_text_encoder: bool,
            quantization: QuantizationConfig,
    ):
        diffusers_sub = ["transformer"]
        if not vae_model_name:
            diffusers_sub.append("vae")

        # Check if transformer_2 exists (dual-transformer architecture for I2V A14B)
        is_local = os.path.isdir(base_model_name)
        if is_local:
            has_transformer_2 = os.path.isdir(os.path.join(base_model_name, "transformer_2"))
        else:
            # For HuggingFace models, check model_index.json for transformer_2
            has_transformer_2 = model_type.is_wan_i2v()
        if has_transformer_2:
            diffusers_sub.append("transformer_2")

        transformers_sub = []
        if include_text_encoder:
            transformers_sub.append("text_encoder")

        self._prepare_sub_modules(
            base_model_name,
            diffusers_modules=diffusers_sub,
            transformers_modules=transformers_sub,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            subfolder="tokenizer",
        )

        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            base_model_name,
            subfolder="scheduler",
        )

        if include_text_encoder:
            text_encoder = self._load_transformers_sub_module(
                UMT5EncoderModel,
                weight_dtypes.text_encoder,
                weight_dtypes.fallback_train_dtype,
                base_model_name,
                "text_encoder",
            )
        else:
            text_encoder = None

        if vae_model_name:
            vae = self._load_diffusers_sub_module(
                AutoencoderKLWan,
                weight_dtypes.vae,
                weight_dtypes.train_dtype,
                vae_model_name,
            )
        else:
            vae = self._load_diffusers_sub_module(
                AutoencoderKLWan,
                weight_dtypes.vae,
                weight_dtypes.train_dtype,
                base_model_name,
                "vae",
            )

        transformer = self._load_diffusers_sub_module(
            WanTransformer3DModel,
            weight_dtypes.transformer,
            weight_dtypes.train_dtype,
            base_model_name,
            "transformer",
            quantization,
        )

        transformer_2 = None
        if has_transformer_2:
            transformer_2 = self._load_diffusers_sub_module(
                WanTransformer3DModel,
                weight_dtypes.transformer_2,
                weight_dtypes.train_dtype,
                base_model_name,
                "transformer_2",
                quantization,
            )

        model.model_type = model_type
        model.tokenizer = tokenizer
        model.noise_scheduler = noise_scheduler
        model.text_encoder = text_encoder
        model.vae = vae
        model.transformer = transformer
        model.transformer_2 = transformer_2

    def load(
            self,
            model: WanModel,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
            quantization: QuantizationConfig,
    ):
        stacktraces = []

        try:
            self.__load_internal(
                model, model_type, weight_dtypes, model_names.base_model, model_names.vae_model,
                model_names.include_text_encoder, quantization,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_diffusers(
                model, model_type, weight_dtypes, model_names.base_model, model_names.vae_model,
                model_names.include_text_encoder, quantization,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        for stacktrace in stacktraces:
            print(stacktrace)
        raise Exception("could not load model: " + model_names.base_model)
