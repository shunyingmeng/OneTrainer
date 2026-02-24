import copy
import os.path
from pathlib import Path

from modules.model.WanModel import WanModel
from modules.modelSaver.mixin.DtypeModelSaverMixin import DtypeModelSaverMixin
from modules.util.enum.ModelFormat import ModelFormat

import torch

from transformers import UMT5EncoderModel


class WanModelSaver(
    DtypeModelSaverMixin,
):
    def __init__(self):
        super().__init__()

    def __save_diffusers(
            self,
            model: WanModel,
            destination: str,
            dtype: torch.dtype | None,
    ):
        pipeline = model.create_pipeline()
        pipeline.to("cpu")
        if dtype is not None:
            tokenizer = pipeline.tokenizer
            tokenizer.__deepcopy__ = lambda memo: tokenizer

            save_pipeline = copy.deepcopy(pipeline)
            save_pipeline.to(device="cpu", dtype=dtype, silence_dtype_warnings=True)

            delattr(tokenizer, '__deepcopy__')
        else:
            save_pipeline = pipeline

        text_encoder = save_pipeline.text_encoder
        if text_encoder is not None:
            text_encoder_save_pretrained = text_encoder.save_pretrained
            def save_pretrained_t5(
                    self,
                    *args,
                    **kwargs,
            ):
                kwargs = dict(kwargs)
                kwargs['max_shard_size'] = '2GB'
                text_encoder_save_pretrained(*args, **kwargs)

            text_encoder.save_pretrained = save_pretrained_t5.__get__(text_encoder, UMT5EncoderModel)

        os.makedirs(Path(destination).absolute(), exist_ok=True)
        save_pipeline.save_pretrained(destination)

        if text_encoder is not None:
            text_encoder.save_pretrained = text_encoder_save_pretrained

        if dtype is not None:
            del save_pipeline

    def __save_internal(
            self,
            model: WanModel,
            destination: str,
    ):
        self.__save_diffusers(model, destination, None)

    def save(
            self,
            model: WanModel,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype | None,
    ):
        match output_model_format:
            case ModelFormat.DIFFUSERS:
                self.__save_diffusers(model, output_model_destination, dtype)
            case ModelFormat.SAFETENSORS:
                raise NotImplementedError("Single-file safetensors saving not supported for WAN models. Use DIFFUSERS format.")
            case ModelFormat.INTERNAL:
                self.__save_internal(model, output_model_destination)
