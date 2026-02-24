from modules.model.WanModel import WanModel
from modules.modelSetup.BaseWanSetup import BaseWanSetup
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.NamedParameterGroup import NamedParameterGroupCollection
from modules.util.optimizer_util import init_model_parameters
from modules.util.torch_util import state_dict_has_prefix
from modules.util.TrainProgress import TrainProgress

import torch


class WanLoRASetup(
    BaseWanSetup,
):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            debug_mode: bool,
    ):
        super().__init__(
            train_device=train_device,
            temp_device=temp_device,
            debug_mode=debug_mode,
        )

    def create_parameters(
            self,
            model: WanModel,
            config: TrainConfig,
    ) -> NamedParameterGroupCollection:
        parameter_group_collection = NamedParameterGroupCollection()

        self._create_model_part_parameters(parameter_group_collection, "text_encoder_lora", model.text_encoder_lora, config.text_encoder)

        if config.train_any_embedding() or config.train_any_output_embedding():
            if config.text_encoder.train_embedding and model.text_encoder is not None:
                self._add_embedding_param_groups(
                    model.all_text_encoder_embeddings(), parameter_group_collection, config.embedding_learning_rate,
                    "embeddings"
                )

        self._create_model_part_parameters(parameter_group_collection, "transformer_lora", model.transformer_lora, config.transformer)

        if model.transformer_2_lora is not None:
            self._create_model_part_parameters(parameter_group_collection, "transformer_2_lora", model.transformer_2_lora, config.transformer_2)

        return parameter_group_collection

    def __setup_requires_grad(
            self,
            model: WanModel,
            config: TrainConfig,
    ):
        self._setup_embeddings_requires_grad(model, config)
        if model.text_encoder is not None:
            model.text_encoder.requires_grad_(False)
        model.transformer.requires_grad_(False)
        if model.transformer_2 is not None:
            model.transformer_2.requires_grad_(False)
        model.vae.requires_grad_(False)

        self._setup_model_part_requires_grad("text_encoder_lora", model.text_encoder_lora, config.text_encoder, model.train_progress)
        self._setup_model_part_requires_grad("transformer_lora", model.transformer_lora, config.transformer, model.train_progress)
        if model.transformer_2_lora is not None:
            self._setup_model_part_requires_grad("transformer_2_lora", model.transformer_2_lora, config.transformer_2, model.train_progress)

    def setup_model(
            self,
            model: WanModel,
            config: TrainConfig,
    ):
        create_te = config.text_encoder.train or state_dict_has_prefix(model.lora_state_dict, "lora_te")

        if model.text_encoder is not None:
            model.text_encoder_lora = LoRAModuleWrapper(
                model.text_encoder, "lora_te", config
            ) if create_te else None

        model.transformer_lora = LoRAModuleWrapper(
            model.transformer, "lora_transformer", config, config.layer_filter.split(",")
        )

        if model.transformer_2 is not None and (config.transformer_2.train or state_dict_has_prefix(model.lora_state_dict, "lora_transformer_2")):
            model.transformer_2_lora = LoRAModuleWrapper(
                model.transformer_2, "lora_transformer_2", config, config.layer_filter.split(",")
            )

        if model.lora_state_dict:
            if model.text_encoder_lora is not None:
                model.text_encoder_lora.load_state_dict(model.lora_state_dict)
            model.transformer_lora.load_state_dict(model.lora_state_dict)
            if model.transformer_2_lora is not None:
                model.transformer_2_lora.load_state_dict(model.lora_state_dict)
            model.lora_state_dict = None

        if model.text_encoder_lora is not None:
            model.text_encoder_lora.set_dropout(config.dropout_probability)
            model.text_encoder_lora.to(dtype=config.lora_weight_dtype.torch_dtype())
            model.text_encoder_lora.hook_to_module()

        model.transformer_lora.set_dropout(config.dropout_probability)
        model.transformer_lora.to(dtype=config.lora_weight_dtype.torch_dtype())
        model.transformer_lora.hook_to_module()

        if model.transformer_2_lora is not None:
            model.transformer_2_lora.set_dropout(config.dropout_probability)
            model.transformer_2_lora.to(dtype=config.lora_weight_dtype.torch_dtype())
            model.transformer_2_lora.hook_to_module()

        if config.train_any_embedding():
            if model.text_encoder is not None:
                model.text_encoder.get_input_embeddings().to(dtype=config.embedding_weight_dtype.torch_dtype())

        self._remove_added_embeddings_from_tokenizer(model.tokenizer)
        self._setup_embeddings(model, config)
        self._setup_embedding_wrapper(model, config)

        params = self.create_parameters(model, config)
        self.__setup_requires_grad(model, config)
        init_model_parameters(model, params, self.train_device)

    def setup_train_device(
            self,
            model: WanModel,
            config: TrainConfig,
    ):
        vae_on_train_device = not config.latent_caching
        text_encoder_on_train_device = \
            config.train_text_encoder_or_embedding() \
            or not config.latent_caching

        model.text_encoder_to(self.train_device if text_encoder_on_train_device else self.temp_device)
        model.vae_to(self.train_device if vae_on_train_device else self.temp_device)
        model.transformer_to(self.train_device)
        model.transformer_2_to(self.train_device)

        if model.text_encoder:
            if config.text_encoder.train:
                model.text_encoder.train()
            else:
                model.text_encoder.eval()

        model.vae.eval()

        if config.transformer.train:
            model.transformer.train()
        else:
            model.transformer.eval()

        if model.transformer_2 is not None:
            if config.transformer_2.train:
                model.transformer_2.train()
            else:
                model.transformer_2.eval()

    def after_optimizer_step(
            self,
            model: WanModel,
            config: TrainConfig,
            train_progress: TrainProgress
    ):
        if config.preserve_embedding_norm:
            self._normalize_output_embeddings(model.all_text_encoder_embeddings())
            if model.embedding_wrapper is not None:
                model.embedding_wrapper.normalize_embeddings()
        self.__setup_requires_grad(model, config)

factory.register(BaseModelSetup, WanLoRASetup, ModelType.WAN_I2V_A14B, TrainingMethod.LORA)
