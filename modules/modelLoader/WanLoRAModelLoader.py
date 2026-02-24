from modules.model.WanModel import WanModel
from modules.modelLoader.wan.WanEmbeddingLoader import WanEmbeddingLoader
from modules.modelLoader.wan.WanLoRALoader import WanLoRALoader
from modules.modelLoader.wan.WanModelLoader import WanModelLoader
from modules.modelLoader.GenericLoRAModelLoader import make_lora_model_loader
from modules.util.enum.ModelType import ModelType

WanLoRAModelLoader = make_lora_model_loader(
    model_spec_map={ModelType.WAN_I2V_A14B: "resources/sd_model_spec/wan_i2v-lora.json"},
    model_class=WanModel,
    model_loader_class=WanModelLoader,
    embedding_loader_class=WanEmbeddingLoader,
    lora_loader_class=WanLoRALoader,
)
