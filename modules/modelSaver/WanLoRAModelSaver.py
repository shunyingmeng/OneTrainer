from modules.model.WanModel import WanModel
from modules.modelSaver.wan.WanEmbeddingSaver import WanEmbeddingSaver
from modules.modelSaver.wan.WanLoRASaver import WanLoRASaver
from modules.modelSaver.GenericLoRAModelSaver import make_lora_model_saver
from modules.util.enum.ModelType import ModelType

WanLoRAModelSaver = make_lora_model_saver(
    ModelType.WAN_I2V_A14B,
    model_class=WanModel,
    lora_saver_class=WanLoRASaver,
    embedding_saver_class=WanEmbeddingSaver,
)
