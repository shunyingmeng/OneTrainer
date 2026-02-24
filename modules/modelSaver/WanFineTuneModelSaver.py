from modules.model.WanModel import WanModel
from modules.modelSaver.wan.WanEmbeddingSaver import WanEmbeddingSaver
from modules.modelSaver.wan.WanModelSaver import WanModelSaver
from modules.modelSaver.GenericFineTuneModelSaver import make_fine_tune_model_saver
from modules.util.enum.ModelType import ModelType

WanFineTuneModelSaver = make_fine_tune_model_saver(
    ModelType.WAN_I2V_A14B,
    model_class=WanModel,
    model_saver_class=WanModelSaver,
    embedding_saver_class=WanEmbeddingSaver,
)
