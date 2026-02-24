from modules.util.convert.lora.convert_lora_util import LoraConversionKeySet, map_prefix_range
from modules.util.convert.lora.convert_t5 import map_t5


def __map_wan_block(key_prefix: LoraConversionKeySet) -> list[LoraConversionKeySet]:
    keys = []

    # Self attention
    keys += [LoraConversionKeySet("self_attn.q", "attn1.to_q", parent=key_prefix)]
    keys += [LoraConversionKeySet("self_attn.k", "attn1.to_k", parent=key_prefix)]
    keys += [LoraConversionKeySet("self_attn.v", "attn1.to_v", parent=key_prefix)]
    keys += [LoraConversionKeySet("self_attn.o", "attn1.to_out.0", parent=key_prefix)]

    # Cross attention (k_img/v_img before k/v to avoid prefix collision in startswith matching)
    keys += [LoraConversionKeySet("cross_attn.q", "attn2.to_q", parent=key_prefix)]
    keys += [LoraConversionKeySet("cross_attn.k_img", "attn2.add_k_proj", parent=key_prefix)]
    keys += [LoraConversionKeySet("cross_attn.v_img", "attn2.add_v_proj", parent=key_prefix)]
    keys += [LoraConversionKeySet("cross_attn.k", "attn2.to_k", parent=key_prefix)]
    keys += [LoraConversionKeySet("cross_attn.v", "attn2.to_v", parent=key_prefix)]
    keys += [LoraConversionKeySet("cross_attn.o", "attn2.to_out.0", parent=key_prefix)]

    # FFN
    keys += [LoraConversionKeySet("ffn.0", "ffn.net.0.proj", parent=key_prefix)]
    keys += [LoraConversionKeySet("ffn.2", "ffn.net.2", parent=key_prefix)]

    return keys


def __map_wan_transformer(key_prefix: LoraConversionKeySet) -> list[LoraConversionKeySet]:
    keys = []

    # Time embedding (original WAN: time_embedding Sequential → diffusers: condition_embedder.time_embedder)
    keys += [LoraConversionKeySet("time_embedding.0", "condition_embedder.time_embedder.linear_1", parent=key_prefix)]
    keys += [LoraConversionKeySet("time_embedding.2", "condition_embedder.time_embedder.linear_2", parent=key_prefix)]

    # Text embedding (original WAN: text_embedding Sequential → diffusers: condition_embedder.text_embedder)
    keys += [LoraConversionKeySet("text_embedding.0", "condition_embedder.text_embedder.linear_1", parent=key_prefix)]
    keys += [LoraConversionKeySet("text_embedding.2", "condition_embedder.text_embedder.linear_2", parent=key_prefix)]

    # Image embedding (I2V only, original WAN: img_emb.proj Sequential → diffusers: condition_embedder.image_embedder.ff FeedForward)
    keys += [LoraConversionKeySet("img_emb.proj.1", "condition_embedder.image_embedder.ff.net.0.proj", parent=key_prefix)]
    keys += [LoraConversionKeySet("img_emb.proj.3", "condition_embedder.image_embedder.ff.net.2", parent=key_prefix)]

    # Output projection (original WAN: head.head → diffusers: proj_out)
    keys += [LoraConversionKeySet("head.head", "proj_out", parent=key_prefix)]

    # Transformer blocks
    for k in map_prefix_range("blocks", "blocks", parent=key_prefix):
        keys += __map_wan_block(k)

    return keys


def convert_wan_lora_key_sets() -> list[LoraConversionKeySet]:
    keys = []

    keys += [LoraConversionKeySet("bundle_emb", "bundle_emb")]
    keys += __map_wan_transformer(LoraConversionKeySet("diffusion_model", "lora_transformer"))
    keys += __map_wan_transformer(LoraConversionKeySet("diffusion_model_2", "lora_transformer_2"))
    keys += map_t5(LoraConversionKeySet("t5", "lora_te"))

    return keys
