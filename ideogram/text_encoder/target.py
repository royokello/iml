from __future__ import annotations

_NUM_LM_LAYERS = 36
_NUM_VISUAL_BLOCKS = 27

_LM_HIGH_SUFFIXES = (
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.down_proj.weight",
)

_LM_LOW_SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
)

_LM_FP32_SUFFIXES = (
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
)

_VISUAL_FP32_SUFFIXES = (
    "norm1.weight",
    "norm1.bias",
    "norm2.weight",
    "norm2.bias",
)

_VISUAL_FP16_SUFFIXES = (
    "attn.proj.weight",
    "attn.qkv.weight",
    "mlp.linear_fc1.weight",
    "mlp.linear_fc2.weight",
    "attn.proj.bias",
    "attn.qkv.bias",
    "mlp.linear_fc1.bias",
    "mlp.linear_fc2.bias",
)


def _build_ideogram_text_encoder_targets() -> dict[str, list[str]]:
    rank_1: list[str] = []
    rank_2: list[str] = []
    fp32: list[str] = []
    fp16: list[str] = []

    for layer_idx in range(_NUM_LM_LAYERS):
        prefix = f"language_model.layers.{layer_idx}."
        for suffix in _LM_HIGH_SUFFIXES:
            rank_1.append(prefix + suffix)
        for suffix in _LM_LOW_SUFFIXES:
            rank_2.append(prefix + suffix)
        for suffix in _LM_FP32_SUFFIXES:
            fp32.append(prefix + suffix)

    # Visual encoder — kept in fp16 (not used for text-only generation)
    for block_idx in range(_NUM_VISUAL_BLOCKS):
        prefix = f"visual.blocks.{block_idx}."
        for suffix in _VISUAL_FP32_SUFFIXES:
            fp32.append(prefix + suffix)
        for suffix in _VISUAL_FP16_SUFFIXES:
            fp16.append(prefix + suffix)

    fp32.append("language_model.norm.weight")

    # Visual global — all fp16
    fp16.append("visual.patch_embed.proj.weight")
    fp16.append("visual.patch_embed.proj.bias")
    fp16.append("visual.pos_embed.weight")
    fp16.append("visual.merger.linear_fc1.weight")
    fp16.append("visual.merger.linear_fc2.weight")
    fp16.append("visual.merger.linear_fc1.bias")
    fp16.append("visual.merger.linear_fc2.bias")
    for i in range(3):
        fp16.append(f"visual.deepstack_merger_list.{i}.linear_fc1.weight")
        fp16.append(f"visual.deepstack_merger_list.{i}.linear_fc2.weight")
        fp16.append(f"visual.deepstack_merger_list.{i}.linear_fc1.bias")
        fp16.append(f"visual.deepstack_merger_list.{i}.linear_fc2.bias")

    # Visual global norms — fp32
    fp32.append("visual.merger.norm.weight")
    fp32.append("visual.merger.norm.bias")
    for i in range(3):
        fp32.append(f"visual.deepstack_merger_list.{i}.norm.weight")
        fp32.append(f"visual.deepstack_merger_list.{i}.norm.bias")

    # LM embedding — fp16
    fp16.append("language_model.embed_tokens.weight")

    return {"rank_1": rank_1, "rank_2": rank_2, "fp32": fp32, "fp16": fp16}
