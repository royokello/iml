from __future__ import annotations

# The Qwen3-VL checkpoint stores the language model under `model.language_model`
# and the vision tower under `model.visual`. Both are kept: text-to-image loads
# the language model only, editing loads the full model. The vision tower stays
# BF16 so editing works without a second checkpoint.
TEXT_ENCODER_INCLUSION_PREFIX = ("model.language_model.", "model.visual.")
TEXT_ENCODER_EXCLUSION_PREFIX = ("lm_head.",)

_NUM_LM_LAYERS = 36
_NUM_VISION_BLOCKS = 27

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

_LM_BF16_SUFFIXES = (
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
)

_VISION_BLOCK_SUFFIXES = (
    "attn.qkv.weight",
    "attn.qkv.bias",
    "attn.proj.weight",
    "attn.proj.bias",
    "mlp.linear_fc1.weight",
    "mlp.linear_fc1.bias",
    "mlp.linear_fc2.weight",
    "mlp.linear_fc2.bias",
    "norm1.weight",
    "norm1.bias",
    "norm2.weight",
    "norm2.bias",
)

_VISION_GLOBAL_SUFFIXES = (
    "patch_embed.proj.weight",
    "patch_embed.proj.bias",
    "pos_embed.weight",
    "merger.norm.weight",
    "merger.norm.bias",
    "merger.linear_fc1.weight",
    "merger.linear_fc1.bias",
    "merger.linear_fc2.weight",
    "merger.linear_fc2.bias",
)

_DEEPSTACK_SUFFIXES = (
    "norm.weight",
    "norm.bias",
    "linear_fc1.weight",
    "linear_fc1.bias",
    "linear_fc2.weight",
    "linear_fc2.bias",
)


def _build_qwen2_text_encoder_targets() -> dict[str, list[str]]:
    """Build the quantized and BF16 tensor groups for the text encoder.

    Returns:
        A mapping of group name to full checkpoint tensor names.
    """
    high: list[str] = []
    low: list[str] = []
    bf16: list[str] = []

    for layer_idx in range(_NUM_LM_LAYERS):
        prefix = f"model.language_model.layers.{layer_idx}."
        for suffix in _LM_HIGH_SUFFIXES:
            high.append(prefix + suffix)
        for suffix in _LM_LOW_SUFFIXES:
            low.append(prefix + suffix)
        for suffix in _LM_BF16_SUFFIXES:
            bf16.append(prefix + suffix)

    bf16.append("model.language_model.norm.weight")

    for block_idx in range(_NUM_VISION_BLOCKS):
        prefix = f"model.visual.blocks.{block_idx}."
        for suffix in _VISION_BLOCK_SUFFIXES:
            bf16.append(prefix + suffix)

    for suffix in _VISION_GLOBAL_SUFFIXES:
        bf16.append(f"model.visual.{suffix}")

    for index in range(3):
        prefix = f"model.visual.deepstack_merger_list.{index}."
        for suffix in _DEEPSTACK_SUFFIXES:
            bf16.append(prefix + suffix)

    return {
        "high": high,
        "low": low,
        "bf16": bf16,
        "sym-high": ["model.language_model.embed_tokens.weight"],
    }


__all__ = [
    "TEXT_ENCODER_EXCLUSION_PREFIX",
    "TEXT_ENCODER_INCLUSION_PREFIX",
    "_build_qwen2_text_encoder_targets",
]
