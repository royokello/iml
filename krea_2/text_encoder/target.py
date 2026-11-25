from __future__ import annotations

_NUM_LM_LAYERS = 36

# Krea 2 text-to-image never reads the vision tower, so the quantized file keeps
# only the language model. The prefix is applied by the quantizer and the
# estimator. Drop it to restore the full Qwen3-VL checkpoint.
_KREA2_TEXT_ENCODER_INCLUSION_PREFIX = "language_model."

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


def _build_krea2_text_encoder_targets() -> dict[str, list[str]]:
    rank_1: list[str] = []
    rank_2: list[str] = []
    bf16: list[str] = []

    for layer_idx in range(_NUM_LM_LAYERS):
        prefix = f"language_model.layers.{layer_idx}."
        for suffix in _LM_HIGH_SUFFIXES:
            rank_1.append(prefix + suffix)
        for suffix in _LM_LOW_SUFFIXES:
            rank_2.append(prefix + suffix)
        for suffix in _LM_BF16_SUFFIXES:
            bf16.append(prefix + suffix)

    bf16.append("language_model.norm.weight")

    return {
        "rank_1": rank_1,
        "rank_2": rank_2,
        "bf16": bf16,
        "q8": ["language_model.embed_tokens.weight"],
    }
