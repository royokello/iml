from __future__ import annotations

_NUM_HIDDEN_LAYERS = 28

_HIGH_SUFFIXES = (
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.down_proj.weight",
)

_LOW_SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
)

_FP32_SUFFIXES = (
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
)

_GLOBAL_FP32 = ("model.norm.weight",)

_GLOBAL_FP16 = ("model.embed_tokens.weight",)


def _build_anima_text_encoder_targets() -> dict[str, list[str]]:
    high: list[str] = []
    low: list[str] = []
    fp32: list[str] = []
    fp16: list[str] = []

    for i in range(_NUM_HIDDEN_LAYERS):
        prefix = f"model.layers.{i}."
        for s in _HIGH_SUFFIXES:
            high.append(prefix + s)
        for s in _LOW_SUFFIXES:
            low.append(prefix + s)
        for s in _FP32_SUFFIXES:
            fp32.append(prefix + s)

    fp32.extend(_GLOBAL_FP32)
    fp16.extend(_GLOBAL_FP16)

    return {"high": high, "low": low, "fp32": fp32, "fp16": fp16}
