from __future__ import annotations

from utils.quant.name import convert_quant_name, mixed_quant_methods

_NUM_LANGUAGE_LAYERS = 35
_NUM_LANGUAGE_KV_PROJECTION_LAYERS = 15

_LANGUAGE_TOKEN_EMBED_WEIGHT = "model.language_model.embed_tokens.weight"
_LANGUAGE_PER_LAYER_TOKEN_EMBED_WEIGHT = "model.language_model.embed_tokens_per_layer.weight"
_KV_PROJECTION_WEIGHT_SUFFIXES = {
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
}

_HIGH_LINEAR_WEIGHT_SUFFIXES = (
    "self_attn.o_proj.weight",
    "self_attn.v_proj.weight",
    "mlp.down_proj.weight",
)
_LOW_LINEAR_WEIGHT_SUFFIXES = (
    "self_attn.k_proj.weight",
    "self_attn.q_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
)

def _build_gemma4_quant_config(method: str) -> dict[str, str | dict[str, str]]:
    high_method, low_method = convert_quant_name(method)

    return {
        "token_embed": low_method,
        "per_layer_token_embed": high_method,
        "linears": {
            "high": high_method,
            "low": low_method,
        },
    }


_GEMMA4_QUANT_METHODS = mixed_quant_methods()
_GEMMA4_QUANT_CONFIGS = {
    method: _build_gemma4_quant_config(method)
    for method in _GEMMA4_QUANT_METHODS
}
