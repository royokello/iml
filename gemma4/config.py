from __future__ import annotations

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
_ALL_LINEAR_WEIGHT_SUFFIXES = _HIGH_LINEAR_WEIGHT_SUFFIXES + _LOW_LINEAR_WEIGHT_SUFFIXES

_GEMMA4_QUANT_CONFIGS = {
    "high": {
        "token_embed": "sym-med",
        "per_layer_token_embed": "sym-high",
        "linears": {
            "sym-med": _ALL_LINEAR_WEIGHT_SUFFIXES,
        },
    },
    "aff-med-max": {
        "token_embed": "aff-med",
        "per_layer_token_embed": "sym-med",
        "linears": {
            "aff-med": _LOW_LINEAR_WEIGHT_SUFFIXES,
            "sym-med": _HIGH_LINEAR_WEIGHT_SUFFIXES,
        },
    },
    "aff-med": {
        "token_embed": "aff-med",
        "per_layer_token_embed": "aff-med",
        "linears": {
            "aff-med": _ALL_LINEAR_WEIGHT_SUFFIXES
        },
    },
    "aff-med-mini": {
        "token_embed": "sym-low",
        "per_layer_token_embed": "aff-med",
        "linears": {
            "aff-med": _HIGH_LINEAR_WEIGHT_SUFFIXES,
            "sym-low": _LOW_LINEAR_WEIGHT_SUFFIXES,
        },
    },
    "aff-high-mini": {
        "token_embed": "aff-med",
        "per_layer_token_embed": "aff-high",
        "linears": {
            "aff-high": _HIGH_LINEAR_WEIGHT_SUFFIXES,
            "aff-med": _LOW_LINEAR_WEIGHT_SUFFIXES,
        },
    },
}
_GEMMA4_QUANT_METHODS = tuple(_GEMMA4_QUANT_CONFIGS)
