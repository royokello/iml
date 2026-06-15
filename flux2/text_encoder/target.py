from __future__ import annotations

_NUM_HIDDEN_LAYERS = 36

# Suffixes that receive higher bit‑width quantisation
_TEXT_ENCODER_HIGH_SUFFIXES = (
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.down_proj.weight",
)

# Suffixes that receive lower bit‑width quantisation
_TEXT_ENCODER_LOW_SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
)


def _build_target_tensors() -> dict[str, list[str]]:
    """
    Return {'high': [...], 'low': [...]} weight names for mixed‑precision
    quantisation of the Qwen3 text encoder.
    Names are prefixed with 'model.' so the loader's _strip_model_prefix can remove it.
    """
    high: list[str] = []
    low: list[str] = []

    for layer_idx in range(_NUM_HIDDEN_LAYERS):
        layer_prefix = f"model.layers.{layer_idx}."
        for suffix in _TEXT_ENCODER_HIGH_SUFFIXES:
            high.append(layer_prefix + suffix)
        for suffix in _TEXT_ENCODER_LOW_SUFFIXES:
            low.append(layer_prefix + suffix)

    return {"high": high, "low": low}