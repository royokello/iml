# flux2/quant/text_encoder.py

_NUM_HIDDEN_LAYERS = 36

_QWEN_LINEAR_WEIGHT_SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)

# Suffixes that receive higher bit‑width quantisation
_HIGH_SUFFIXES = {
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.down_proj.weight",
}


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
        for suffix in _QWEN_LINEAR_WEIGHT_SUFFIXES:
            full_name = layer_prefix + suffix
            if suffix in _HIGH_SUFFIXES:
                high.append(full_name)
            else:
                low.append(full_name)

    return {"high": high, "low": low}