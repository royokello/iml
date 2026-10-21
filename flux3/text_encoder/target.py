from __future__ import annotations

from flux2.text_encoder.target import (
    _TEXT_ENCODER_HIGH_SUFFIXES,
    _TEXT_ENCODER_LOW_SUFFIXES,
)

# Output layers 4..32 are used, so layers 0..32 are enough.
_NUM_LAYERS = 33
_LAYER_PREFIX = "model.language_model.layers."

# The quant file keeps only the language model, layers 0..32. The final norm
# and layers 33..35 are not used, so they are left out.
_FLUX3_TEXT_ENCODER_INCLUSION_PREFIX = "model.language_model."
_FLUX3_TEXT_ENCODER_EXCLUSION_PREFIXES = (
    "model.language_model.layers.33.",
    "model.language_model.layers.34.",
    "model.language_model.layers.35.",
    "model.language_model.norm.",
)


def _build_flux3_text_encoder_target_tensors() -> dict[str, list[str]]:
    high: list[str] = []
    low: list[str] = []

    for layer_idx in range(_NUM_LAYERS):
        layer_prefix = f"{_LAYER_PREFIX}{layer_idx}."
        for suffix in _TEXT_ENCODER_HIGH_SUFFIXES:
            high.append(layer_prefix + suffix)
        for suffix in _TEXT_ENCODER_LOW_SUFFIXES:
            low.append(layer_prefix + suffix)

    return {"high": high, "low": low}
