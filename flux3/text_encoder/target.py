from __future__ import annotations

from flux2.text_encoder.target import (
    _TEXT_ENCODER_HIGH_SUFFIXES,
    _TEXT_ENCODER_LOW_SUFFIXES,
)

# Output layers 4..32 are used, so layers 0..32 are enough.
_NUM_LAYERS = 33
_LAYER_PREFIX = "model.language_model.layers."


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
