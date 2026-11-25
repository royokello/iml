from __future__ import annotations

from utils.quant.name import QUANT_METHODS_BY_MAIN_BITS

_QUANT_METHOD_SPECS = {
    "q2": ("affine", "low"),
    "q3": ("symmetric", "low"),
    "q4": ("affine", "med"),
    "q5": ("affine", "high"),
    "q6": ("symmetric", "med"),
    "q8": ("symmetric", "high"),
}

if set(_QUANT_METHOD_SPECS) != set(QUANT_METHODS_BY_MAIN_BITS):
    raise RuntimeError(
        "Quantization specs and the method ladder disagree: "
        f"{sorted(_QUANT_METHOD_SPECS)} against {sorted(QUANT_METHODS_BY_MAIN_BITS)}."
    )


def _quant_method_spec(method: str) -> tuple[str, str]:
    """Return the family and mode of a quantization method.

    Args:
        method: A base method name such as `q6`. A variant name such as
            `q6-mini` is rejected, because a rank expands to a base method
            before it reaches a quantized module.

    Returns:
        The quantization family (`symmetric` or `affine`) and the mode
        (`high`, `med`, or `low`).

    Raises:
        ValueError: If the method is not a base method in the ladder.
    """
    try:
        return _QUANT_METHOD_SPECS[method]
    except KeyError as exc:
        raise ValueError(f"Unknown base quantization method: {method!r}.") from exc


def quant_method_family(method: str) -> str:
    return _quant_method_spec(method)[0]


def quant_method_mode(method: str) -> str:
    return _quant_method_spec(method)[1]


__all__ = [
    "quant_method_family",
    "quant_method_mode",
]
