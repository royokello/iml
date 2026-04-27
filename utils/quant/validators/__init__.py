from __future__ import annotations

CLI_QUANT_METHODS = ("sym-high", "sym-low", "aff-high", "aff-low")
STORED_QUANT_METHODS = ("symmetric_high", "symmetric_low", "affine_high", "affine_low")

_METHOD_ALIASES = {
    "sym-high": "symmetric_high",
    "sym_high": "symmetric_high",
    "symmetric-high": "symmetric_high",
    "symmetric_high": "symmetric_high",
    "symmetric": "symmetric_high",
    "sym-low": "symmetric_low",
    "sym_low": "symmetric_low",
    "symmetric-low": "symmetric_low",
    "symmetric_low": "symmetric_low",
    "aff-high": "affine_high",
    "aff_high": "affine_high",
    "affine-high": "affine_high",
    "affine_high": "affine_high",
    "aff-low": "affine_low",
    "aff_low": "affine_low",
    "affine-low": "affine_low",
    "affine_low": "affine_low",
    "affine": "affine_low",
}


def normalize_quant_method(method: str) -> str:
    normalized = method.strip().lower()
    try:
        return _METHOD_ALIASES[normalized]
    except KeyError as exc:
        allowed = ", ".join(CLI_QUANT_METHODS)
        raise ValueError(f"Unsupported quantization method: {method!r}. Expected one of: {allowed}.") from exc


def quant_method_family(method: str) -> str:
    canonical = normalize_quant_method(method)
    if canonical.startswith("symmetric_"):
        return "symmetric"
    return "affine"


def quant_method_mode(method: str) -> str:
    return normalize_quant_method(method).rsplit("_", 1)[1]


__all__ = [
    "CLI_QUANT_METHODS",
    "STORED_QUANT_METHODS",
    "normalize_quant_method",
    "quant_method_family",
    "quant_method_mode",
]
