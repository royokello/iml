from __future__ import annotations

STORED_QUANT_METHODS = (
    "symmetric_high",
    "symmetric_med",
    "symmetric_low",
    "affine_high",
    "affine_med",
    "affine_low",
)

def quant_method_family(method: str) -> str:
    prefix = method.split("_", 1)[0].split("-", 1)[0]
    if prefix in {"sym", "symmetric"}:
        return "symmetric"
    return "affine"


def quant_method_mode(method: str) -> str:
    return method.rsplit("_", 1)[-1].rsplit("-", 1)[-1]


__all__ = [
    "STORED_QUANT_METHODS",
    "quant_method_family",
    "quant_method_mode",
]
