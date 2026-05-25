__all__ = [
    "quantize_model_tensors",
    "QuantizedLinear",
    "replace_targeted_linear_modules",
    "target_tensors_to_linear_names",
    "quantize_to_affine",
    "quantize_to_symmetric",
]


def __getattr__(name: str):
    if name == "quantize_model_tensors":
        from .model import quantize_model_tensors

        return quantize_model_tensors
    if name == "QuantizedLinear":
        from .linear import QuantizedLinear

        return QuantizedLinear
    if name in {"replace_targeted_linear_modules", "target_tensors_to_linear_names"}:
        from .replace import replace_targeted_linear_modules, target_tensors_to_linear_names

        return {
            "replace_targeted_linear_modules": replace_targeted_linear_modules,
            "target_tensors_to_linear_names": target_tensors_to_linear_names,
        }[name]
    if name == "quantize_to_affine":
        from .to.affine import quantize_to_affine

        return quantize_to_affine
    if name == "quantize_to_symmetric":
        from .to.symmetric import quantize_to_symmetric

        return quantize_to_symmetric
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
