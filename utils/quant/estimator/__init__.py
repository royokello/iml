__all__ = [
    "get_safetensors_quantized_size",
    "get_safetensors_tensor_metadata",
    "get_torch_quantized_size",
    "get_torch_tensor_metadata",
]


def __getattr__(name: str):
    if name in {"get_safetensors_quantized_size", "get_safetensors_tensor_metadata"}:
        from .safetensors import get_safetensors_quantized_size, get_safetensors_tensor_metadata

        return {
            "get_safetensors_quantized_size": get_safetensors_quantized_size,
            "get_safetensors_tensor_metadata": get_safetensors_tensor_metadata,
        }[name]
    if name in {"get_torch_quantized_size", "get_torch_tensor_metadata"}:
        from .torch import get_torch_quantized_size, get_torch_tensor_metadata

        return {
            "get_torch_quantized_size": get_torch_quantized_size,
            "get_torch_tensor_metadata": get_torch_tensor_metadata,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
