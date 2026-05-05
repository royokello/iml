from __future__ import annotations

from pathlib import Path

from safetensors import safe_open

from .utils import estimate_quantized_safetensors_size


def get_safetensors_tensor_metadata(path: Path) -> list[tuple[str, str, tuple[int, ...]]]:
    tensors: list[tuple[str, str, tuple[int, ...]]] = []
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for name in handle.keys():
            tensor_slice = handle.get_slice(name)
            dtype = tensor_slice.get_dtype()
            shape = tensor_slice.get_shape()
            tensor_shape = tuple(int(dim) for dim in shape)
            if any(dim < 0 for dim in tensor_shape):
                raise ValueError(f"Invalid negative dimension for {name!r} in {path}: {tensor_shape}")
            tensors.append((name, dtype, tensor_shape))
    return tensors


def get_safetensors_quantized_size(
    path: Path,
    methods_by_name: dict[str, str],
    inclusion_prefix: str | tuple[str, ...] | None,
    exclusion_prefix: str | tuple[str, ...] | None,
) -> int:
    """Estimate the output safetensors file size for ``_quantize_safetensors_file``.

    Tensor shapes and dtypes are read from the safetensors header only; tensor data is
    never materialized.
    """
    path = Path(path)
    if path.suffix != ".safetensors":
        raise ValueError(f"Expected a .safetensors file: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Safetensors file not found: {path}")

    return estimate_quantized_safetensors_size(
        get_safetensors_tensor_metadata(path),
        methods_by_name,
        inclusion_prefix,
        exclusion_prefix,
    )


__all__ = ["get_safetensors_quantized_size", "get_safetensors_tensor_metadata"]
