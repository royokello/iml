from __future__ import annotations

from pathlib import Path

import torch

from .utils import estimate_quantized_safetensors_size

_TORCH_TO_SAFETENSORS_DTYPE = {
    torch.int8: "I8",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.int32: "I32",
    torch.float32: "F32",
}


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(loaded, dict):
        for key in ("state_dict", "model", "module"):
            value = loaded.get(key)
            if isinstance(value, dict):
                return value
        return loaded

    raise ValueError(f"Unsupported PyTorch checkpoint structure: {path}")


def _tensor_metadata(name: str, tensor: torch.Tensor) -> tuple[str, str, tuple[int, ...]]:
    try:
        dtype = _TORCH_TO_SAFETENSORS_DTYPE[tensor.dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported tensor dtype for {name}: {tensor.dtype}") from exc

    return name, dtype, tuple(int(dim) for dim in tensor.shape)


def get_torch_tensor_metadata(path: Path) -> list[tuple[str, str, tuple[int, ...]]]:
    path = Path(path)
    if path.suffix != ".pth":
        raise ValueError(f"Expected a .pth file: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"PyTorch checkpoint not found: {path}")

    state_dict = _load_state_dict(path)
    return [_tensor_metadata(name, tensor) for name, tensor in state_dict.items()]


def get_torch_quantized_size(
    path: Path,
    methods_by_name: dict[str, str],
    inclusion_prefix: str | tuple[str, ...] | None,
    exclusion_prefix: str | tuple[str, ...] | None,
) -> int:
    """Estimate the safetensors output size for ``_quantize_state_dict``."""
    path = Path(path)
    if path.suffix != ".pth":
        raise ValueError(f"Expected a .pth file: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"PyTorch checkpoint not found: {path}")

    return estimate_quantized_safetensors_size(
        get_torch_tensor_metadata(path),
        methods_by_name,
        inclusion_prefix,
        exclusion_prefix,
    )


__all__ = ["get_torch_quantized_size", "get_torch_tensor_metadata"]
