from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file as safe_load_file

from .double import quantize_to_double_block
from .single import quantize_to_single_block


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        return safe_load_file(str(path))

    if path.suffix == ".pth":
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(loaded, dict):
            for key in ("state_dict", "model", "module"):
                value = loaded.get(key)
                if isinstance(value, dict):
                    return value
            return loaded

    raise ValueError(f"Unsupported model file: {path}")


def quantize_model_tensors(
    files: str | Path | list[str | Path],
    tensors: list[str],
    method: str,
) -> dict[str, torch.Tensor]:
    if method not in {"single", "double"}:
        raise ValueError(f"Unsupported quantization method: {method!r}")

    if isinstance(files, (str, Path)):
        paths = [Path(files)]
    else:
        paths = [Path(file) for file in files]

    if not paths:
        raise ValueError("files must contain at least one model path.")

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")

    requested_names = list(dict.fromkeys(tensors))
    if not requested_names:
        raise ValueError("tensors must contain at least one tensor name.")

    requested = set(requested_names)
    result: dict[str, torch.Tensor] = {}

    for path in paths:
        state_dict = _load_state_dict(path)
        for name, tensor in state_dict.items():
            if name in requested:
                weight_name = name
                scales_name = f"{name}.scales"
                super_scales_name = f"{name}.super_scales"
                if method == "single":
                    qweight, scales = quantize_to_single_block(tensor)
                    result[weight_name] = qweight
                    result[scales_name] = scales
                else:
                    qweight, sub_scales, super_scales = quantize_to_double_block(tensor)
                    result[weight_name] = qweight
                    result[scales_name] = sub_scales
                    result[super_scales_name] = super_scales
            else:
                if tensor.dtype in {torch.float32, torch.bfloat16}:
                    result[name] = tensor.to(dtype=torch.float16)
                else:
                    result[name] = tensor

    return result


__all__ = ["quantize_model_tensors"]
