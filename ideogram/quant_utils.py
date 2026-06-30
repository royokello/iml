from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import torch
from safetensors import safe_open

from utils.quant.model import _build_quantized_or_fp16_tensors


def _find_scale_map(handle: safe_open) -> dict[str, str]:
    scale_map: dict[str, str] = {}
    for name in handle.keys():
        if name.endswith("weight_scale"):
            weight_name = name.removesuffix("_scale")
            if weight_name in handle.keys():
                scale_map[weight_name] = name
    return scale_map


def quantize_f8_model_tensors(
    files: str | Path | list[str | Path],
    targets: Mapping[str, Iterable[str]],
    *,
    exclude_prefix: str | None = None,
) -> dict[str, torch.Tensor]:
    if isinstance(files, (str, Path)):
        paths = [Path(files)]
    else:
        paths = [Path(file) for file in files]

    methods_by_name: dict[str, str] = {}
    for method, tensor_names in targets.items():
        for name in tensor_names:
            methods_by_name[name] = method

    result: dict[str, torch.Tensor] = {}

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")

        with safe_open(str(path), framework="pt", device="cpu") as handle:
            scale_map = _find_scale_map(handle)
            names = handle.keys()
            total = len(names)
            processed = 0

            for i, name in enumerate(names, start=1):
                if name in scale_map.values():
                    continue
                if exclude_prefix is not None and name.startswith(exclude_prefix):
                    continue

                tensor = handle.get_tensor(name)
                scale_name = scale_map.get(name)

                if scale_name is not None:
                    scale = handle.get_tensor(scale_name)
                    tensor = tensor.to(device="cuda", dtype=torch.float32) * scale.to("cuda").unsqueeze(-1)

                method = methods_by_name.get(name)
                label = method or (
                    "fp16" if tensor.dtype in {torch.float32, torch.bfloat16} else "pass"
                )
                print(f"{i}/{total}: {name} {tuple(tensor.shape)} at {label}")
                processed += 1

                result.update(
                    _build_quantized_or_fp16_tensors(
                        name=name,
                        tensor=tensor,
                        target_method=method,
                    )
                )
                del tensor

        print(f"Processed {processed}/{total} tensors from {path}.")

    return result
