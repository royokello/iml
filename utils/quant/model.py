from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import torch
from safetensors import safe_open

from .to.affine import quantize_to_affine
from .to.symmetric import quantize_to_symmetric
from .validators import normalize_quant_method, quant_method_family, quant_method_mode


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".pth":
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(loaded, dict):
            for key in ("state_dict", "model", "module"):
                value = loaded.get(key)
                if isinstance(value, dict):
                    return value
            return loaded

    raise ValueError(f"Unsupported model file: {path}")


def _store_quantized_or_fp16(
    result: dict[str, torch.Tensor],
    name: str,
    tensor: torch.Tensor,
    target_method: str | None,
) -> None:
    if target_method is not None:
        target_method = normalize_quant_method(target_method)
        metadata_base_name = name.removesuffix(".weight")
        sub_scales_name = f"{metadata_base_name}.sub_scales"
        super_scales_name = f"{metadata_base_name}.super_scales"
        if quant_method_family(target_method) == "symmetric":
            mode = quant_method_mode(target_method)
            qweight, sub_scales, super_scales = quantize_to_symmetric(
                tensor,
                mode=mode,
            )
            result[name] = qweight
            result[sub_scales_name] = sub_scales
            if super_scales is not None:
                result[super_scales_name] = super_scales
        else:
            sub_mins_name = f"{metadata_base_name}.sub_mins"
            super_mins_name = f"{metadata_base_name}.super_mins"
            qweight, sub_scales, sub_mins, super_scales, super_mins = quantize_to_affine(
                tensor,
                mode=quant_method_mode(target_method),
            )
            result[name] = qweight
            result[sub_scales_name] = sub_scales
            result[sub_mins_name] = sub_mins
            result[super_scales_name] = super_scales
            result[super_mins_name] = super_mins
    elif tensor.dtype in {torch.float32, torch.bfloat16}:
        result[name] = tensor.to(dtype=torch.float16)
    else:
        result[name] = tensor


def _quantize_safetensors_file(
    path: Path,
    methods_by_name: dict[str, str],
    inclusion_prefix: str | tuple[str, ...] | None,
    exclusion_prefix: str | tuple[str, ...] | None,
    result: dict[str, torch.Tensor],
) -> None:
    skipped = 0
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        names = handle.keys()
        total = len(names)
        for i, name in enumerate(names, start=1):
            print(f"{i}/{total}: {name} ...")
            if inclusion_prefix is not None and not name.startswith(inclusion_prefix):
                skipped += 1
                continue

            if exclusion_prefix is not None and name.startswith(exclusion_prefix):
                skipped += 1
                continue

            tensor = handle.get_tensor(name)
            _store_quantized_or_fp16(result, name, tensor, methods_by_name.get(name))
            del tensor

    if skipped:
        print(f"Skipped {skipped} tensors from {path}.")


def _quantize_state_dict(
    path: Path,
    methods_by_name: dict[str, str],
    inclusion_prefix: str | tuple[str, ...] | None,
    exclusion_prefix: str | tuple[str, ...] | None,
    result: dict[str, torch.Tensor],
) -> None:
    state_dict = _load_state_dict(path)
    names = list(state_dict)
    total = len(names)
    skipped = 0
    for i, name in enumerate(names, start=1):
        print(f"{i}/{total}: {name} ...")
        tensor = state_dict.pop(name)
        if inclusion_prefix is not None and not name.startswith(inclusion_prefix):
            skipped += 1
            del tensor
            continue

        if exclusion_prefix is not None and name.startswith(exclusion_prefix):
            skipped += 1
            del tensor
            continue

        _store_quantized_or_fp16(result, name, tensor, methods_by_name.get(name))
        del tensor

    if skipped:
        print(f"Skipped {skipped} tensors from {path}.")


def quantize_model_tensors(
    files: str | Path | list[str | Path],
    targets: Mapping[str, Iterable[str]],
    inclusion_prefix: str | tuple[str, ...] | None = None,
    exclusion_prefix: str | tuple[str, ...] | None = None,
) -> dict[str, torch.Tensor]:
    if isinstance(files, (str, Path)):
        paths = [Path(files)]
    else:
        paths = [Path(file) for file in files]

    if not paths:
        raise ValueError("files must contain at least one model path.")

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")

    methods_by_name: dict[str, str] = {}
    for method, tensor_names in targets.items():
        for name in tensor_names:
            methods_by_name[name] = method

    if not methods_by_name:
        raise ValueError("targets must contain at least one tensor name.")

    result: dict[str, torch.Tensor] = {}

    for path in paths:
        if path.suffix == ".safetensors":
            _quantize_safetensors_file(
                path,
                methods_by_name,
                inclusion_prefix,
                exclusion_prefix,
                result,
            )
        else:
            _quantize_state_dict(
                path,
                methods_by_name,
                inclusion_prefix,
                exclusion_prefix,
                result,
            )

    return result


__all__ = ["quantize_model_tensors"]
