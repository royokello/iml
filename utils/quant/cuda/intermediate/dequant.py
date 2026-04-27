from __future__ import annotations

import importlib
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def _load_prebuilt_intermediate_dequant_module():
    module_name = "dequantize_from_intermediate_cuda"
    cuda_dir = Path(__file__).resolve().parent

    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        pass

    suffixes = importlib.machinery.EXTENSION_SUFFIXES
    for suffix in suffixes:
        matches = sorted(cuda_dir.glob(f"{module_name}*{suffix}"))
        if not matches:
            continue
        module_path = matches[0]
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    raise ModuleNotFoundError(
        "Prebuilt module 'dequantize_from_intermediate_cuda' not found. Build it first with "
        "'python setup.py build_ext --inplace' in utils/quant/cuda/intermediate."
    )


def dequantize_from_intermediate(
    qweight: torch.Tensor,
    super_scales: torch.Tensor,
    original_shape: tuple[int, ...] | torch.Size,
) -> torch.Tensor:
    if qweight.dtype != torch.int8:
        raise TypeError("qweight must be int8.")
    if super_scales.dtype != torch.float32:
        raise TypeError("super_scales must be float32.")
    if not qweight.is_cuda:
        raise TypeError("qweight must be a CUDA tensor.")
    if not super_scales.is_cuda:
        raise TypeError("super_scales must be a CUDA tensor.")

    original_shape = tuple(original_shape)
    original_numel = 1
    for dim in original_shape:
        original_numel *= int(dim)

    if qweight.ndim != 2:
        raise ValueError("qweight must have shape [num_super_blocks, super_block_size].")

    num_super_blocks, super_block_size = qweight.shape
    padded_numel = int(num_super_blocks) * int(super_block_size)

    if original_numel > padded_numel:
        raise ValueError(
            "original_shape contains more values than the quantized blocks: "
            f"expected at most {padded_numel}, got {original_numel}."
        )
    if super_scales.numel() != num_super_blocks:
        raise ValueError("super_scales must contain one value per super-block.")

    if original_numel == 0:
        return torch.empty(original_shape, dtype=torch.float16, device=qweight.device)

    output = torch.empty((original_numel,), dtype=torch.float16, device=qweight.device)
    module = _load_prebuilt_intermediate_dequant_module()
    module.dequantize_from_intermediate_fp16(
        qweight.contiguous(),
        super_scales.contiguous(),
        output,
        original_numel,
        int(super_block_size),
    )
    return output.view(original_shape)


__all__ = ["dequantize_from_intermediate"]
