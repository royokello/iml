from __future__ import annotations

import importlib
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import torch

SUB_BLOCK_SIZE = 32


@lru_cache(maxsize=1)
def _load_prebuilt_affine_low_dequant_module():
    module_name = "dequantize_from_affine_low_cuda"
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
        "Prebuilt module 'dequantize_from_affine_low_cuda' not found. Build it first with "
        "'python setup.py build_ext --inplace' in utils/quant/cuda/affine_low."
    )


def dequantize_from_affine_low(
    qweight: torch.Tensor,
    sub_scales: torch.Tensor,
    sub_mins: torch.Tensor,
    super_scales: torch.Tensor,
    super_mins: torch.Tensor,
    original_shape: tuple[int, ...] | torch.Size,
) -> torch.Tensor:
    if qweight.dtype != torch.uint8:
        raise TypeError("qweight must be uint8.")
    if sub_scales.dtype != torch.int8:
        raise TypeError("sub_scales must be int8.")
    if sub_mins.dtype != torch.int8:
        raise TypeError("sub_mins must be int8.")
    if super_scales.dtype != torch.float16:
        raise TypeError("super_scales must be float16.")
    if super_mins.dtype != torch.float16:
        raise TypeError("super_mins must be float16.")
    if not qweight.is_cuda:
        raise TypeError("qweight must be a CUDA tensor.")
    if not sub_scales.is_cuda:
        raise TypeError("sub_scales must be a CUDA tensor.")
    if not sub_mins.is_cuda:
        raise TypeError("sub_mins must be a CUDA tensor.")
    if not super_scales.is_cuda:
        raise TypeError("super_scales must be a CUDA tensor.")
    if not super_mins.is_cuda:
        raise TypeError("super_mins must be a CUDA tensor.")

    original_shape = tuple(original_shape)
    original_numel = 1
    for dim in original_shape:
        original_numel *= int(dim)

    if sub_scales.ndim != 2:
        raise ValueError("sub_scales must have shape [num_super_blocks, sub_blocks_per_super].")
    if sub_mins.shape != sub_scales.shape:
        raise ValueError("sub_mins must have the same shape as sub_scales.")

    num_super_blocks, sub_blocks_per_super = sub_scales.shape
    super_block_size = int(sub_blocks_per_super) * SUB_BLOCK_SIZE
    padded_numel = int(num_super_blocks) * super_block_size

    if original_numel > padded_numel:
        raise ValueError(
            "original_shape contains more values than the quantized blocks: "
            f"expected at most {padded_numel}, got {original_numel}."
        )
    if qweight.numel() * 2 < padded_numel:
        raise ValueError("qweight does not contain enough packed values.")
    if super_scales.numel() != num_super_blocks:
        raise ValueError("super_scales must contain one value per super-block.")
    if super_mins.numel() != num_super_blocks:
        raise ValueError("super_mins must contain one value per super-block.")

    if original_numel == 0:
        return torch.empty(original_shape, dtype=torch.float16, device=qweight.device)

    output = torch.empty((original_numel,), dtype=torch.float16, device=qweight.device)
    module = _load_prebuilt_affine_low_dequant_module()
    module.dequantize_from_affine_low_fp16(
        qweight.contiguous(),
        sub_scales.contiguous(),
        sub_mins.contiguous(),
        super_scales.contiguous(),
        super_mins.contiguous(),
        output,
        original_numel,
        super_block_size,
    )
    return output.view(original_shape)


__all__ = ["dequantize_from_affine_low"]
