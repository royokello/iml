from __future__ import annotations

import importlib
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import torch

from .to import SUB_BLOCKS_PER_SUPER


@lru_cache(maxsize=1)
def _load_prebuilt_double_dequant_module():
    module_name = "dequantize_from_double_block_cuda"
    cuda_dir = Path(__file__).resolve().parent / "cuda"

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
        "Prebuilt module 'dequantize_from_double_block_cuda' not found. Build it first with "
        "'python setup.py build_ext --inplace' in utils/quant/double/cuda."
    )


def dequantize_from_double_block(
    tensor: torch.Tensor,
    sub_scales: torch.Tensor,
    super_scales: torch.Tensor,
    *,
    original_numel: int | None = None,
) -> torch.Tensor:
    if tensor.dtype != torch.int8:
        raise TypeError("tensor must be int8.")
    if sub_scales.dtype != torch.int8:
        raise TypeError("sub_scales must be int8.")
    if super_scales.dtype != torch.float16:
        raise TypeError("super_scales must be float16.")
    if not tensor.is_cuda:
        raise TypeError("tensor must be a CUDA tensor.")
    if not sub_scales.is_cuda:
        raise TypeError("sub_scales must be a CUDA tensor.")
    if not super_scales.is_cuda:
        raise TypeError("super_scales must be a CUDA tensor.")
    if sub_scales.numel() != super_scales.numel() * SUB_BLOCKS_PER_SUPER:
        raise ValueError(
            f"sub_scales must contain exactly {SUB_BLOCKS_PER_SUPER} int8 values per super block."
        )

    if original_numel is None:
        original_numel = tensor.numel() * 2
    elif original_numel < 0:
        raise ValueError("original_numel must be non-negative.")

    decoded_capacity = tensor.numel() * 2
    if original_numel > decoded_capacity:
        raise ValueError("original_numel exceeds the packed tensor decode capacity.")

    original_shape = (original_numel,)

    if tensor.numel() == 0:
        return torch.empty(original_shape, device=tensor.device, dtype=torch.float16)

    output_fp16 = torch.empty(original_shape, device=tensor.device, dtype=torch.float16)
    module = _load_prebuilt_double_dequant_module()
    module.dequantize_from_double_block_fp16(
        tensor.contiguous(),
        sub_scales.contiguous(),
        super_scales.contiguous(),
        output_fp16,
        original_numel,
    )
    return output_fp16
