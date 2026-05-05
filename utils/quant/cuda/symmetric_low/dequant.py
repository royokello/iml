from __future__ import annotations

import importlib
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import torch

SUB_BLOCK_SIZE = 16
SUPER_BLOCK_SIZE = 256
HALF_SUPER_BLOCK_SIZE = 128
SUPPORTED_SUPER_BLOCK_SIZES = (SUPER_BLOCK_SIZE, HALF_SUPER_BLOCK_SIZE)
PACKED_WEIGHT_WORDS_PER_SUB_BLOCK = 2
SUB_SCALE_BITS = 6


def _select_super_block_size(row_size: int) -> int:
    for super_block_size in SUPPORTED_SUPER_BLOCK_SIZES:
        if row_size % super_block_size == 0:
            return super_block_size
    raise ValueError(
        "Unsupported linear weight shape for symmetric-low CUDA dequantization: "
        f"in_features={row_size}. Input features must be divisible by "
        f"{HALF_SUPER_BLOCK_SIZE} or {SUPER_BLOCK_SIZE}."
    )


def _packed_scale_words_per_super(sub_blocks_per_super: int) -> int:
    return (sub_blocks_per_super * SUB_SCALE_BITS + 31) // 32


@lru_cache(maxsize=1)
def _load_prebuilt_symmetric_low_dequant_module():
    module_name = "dequantize_from_symmetric_low_cuda"
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
        "Prebuilt module 'dequantize_from_symmetric_low_cuda' not found. Build it first with "
        "'python setup.py build_ext --inplace' in utils/quant/cuda/symmetric_low."
    )


def dequantize_from_symmetric_low(
    qweight: torch.Tensor,
    sub_scales: torch.Tensor,
    super_scales: torch.Tensor,
    original_shape: tuple[int, ...] | torch.Size,
) -> torch.Tensor:
    if qweight.dtype != torch.int32:
        raise TypeError("qweight must be int32.")
    if sub_scales.dtype != torch.int32:
        raise TypeError("sub_scales must be packed int32.")
    if super_scales.dtype != torch.float16:
        raise TypeError("super_scales must be float16.")
    if not qweight.is_cuda:
        raise TypeError("qweight must be a CUDA tensor.")
    if not sub_scales.is_cuda:
        raise TypeError("sub_scales must be a CUDA tensor.")
    if not super_scales.is_cuda:
        raise TypeError("super_scales must be a CUDA tensor.")

    original_shape = tuple(original_shape)
    if len(original_shape) != 2:
        raise ValueError(
            "original_shape must be a 2D linear weight shape "
            f"(out_features, in_features), got {original_shape}."
        )
    row_count = int(original_shape[0])
    row_size = int(original_shape[1])
    original_numel = row_count * row_size

    if qweight.ndim != 2:
        raise ValueError("qweight must have shape [num_super_blocks, packed_words_per_super_block].")
    if sub_scales.ndim != 2:
        raise ValueError("sub_scales must have shape [num_super_blocks, packed_scale_words_per_super].")
    if qweight.shape[0] != sub_scales.shape[0]:
        raise ValueError("qweight and sub_scales must have the same number of super-blocks.")

    super_block_size = _select_super_block_size(row_size)
    sub_blocks_per_super = super_block_size // SUB_BLOCK_SIZE
    blocks_per_row = row_size // super_block_size
    expected_num_super_blocks = row_count * blocks_per_row
    expected_packed_weight_words = sub_blocks_per_super * PACKED_WEIGHT_WORDS_PER_SUB_BLOCK
    expected_packed_scale_words = _packed_scale_words_per_super(sub_blocks_per_super)

    if qweight.shape != (expected_num_super_blocks, expected_packed_weight_words):
        raise ValueError(
            "qweight shape does not match original_shape: "
            f"expected {(expected_num_super_blocks, expected_packed_weight_words)}, got {tuple(qweight.shape)}."
        )
    if sub_scales.shape != (expected_num_super_blocks, expected_packed_scale_words):
        raise ValueError(
            "sub_scales shape does not match original_shape: "
            f"expected {(expected_num_super_blocks, expected_packed_scale_words)}, got {tuple(sub_scales.shape)}."
        )
    if super_scales.shape != (expected_num_super_blocks,):
        raise ValueError("super_scales must contain one value per super-block.")

    if original_numel == 0:
        return torch.empty(original_shape, dtype=torch.float16, device=qweight.device)

    output = torch.empty((original_numel,), dtype=torch.float16, device=qweight.device)
    module = _load_prebuilt_symmetric_low_dequant_module()
    module.dequantize_from_symmetric_low_fp16(
        qweight.contiguous(),
        sub_scales.contiguous(),
        super_scales.contiguous(),
        output,
        original_numel,
        super_block_size,
    )
    return output.view(original_shape)


__all__ = ["dequantize_from_symmetric_low"]
