from __future__ import annotations

import importlib
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import torch

SUB_BLOCK_SIZE = 32
SUPER_BLOCK_SIZE = 256
HALF_SUPER_BLOCK_SIZE = 128
PACKED_WORDS_PER_WEIGHT_SUB_BLOCK = 4
META_BITS = 6


@lru_cache(maxsize=1)
def _load_prebuilt_affine_med_dequant_module():
    module_name = "dequantize_from_affine_med_cuda"
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
        "Prebuilt module 'dequantize_from_affine_med_cuda' not found. Build it first with "
        "'python setup.py build_ext --inplace' in utils/quant/cuda/affine_med."
    )


def _select_super_block_size(row_size: int) -> int:
    if row_size % SUPER_BLOCK_SIZE == 0:
        return SUPER_BLOCK_SIZE
    if row_size % HALF_SUPER_BLOCK_SIZE == 0:
        return HALF_SUPER_BLOCK_SIZE
    raise ValueError(
        "Unsupported linear weight shape for affine med dequantization: "
        f"in_features={row_size}. Input features must be divisible by "
        f"{HALF_SUPER_BLOCK_SIZE} or {SUPER_BLOCK_SIZE}."
    )


def _packed_words_for_values(value_count: int, bits: int) -> int:
    return (value_count * bits + 31) // 32


def dequantize_from_affine_med(
    qweight: torch.Tensor,
    sub_scales: torch.Tensor,
    sub_mins: torch.Tensor,
    super_scales: torch.Tensor,
    super_mins: torch.Tensor,
    original_shape: tuple[int, ...] | torch.Size,
) -> torch.Tensor:
    if qweight.dtype != torch.int32:
        raise TypeError("qweight must be int32.")
    if sub_scales.dtype != torch.int32:
        raise TypeError("sub_scales must be int32.")
    if sub_mins.dtype != torch.int32:
        raise TypeError("sub_mins must be int32.")
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
    if len(original_shape) != 2:
        raise ValueError(
            "Affine med dequantization expects a 2D linear weight shape "
            f"(out_features, in_features), got {original_shape}."
        )

    row_count = int(original_shape[0])
    row_size = int(original_shape[1])
    original_numel = row_count * row_size

    if original_numel == 0:
        return torch.empty(original_shape, dtype=torch.float16, device=qweight.device)

    super_block_size = _select_super_block_size(row_size)
    sub_blocks_per_super = super_block_size // SUB_BLOCK_SIZE
    blocks_per_row = row_size // super_block_size
    num_super_blocks = row_count * blocks_per_row
    metadata_words_per_super = _packed_words_for_values(sub_blocks_per_super, META_BITS)
    expected_qweight_shape = (
        num_super_blocks,
        sub_blocks_per_super * PACKED_WORDS_PER_WEIGHT_SUB_BLOCK,
    )
    expected_metadata_shape = (num_super_blocks, metadata_words_per_super)

    if qweight.shape != expected_qweight_shape:
        raise ValueError(
            "qweight shape does not match original_shape for affine med: "
            f"expected {expected_qweight_shape}, got {tuple(qweight.shape)}."
        )
    if sub_scales.shape != expected_metadata_shape:
        raise ValueError(
            "sub_scales shape does not match original_shape for affine med: "
            f"expected {expected_metadata_shape}, got {tuple(sub_scales.shape)}."
        )
    if sub_mins.shape != expected_metadata_shape:
        raise ValueError(
            "sub_mins shape does not match original_shape for affine med: "
            f"expected {expected_metadata_shape}, got {tuple(sub_mins.shape)}."
        )
    if super_scales.shape != (num_super_blocks,):
        raise ValueError(
            "super_scales shape does not match original_shape for affine med: "
            f"expected {(num_super_blocks,)}, got {tuple(super_scales.shape)}."
        )
    if super_mins.shape != (num_super_blocks,):
        raise ValueError(
            "super_mins shape does not match original_shape for affine med: "
            f"expected {(num_super_blocks,)}, got {tuple(super_mins.shape)}."
        )

    output = torch.empty((original_numel,), dtype=torch.float16, device=qweight.device)
    module = _load_prebuilt_affine_med_dequant_module()
    module.dequantize_from_affine_med_fp16(
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


__all__ = ["dequantize_from_affine_med"]
