from __future__ import annotations

import importlib
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import torch

HIGH_BLOCK_SIZE = 32


@lru_cache(maxsize=1)
def _load_prebuilt_symmetric_high_dequant_module():
    module_name = "dequantize_from_symmetric_high_cuda"
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
        "Prebuilt module 'dequantize_from_symmetric_high_cuda' not found. Build it first with "
        "'python setup.py build_ext --inplace' in utils/quant/cuda/symmetric_high."
    )


def dequantize_from_symmetric_high(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    super_scales: torch.Tensor | None = None,
    original_shape: tuple[int, ...] | torch.Size | None = None,
) -> torch.Tensor:
    if original_shape is None:
        if super_scales is None:
            raise TypeError("original_shape is required.")
        original_shape = super_scales
        super_scales = None

    if qweight.dtype != torch.int8:
        raise TypeError("qweight must be int8.")
    if scales.dtype != torch.float16:
        raise TypeError("scales must be float16.")
    if super_scales is not None:
        raise TypeError("symmetric-high no longer uses super_scales; pass None.")
    if not qweight.is_cuda:
        raise TypeError("qweight must be a CUDA tensor.")
    if not scales.is_cuda:
        raise TypeError("scales must be a CUDA tensor.")

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
        raise ValueError("qweight must have shape [num_blocks, 32].")
    if scales.ndim != 1:
        raise ValueError("scales must have shape [num_blocks].")

    blocks_per_row = (row_size + HIGH_BLOCK_SIZE - 1) // HIGH_BLOCK_SIZE if row_size else 0
    expected_num_blocks = row_count * blocks_per_row

    if qweight.shape != (expected_num_blocks, HIGH_BLOCK_SIZE):
        raise ValueError(
            "qweight shape does not match original_shape: "
            f"expected {(expected_num_blocks, HIGH_BLOCK_SIZE)}, got {tuple(qweight.shape)}."
        )
    if scales.shape != (expected_num_blocks,):
        raise ValueError(
            "scales shape does not match original_shape: "
            f"expected {(expected_num_blocks,)}, got {tuple(scales.shape)}."
        )

    if original_numel == 0:
        return torch.empty(original_shape, dtype=torch.float16, device=qweight.device)

    output = torch.empty((original_numel,), dtype=torch.float16, device=qweight.device)
    module = _load_prebuilt_symmetric_high_dequant_module()
    module.dequantize_from_symmetric_high_fp16(
        qweight.contiguous(),
        scales.contiguous(),
        output,
        original_numel,
        row_size,
        blocks_per_row,
    )
    return output.view(original_shape)


__all__ = ["dequantize_from_symmetric_high"]
