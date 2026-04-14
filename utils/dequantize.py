"""Utilities for reversing the block-wise tensor quantization format.

This module reconstructs floating-point tensors from the compact representation
produced by `utils.quantize.quantize_to_block`.

The dequantizer supports:
- `quantization_precision="fp16"`: a simple dtype cast back to the requested
  output dtype.
- `quantization_precision="int8"`: blockwise symmetric dequantization using
  the stored per-block scales.
- `quantization_precision="int4"`: unpacking two signed 4-bit values from
  each stored byte before blockwise dequantization.

Scale storage modes:
- `fp32`: scales are stored directly as float32 values.
- `fp16`: scales are stored directly as float16 values.
- `e8m0`: scales are stored as int8 log2 exponents and expanded as `2 ** exp`.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F


@lru_cache(maxsize=1)
def _load_prebuilt_int4_dequant_module():
    module_name = "int4_dequant_cuda"
    cuda_dir = Path(__file__).resolve().parent / "cuda" / "int4_dequant"

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
        "Prebuilt module 'int4_dequant_cuda' not found. Build it first with "
        "'python setup.py build_ext --inplace' in utils/cuda/int4_dequant."
    )


def _can_use_cuda_int4_fast_path(
    tensor: torch.Tensor,
    scales: torch.Tensor,
    *,
    quantization_precision: str,
    scaling_precision: str,
) -> bool:
    return (
        quantization_precision == "int4"
        and scaling_precision == "fp16"
        and tensor.is_cuda
        and scales.is_cuda
        and tensor.dtype == torch.uint8
        and scales.dtype == torch.float16
    )


def dequantize_from_block(
    tensor: torch.Tensor,
    scales: torch.Tensor,
    block_size,
    quantization_precision,
    scaling_precision,
    output_dtype: torch.dtype = torch.float16,
    output_shape: tuple[int, ...] | torch.Size | None = None,
    use_int4_cuda_kernel: bool = True,
) -> torch.Tensor:
    """Reconstruct a floating-point tensor from blockwise quantized values.

    Args:
        tensor: Quantized tensor with the same shape as the original input.
        scales: Stored per-block scales or scale exponents.
        block_size: Number of values per quantization block. Must match the
            value used during quantization.
        quantization_precision: The data representation used by `tensor`.
            Use `"fp16"` for passthrough casts or `"int8"` for blockwise
            dequantization.
        scaling_precision: The storage format used for `scales`.
            `"fp32"` and `"fp16"` are direct scales; `"e8m0"` stores
            exponents.
        output_dtype: Desired dtype of the reconstructed tensor.
        output_shape: Expected dense output shape. Required to recover the
            original layout from packed int4 tensors.
        use_int4_cuda_kernel: Whether to use the prebuilt CUDA int4 fast path
            when the inputs match the supported fp16-scale CUDA configuration.

    Returns:
        The dequantized tensor reshaped to the original layout.

    Notes:
        The input is flattened, padded to block boundaries if needed, and then
        multiplied by the expanded scales block by block.
    """
    if block_size not in {32, 64, 128}:
        raise ValueError("block_size must be one of: 32, 64, 128.")

    if quantization_precision not in {"fp16", "int8", "int4"}:
        raise ValueError('quantization_precision must be "fp16", "int8", or "int4".')

    if scaling_precision not in {"fp32", "fp16", "e8m0"}:
        raise ValueError('scaling_precision must be "fp32", "fp16", or "e8m0".')

    if quantization_precision == "fp16":
        return tensor.to(dtype=output_dtype)

    if output_shape is None:
        if quantization_precision == "int4":
            original_shape = (tensor.numel() * 2,)
        else:
            original_shape = tuple(tensor.shape)
    else:
        original_shape = tuple(output_shape)

    original_numel = 1
    for dim in original_shape:
        original_numel *= dim

    if use_int4_cuda_kernel and _can_use_cuda_int4_fast_path(
        tensor,
        scales,
        quantization_precision=quantization_precision,
        scaling_precision=scaling_precision,
    ):
        output_fp16 = torch.empty(original_shape, device=tensor.device, dtype=torch.float16)
        module = _load_prebuilt_int4_dequant_module()
        module.dequantize_int4_fp16(
            tensor.contiguous(),
            scales.contiguous(),
            output_fp16,
            original_numel,
            block_size,
        )
        if output_dtype == torch.float16:
            return output_fp16
        return output_fp16.to(dtype=output_dtype)

    if quantization_precision == "int4":
        packed = tensor.flatten().to(dtype=torch.uint8)
        lo = (packed & 0x0F).to(dtype=torch.int8)
        hi = ((packed >> 4) & 0x0F).to(dtype=torch.int8)
        lo = lo - ((lo & 0x08) << 1)
        hi = hi - ((hi & 0x08) << 1)
        flattened = torch.stack([lo, hi], dim=1).flatten()
        flattened = flattened[:original_numel]
    else:
        flattened = tensor.flatten()

    pad_len = (block_size - (original_numel % block_size)) % block_size
    if pad_len:
        # Mirror the quantizer's padding so block shapes line up.
        flattened = F.pad(flattened, (0, pad_len))

    reshaped = flattened.view(-1, block_size).to(dtype=output_dtype)

    if scaling_precision == "e8m0":
        # Rebuild the scale from its exponent encoding.
        effective_scales = torch.pow(
            torch.tensor(2.0, dtype=output_dtype, device=tensor.device),
            scales.to(device=tensor.device, dtype=torch.float32).view(-1, 1),
        )
    else:
        effective_scales = scales.to(device=tensor.device, dtype=output_dtype).view(-1, 1)

    dequantized = reshaped * effective_scales
    dequantized_flat = dequantized.flatten()

    if pad_len:
        dequantized_flat = dequantized_flat[:-pad_len]

    return dequantized_flat.view(original_shape).to(dtype=output_dtype)
