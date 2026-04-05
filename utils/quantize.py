"""Utilities for simple block-wise tensor quantization.

The helper in this module is intentionally small: it takes a floating-point
tensor, flattens it into fixed-size blocks, computes a per-block scale, and
returns the quantized values plus the stored scales.

Supported modes:
- `quantization_precision="fp16"`: casts the tensor to fp16 and returns a
  trivial scale tensor.
- `quantization_precision="int8"`: performs symmetric blockwise int8
  quantization with a per-block scale.
- `quantization_precision="int4"`: performs symmetric blockwise int4
  quantization and packs two signed 4-bit values into each stored byte.

Scale storage modes:
- `fp32`: keep scales as float32.
- `fp16`: keep scales as float16.
- `e8m0`: store the base-2 exponent in int8 form and reconstruct the scale
  as `2 ** exponent` at use time.
"""

import torch
import torch.nn.functional as F

INT8_MIN = -128
INT8_MAX = 127
INT4_MIN = -8
INT4_MAX = 7
SCALE_MIN = torch.finfo(torch.float32).tiny

def quantize_to_block(
    tensor: torch.Tensor,
    block_size: int = 64,
    quantization_precision: str = "fp16",
    scaling_precision: str = "fp32",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a floating-point tensor in fixed-size blocks.

    Args:
        tensor: Input floating-point tensor to quantize.
        block_size: Number of values per quantization block. Must be 32, 64,
            or 128.
        quantization_precision: Output precision for the data tensor.
            Use `"fp16"` for a simple cast, `"int8"` for blockwise int8
            quantization, or `"int4"` for packed blockwise int4 quantization.
        scaling_precision: Storage format for the per-block scale values.
            `"fp32"` and `"fp16"` store the scale directly, while `"e8m0"`
            stores the log2 exponent in int8 form.

    Returns:
        A pair `(quantized_tensor, scales)`:
        - `quantized_tensor` has the same shape as the input for `fp16` and
          `int8`, and a packed byte shape for `int4`.
        - `scales` is either a scalar tensor for fp16 mode or one scale per
          quantization block for int8/int4 mode.

    Notes:
        The tensor is flattened before block processing and padded to a full
        block if needed. Padding is removed before returning the quantized
        tensor. In `int4` mode, the returned tensor stores packed bytes.
    """
    if not torch.is_floating_point(tensor):
        raise TypeError("Quantization expects fp16/fp32 tensors.")

    if block_size not in {32, 64, 128}:
        raise ValueError("block_size must be one of: 32, 64, 128.")

    if quantization_precision not in {"fp16", "int8", "int4"}:
        raise ValueError('quantization_precision must be "fp16", "int8", or "int4".')

    if scaling_precision not in {"fp32", "fp16", "e8m0"}:
        raise ValueError('scaling_precision must be "fp32", "fp16", or "e8m0".')

    arr32 = tensor.detach().to(dtype=torch.float32)

    if arr32.numel() == 0:
        if quantization_precision == "fp16":
            quantized = arr32.to(dtype=torch.float16)
        elif quantization_precision == "int8":
            quantized = arr32.to(dtype=torch.int8)
        else:
            quantized = arr32.flatten().to(dtype=torch.uint8)

        if scaling_precision == "fp32":
            scales = torch.ones((), dtype=torch.float32, device=tensor.device)
        elif scaling_precision == "fp16":
            scales = torch.ones((), dtype=torch.float16, device=tensor.device)
        else:
            scales = torch.zeros((), dtype=torch.int8, device=tensor.device)

        return quantized, scales

    if quantization_precision == "fp16":
        if scaling_precision == "fp32":
            scales = torch.ones((), dtype=torch.float32, device=tensor.device)
        elif scaling_precision == "fp16":
            scales = torch.ones((), dtype=torch.float16, device=tensor.device)
        else:
            scales = torch.zeros((), dtype=torch.int8, device=tensor.device)

        return arr32.to(dtype=torch.float16), scales

    original_shape = arr32.shape
    flattened = arr32.flatten()

    pad_len = (block_size - (flattened.numel() % block_size)) % block_size
    if pad_len:
        # Pad only for block formation; remove it before returning.
        flattened = F.pad(flattened, (0, pad_len))

    reshaped = flattened.view(-1, block_size)
    max_abs = reshaped.abs().max(dim=1, keepdim=True).values
    max_q = float(INT8_MAX) if quantization_precision == "int8" else float(INT4_MAX)
    raw_scales = torch.clamp(max_abs / max_q, min=SCALE_MIN)

    if scaling_precision == "fp32":
        effective_scales = raw_scales
        stored_scales = raw_scales.flatten().to(dtype=torch.float32)
    elif scaling_precision == "fp16":
        effective_scales = raw_scales.to(dtype=torch.float16).to(dtype=torch.float32)
        stored_scales = raw_scales.flatten().to(dtype=torch.float16)
    else:
        # Store the scale as a log2 exponent to keep the metadata compact.
        exponents = torch.round(torch.log2(raw_scales)).to(dtype=torch.int32)
        exponents = torch.clamp(exponents, -127, 127)
        effective_scales = torch.pow(
            torch.tensor(2.0, dtype=torch.float32, device=tensor.device),
            exponents.to(dtype=torch.float32),
        )
        stored_scales = exponents.flatten().to(dtype=torch.int8)

    if quantization_precision == "int8":
        quantized = torch.round(reshaped / effective_scales).clamp(INT8_MIN, INT8_MAX).to(dtype=torch.int8)
        quantized_flat = quantized.flatten()

        if pad_len:
            quantized_flat = quantized_flat[:-pad_len]

        return quantized_flat.view(original_shape), stored_scales

    quantized = torch.round(reshaped / effective_scales).clamp(INT4_MIN, INT4_MAX).to(dtype=torch.int8)
    quantized_flat = quantized.flatten()

    if pad_len:
        quantized_flat = quantized_flat[:-pad_len]

    if quantized_flat.numel() % 2:
        quantized_flat = F.pad(quantized_flat, (0, 1))

    paired = quantized_flat.view(-1, 2)
    lo = paired[:, 0] & 0x0F
    hi = (paired[:, 1] & 0x0F) << 4
    packed = (lo | hi).to(dtype=torch.uint8)

    return packed, stored_scales
