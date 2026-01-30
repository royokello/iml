from __future__ import annotations

import torch
import torch.nn.functional as F

INT8_MIN = -128
INT8_MAX = 127
SCALE_MIN = 1e-8
BLOCK_SIZE = 32


def quantize_to_block32(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not torch.is_floating_point(tensor):
        raise TypeError("Quantization expects fp16/fp32 tensors.")

    arr32 = tensor.detach().to(dtype=torch.float32)
    if arr32.numel() == 0:
        return arr32.to(dtype=torch.int8), torch.ones((), dtype=torch.float16, device=tensor.device)

    original_shape = arr32.shape
    flattened = arr32.flatten()

    pad_len = (BLOCK_SIZE - (flattened.numel() % BLOCK_SIZE)) % BLOCK_SIZE
    if pad_len:
        flattened = F.pad(flattened, (0, pad_len))

    reshaped = flattened.view(-1, BLOCK_SIZE)
    max_abs = reshaped.abs().max(dim=1, keepdim=True).values
    scales = torch.clamp(max_abs / float(INT8_MAX), min=SCALE_MIN)

    quantized = torch.round(reshaped / scales).clamp(INT8_MIN, INT8_MAX).to(dtype=torch.int8)
    quantized_flat = quantized.flatten()
    if pad_len:
        quantized_flat = quantized_flat[:-pad_len]

    return quantized_flat.view(original_shape), scales.flatten().to(dtype=torch.float16)
