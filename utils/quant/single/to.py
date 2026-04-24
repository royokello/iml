import torch
import torch.nn.functional as F

INT8_MIN = -128
INT8_MAX = 127
SCALE_MIN = torch.finfo(torch.float32).tiny

BLOCK_SIZE = 16


def quantize_to_single_block(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not torch.is_floating_point(tensor):
        raise TypeError("Quantization expects fp16/fp32 tensors.")

    arr32 = tensor.detach().to(dtype=torch.float32)
    original_shape = arr32.shape
    flattened = arr32.flatten()

    if flattened.numel() == 0:
        return (
            torch.empty_like(arr32, dtype=torch.int8),
            torch.empty((0,), dtype=torch.float16, device=tensor.device),
        )

    pad_len = (BLOCK_SIZE - (flattened.numel() % BLOCK_SIZE)) % BLOCK_SIZE
    if pad_len:
        flattened = F.pad(flattened, (0, pad_len))

    blocks = flattened.view(-1, BLOCK_SIZE)
    max_abs = blocks.abs().amax(dim=1, keepdim=True)
    scales = torch.clamp(max_abs / float(INT8_MAX), min=SCALE_MIN)

    quantized = torch.round(blocks / scales).clamp(INT8_MIN, INT8_MAX).to(torch.int8)
    quantized_flat = quantized.flatten()

    if pad_len:
        quantized_flat = quantized_flat[:-pad_len]

    return quantized_flat.view(original_shape), scales.flatten().to(torch.float16)
