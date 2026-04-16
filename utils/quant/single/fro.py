import torch
import torch.nn.functional as F

BLOCK_SIZE = 32


def dequantize_from_single_block(
    tensor: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    if tensor.dtype != torch.int8:
        raise TypeError("tensor must be int8.")

    q = tensor.to(torch.float16)
    original_shape = q.shape
    flattened = q.flatten()

    if flattened.numel() == 0:
        return torch.empty_like(q, dtype=torch.float16)

    pad_len = (BLOCK_SIZE - (flattened.numel() % BLOCK_SIZE)) % BLOCK_SIZE
    if pad_len:
        flattened = F.pad(flattened, (0, pad_len))

    blocks = flattened.view(-1, BLOCK_SIZE)
    dequantized = blocks * scales.to(torch.float16).view(-1, 1)
    dequantized_flat = dequantized.flatten()

    if pad_len:
        dequantized_flat = dequantized_flat[:-pad_len]

    return dequantized_flat.view(original_shape)