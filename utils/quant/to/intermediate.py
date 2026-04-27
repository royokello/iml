
import torch
import torch.nn.functional as F

INT8_MIN = -128
INT8_MAX = 127

SUPER_BLOCK_SIZE = 256
SUB_BLOCK_SIZE = 16


def quantize_to_intermediate(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    weights: int8
    sub scale: int16 bsums
    superscale: fp32 d
    blocks: 16 weights per sub block x 16 sub blocks = 256 weights super block
    """
    if not torch.is_floating_point(tensor):
        raise TypeError("Quantization expects fp16/fp32 tensors.")

    block_size = int(tensor.shape[1]) if tensor.ndim == 2 else int(tensor.numel())
    super_block_size = SUPER_BLOCK_SIZE if block_size % SUPER_BLOCK_SIZE == 0 else block_size
    if super_block_size % SUB_BLOCK_SIZE != 0:
        raise ValueError(
            "Unsupported block size for intermediate quantization: "
            f"{block_size}. Block size must be divisible by {SUB_BLOCK_SIZE}."
        )
    sub_blocks_per_super = super_block_size // SUB_BLOCK_SIZE

    if tensor.numel() == 0:
        return (
            torch.empty((0, super_block_size), dtype=torch.int8, device=tensor.device),
            torch.empty((0, sub_blocks_per_super), dtype=torch.int16, device=tensor.device),
            torch.empty((0,), dtype=torch.float32, device=tensor.device),
        )

    arr32 = tensor.detach().to(dtype=torch.float32)
    if arr32.ndim == 2:
        row_size = arr32.shape[1]
        pad_len = (super_block_size - (row_size % super_block_size)) % super_block_size
        if pad_len:
            arr32 = F.pad(arr32, (0, pad_len))
        super_blocks = arr32.contiguous().view(-1, super_block_size)
    else:
        flattened = arr32.flatten()
        pad_len = (super_block_size - (flattened.numel() % super_block_size)) % super_block_size
        if pad_len:
            flattened = F.pad(flattened, (0, pad_len))
        super_blocks = flattened.view(-1, super_block_size)

    absmax = super_blocks.abs().amax(dim=1)
    super_scales = absmax / float(INT8_MAX)
    safe_super_scales = torch.where(
        super_scales > 0,
        super_scales,
        torch.ones_like(super_scales),
    )

    qweight = torch.round(super_blocks / safe_super_scales.unsqueeze(1))
    qweight = torch.where(
        super_scales.unsqueeze(1) > 0,
        qweight,
        torch.zeros_like(qweight),
    )
    qweight = qweight.clamp(INT8_MIN, INT8_MAX).to(torch.int8)

    bsums = qweight.view(-1, sub_blocks_per_super, SUB_BLOCK_SIZE).sum(dim=2).to(torch.int16)

    return qweight, bsums, super_scales.to(torch.float32)
