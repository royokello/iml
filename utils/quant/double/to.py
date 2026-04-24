import torch
import torch.nn.functional as F

INT4_MIN = -8
INT4_MAX = 7
SCALE_MIN = torch.finfo(torch.float32).tiny
LOCAL_SCALE_CODE_MAX = 127

SUPER_BLOCK_SIZE = 128
SUB_BLOCK_SIZE = 16
SUB_BLOCKS_PER_SUPER = SUPER_BLOCK_SIZE // SUB_BLOCK_SIZE


def quantize_to_double_block(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not torch.is_floating_point(tensor):
        raise TypeError("Quantization expects fp16/fp32 tensors.")

    arr32 = tensor.detach().to(dtype=torch.float32).flatten()

    if arr32.numel() == 0:
        return (
            torch.empty((0,), dtype=torch.int8, device=tensor.device),
            torch.empty((0, SUB_BLOCKS_PER_SUPER), dtype=torch.int8, device=tensor.device),
            torch.empty((0,), dtype=torch.float16, device=tensor.device),
        )

    pad_len = (SUPER_BLOCK_SIZE - (arr32.numel() % SUPER_BLOCK_SIZE)) % SUPER_BLOCK_SIZE
    if pad_len:
        arr32 = F.pad(arr32, (0, pad_len))

    super_blocks = arr32.view(-1, SUPER_BLOCK_SIZE)
    sub_blocks = super_blocks.view(-1, SUB_BLOCKS_PER_SUPER, SUB_BLOCK_SIZE)

    local_absmax = sub_blocks.abs().amax(dim=2)
    super_absmax = local_absmax.amax(dim=1)

    super_scales_fp32 = torch.clamp(super_absmax / (LOCAL_SCALE_CODE_MAX * INT4_MAX), min=SCALE_MIN)

    # Store 0 for true zero blocks. For nonzero blocks, use ceil so the decoded
    # block scale does not undershoot the block absmax target.
    local_scale_codes_fp32 = torch.ceil(local_absmax / (super_scales_fp32.unsqueeze(1) * INT4_MAX))
    local_scale_codes = torch.where(
        local_absmax > 0,
        local_scale_codes_fp32.clamp(1, LOCAL_SCALE_CODE_MAX),
        torch.zeros_like(local_scale_codes_fp32),
    ).to(torch.int8)

    effective_scales = super_scales_fp32.unsqueeze(1) * local_scale_codes.to(torch.float32)
    safe_effective_scales = torch.where(
        local_scale_codes > 0,
        effective_scales,
        torch.ones_like(effective_scales),
    )

    quantized = torch.round(sub_blocks / safe_effective_scales.unsqueeze(-1))
    quantized = quantized.clamp(INT4_MIN, INT4_MAX).to(torch.int8).flatten()

    if quantized.numel() % 2:
        quantized = F.pad(quantized, (0, 1))

    paired = quantized.view(-1, 2)
    lo = paired[:, 0] & 0x0F
    hi = (paired[:, 1] & 0x0F) << 4
    packed = (lo | hi).to(torch.int8)

    return packed, local_scale_codes, super_scales_fp32.to(torch.float16)
