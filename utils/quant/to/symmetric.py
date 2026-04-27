import torch
import torch.nn.functional as F

INT6_MIN = -32
INT6_MAX = 31
INT8_MIN = -128
INT8_MAX = 127
SCALE_CODE_MIN = 0
SCALE_CODE_MAX = 127

SUPER_BLOCK_SIZE = 256
SUB_BLOCK_SIZE = 16
INT6_PACKED_WORDS_PER_SUB_BLOCK = 3


def _pack_int6_sub_blocks(qweight: torch.Tensor) -> torch.Tensor:
    if qweight.shape[-1] != SUB_BLOCK_SIZE:
        raise ValueError(f"int6 packing expects sub-blocks of {SUB_BLOCK_SIZE} values.")

    codes = qweight.to(torch.int64) & 0x3F
    packed = torch.zeros(
        (*codes.shape[:-1], INT6_PACKED_WORDS_PER_SUB_BLOCK),
        dtype=torch.int64,
        device=qweight.device,
    )

    bit_offset = 0
    for value_index in range(SUB_BLOCK_SIZE):
        word_index = bit_offset // 32
        word_shift = bit_offset % 32
        value = codes[..., value_index]

        packed[..., word_index] |= value << word_shift
        if word_shift > 26:
            packed[..., word_index + 1] |= value >> (32 - word_shift)

        bit_offset += 6

    signed_packed = torch.where(packed >= (1 << 31), packed - (1 << 32), packed)
    return signed_packed.to(torch.int32).view(qweight.shape[0], -1)


def quantize_to_symmetric(
    tensor: torch.Tensor,
    mode: str = "high",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    mode="high": int8 weights
    mode="low": packed signed int6 weights, 3 int32 words per 16-weight sub-block
    sub scale: int8
    superscale: fp16 d
    blocks: 16 weights per sub block x 16 sub blocks = 256 weights super block
    """
    mode = mode.strip().lower()
    if mode not in {"high", "low"}:
        raise ValueError(f"Unsupported symmetric quantization mode: {mode!r}. Expected 'high' or 'low'.")
    if not torch.is_floating_point(tensor):
        raise TypeError("Quantization expects fp16/fp32 tensors.")

    block_size = int(tensor.shape[1]) if tensor.ndim == 2 else int(tensor.numel())
    super_block_size = SUPER_BLOCK_SIZE if block_size % SUPER_BLOCK_SIZE == 0 else block_size
    if super_block_size % SUB_BLOCK_SIZE != 0:
        raise ValueError(
            "Unsupported block size for symmetric quantization: "
            f"{block_size}. Block size must be divisible by {SUB_BLOCK_SIZE}."
        )
    sub_blocks_per_super = super_block_size // SUB_BLOCK_SIZE
    qweight_columns = (
        super_block_size
        if mode == "high"
        else sub_blocks_per_super * INT6_PACKED_WORDS_PER_SUB_BLOCK
    )
    qweight_dtype = torch.int8 if mode == "high" else torch.int32

    if tensor.numel() == 0:
        return (
            torch.empty((0, qweight_columns), dtype=qweight_dtype, device=tensor.device),
            torch.empty((0, sub_blocks_per_super), dtype=torch.int8, device=tensor.device),
            torch.empty((0,), dtype=torch.float16, device=tensor.device),
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

    sub_blocks = super_blocks.view(-1, sub_blocks_per_super, SUB_BLOCK_SIZE)

    absmax = sub_blocks.abs().amax(dim=2)
    weight_min = INT8_MIN if mode == "high" else INT6_MIN
    weight_max = INT8_MAX if mode == "high" else INT6_MAX
    sub_scale_values = absmax / float(weight_max)
    safe_sub_scale_values = torch.where(
        sub_scale_values > 0,
        sub_scale_values,
        torch.ones_like(sub_scale_values),
    )

    qweight = torch.round(sub_blocks / safe_sub_scale_values.unsqueeze(-1))
    qweight = torch.where(
        sub_scale_values.unsqueeze(-1) > 0,
        qweight,
        torch.zeros_like(qweight),
    )
    qweight = qweight.clamp(weight_min, weight_max).to(torch.int8)
    qweight_out = (
        qweight.view(-1, super_block_size)
        if mode == "high"
        else _pack_int6_sub_blocks(qweight)
    )

    super_scales = sub_scale_values.abs().amax(dim=1) / float(SCALE_CODE_MAX)
    safe_super_scales = torch.where(super_scales > 0, super_scales, torch.ones_like(super_scales))

    sub_scales = torch.round(sub_scale_values / safe_super_scales.unsqueeze(1))
    sub_scales = torch.where(
        super_scales.unsqueeze(1) > 0,
        sub_scales,
        torch.zeros_like(sub_scales),
    )
    sub_scales = sub_scales.clamp(SCALE_CODE_MIN, SCALE_CODE_MAX).to(torch.int8)

    return (
        qweight_out,
        sub_scales,
        super_scales.to(torch.float16),
    )
