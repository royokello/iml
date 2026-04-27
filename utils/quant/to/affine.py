import torch
import torch.nn.functional as F

INT4_MIN = 0
INT4_MAX = 15
INT5_MIN = 0
INT5_MAX = 31
SCALE_CODE_MIN = 0
SCALE_CODE_MAX = 127
MIN_CODE_MIN = -128
MIN_CODE_MAX = 127

SUPER_BLOCK_SIZE = 256
SUB_BLOCK_SIZE = 32
INT5_PACKED_WORDS_PER_SUB_BLOCK = 5


def _pack_uint5_sub_blocks(qweight: torch.Tensor) -> torch.Tensor:
    if qweight.shape[-1] != SUB_BLOCK_SIZE:
        raise ValueError(f"uint5 packing expects sub-blocks of {SUB_BLOCK_SIZE} values.")

    codes = qweight.to(torch.int64) & 0x1F
    packed = torch.zeros(
        (*codes.shape[:-1], INT5_PACKED_WORDS_PER_SUB_BLOCK),
        dtype=torch.int64,
        device=qweight.device,
    )

    bit_offset = 0
    for value_index in range(SUB_BLOCK_SIZE):
        word_index = bit_offset // 32
        word_shift = bit_offset % 32
        value = codes[..., value_index]

        packed[..., word_index] |= value << word_shift
        if word_shift > 27:
            packed[..., word_index + 1] |= value >> (32 - word_shift)

        bit_offset += 5

    signed_packed = torch.where(packed >= (1 << 31), packed - (1 << 32), packed)
    return signed_packed.to(torch.int32).view(qweight.shape[0], -1)


def quantize_to_affine(
    tensor: torch.Tensor,
    mode: str = "low",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    mode="high": packed uint5 weights, 5 int32 words per 32-weight sub-block
    mode="low": packed uint4 weights
    sub scale: int8 scale + int8 min
    superscale: fp16 d + fp16 dmin
    blocks: 32 weights per sub block x 8 sub blocks = 256 weights super block
    """
    mode = mode.strip().lower()
    if mode not in {"high", "low"}:
        raise ValueError(f"Unsupported affine quantization mode: {mode!r}. Expected 'high' or 'low'.")
    if not torch.is_floating_point(tensor):
        raise TypeError("Quantization expects fp16/fp32 tensors.")

    block_size = int(tensor.shape[1]) if tensor.ndim == 2 else int(tensor.numel())
    super_block_size = SUPER_BLOCK_SIZE if block_size % SUPER_BLOCK_SIZE == 0 else block_size
    if super_block_size % SUB_BLOCK_SIZE != 0:
        raise ValueError(
            "Unsupported block size for affine quantization: "
            f"{block_size}. Block size must be divisible by {SUB_BLOCK_SIZE}."
        )
    sub_blocks_per_super = super_block_size // SUB_BLOCK_SIZE
    qweight_columns = (
        super_block_size // 2
        if mode == "low"
        else sub_blocks_per_super * INT5_PACKED_WORDS_PER_SUB_BLOCK
    )
    qweight_dtype = torch.uint8 if mode == "low" else torch.int32

    if tensor.numel() == 0:
        return (
            torch.empty((0, qweight_columns), dtype=qweight_dtype, device=tensor.device),
            torch.empty((0, sub_blocks_per_super), dtype=torch.int8, device=tensor.device),
            torch.empty((0, sub_blocks_per_super), dtype=torch.int8, device=tensor.device),
            torch.empty((0,), dtype=torch.float16, device=tensor.device),
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

    sub_mins = sub_blocks.amin(dim=2)
    sub_maxes = sub_blocks.amax(dim=2)
    weight_min = INT4_MIN if mode == "low" else INT5_MIN
    weight_max = INT4_MAX if mode == "low" else INT5_MAX
    sub_scales = (sub_maxes - sub_mins) / float(weight_max)

    safe_sub_scales = torch.where(
        sub_scales > 0,
        sub_scales,
        torch.ones_like(sub_scales),
    )
    quantized = torch.round((sub_blocks - sub_mins.unsqueeze(-1)) / safe_sub_scales.unsqueeze(-1))
    quantized = torch.where(
        sub_scales.unsqueeze(-1) > 0,
        quantized,
        torch.zeros_like(quantized),
    )
    quantized = quantized.clamp(weight_min, weight_max).to(torch.uint8)
    if mode == "low":
        paired = quantized.flatten().view(-1, 2)
        packed = (paired[:, 0] | (paired[:, 1] << 4)).to(torch.uint8)
        packed = packed.view(-1, super_block_size // 2)
    else:
        packed = _pack_uint5_sub_blocks(quantized)

    super_scales = sub_scales.abs().amax(dim=1) / float(SCALE_CODE_MAX)
    safe_super_scales = torch.where(super_scales > 0, super_scales, torch.ones_like(super_scales))
    sub_scale_codes = torch.round(sub_scales / safe_super_scales.unsqueeze(1))
    sub_scale_codes = torch.where(
        super_scales.unsqueeze(1) > 0,
        sub_scale_codes,
        torch.zeros_like(sub_scale_codes),
    )
    sub_scale_codes = sub_scale_codes.clamp(SCALE_CODE_MIN, SCALE_CODE_MAX).to(torch.int8)

    super_mins = sub_mins.abs().amax(dim=1) / float(SCALE_CODE_MAX)
    safe_super_mins = torch.where(super_mins > 0, super_mins, torch.ones_like(super_mins))
    sub_min_codes = torch.round(sub_mins / safe_super_mins.unsqueeze(1))
    sub_min_codes = torch.where(
        super_mins.unsqueeze(1) > 0,
        sub_min_codes,
        torch.zeros_like(sub_min_codes),
    )
    sub_min_codes = sub_min_codes.clamp(MIN_CODE_MIN, MIN_CODE_MAX).to(torch.int8)

    return (
        packed,
        sub_scale_codes,
        sub_min_codes,
        super_scales.to(torch.float16),
        super_mins.to(torch.float16),
    )
