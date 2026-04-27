import torch

SUB_BLOCK_SIZE = 32
INT5_PACKED_WORDS_PER_SUB_BLOCK = 5


def _unpack_uint5_sub_blocks(qweight: torch.Tensor, num_super_blocks: int, sub_blocks_per_super: int) -> torch.Tensor:
    packed = qweight.contiguous().view(num_super_blocks, sub_blocks_per_super, INT5_PACKED_WORDS_PER_SUB_BLOCK)
    words = packed.to(torch.int64) & 0xFFFFFFFF
    unpacked = torch.empty(
        (num_super_blocks, sub_blocks_per_super, SUB_BLOCK_SIZE),
        dtype=torch.uint8,
        device=qweight.device,
    )

    for value_index in range(SUB_BLOCK_SIZE):
        bit_offset = value_index * 5
        word_index = bit_offset // 32
        word_shift = bit_offset % 32

        code = words[..., word_index] >> word_shift
        if word_shift > 27:
            code |= words[..., word_index + 1] << (32 - word_shift)
        unpacked[..., value_index] = (code & 0x1F).to(torch.uint8)

    return unpacked


def dequantize_from_affine(
    qweight: torch.Tensor,
    sub_scales: torch.Tensor,
    sub_mins: torch.Tensor,
    super_scales: torch.Tensor,
    super_mins: torch.Tensor,
    original_shape: tuple[int, ...] | torch.Size,
    mode: str = "low",
) -> torch.Tensor:
    """
Dequantize affine block-quantized weights.

Format:
    weights:
        mode="low": packed uint8 (2 × uint4 per byte)
        mode="high": packed int32 (5 × uint5 words per 32-weight sub-block)
    sub scales: int8 [0..127]
    sub mins: int8 [-128..127]
    super scales: fp16 d
    super mins: fp16 dmin
    blocks: 32 weights per sub-block × N per super-block

Reconstruction:
    q ∈ [0, 15] for mode="low"
    q ∈ [0, 31] for mode="high"

    real_scale = sub_scales * super_scales
    real_min   = sub_mins   * super_mins

    weight = q * real_scale + real_min

Args:
    qweight: packed uint8 weights
    sub_scales: int8 scale codes
    sub_mins: int8 min codes
    super_scales: fp16
    super_mins: fp16
    original_shape: shape of the original tensor before quantization
    mode: "low" for packed uint4 weights, "high" for packed uint5 weights

Returns:
    Dequantized float16 tensor with original_shape.
    Any padding introduced during quantization is removed.
"""
    mode = mode.strip().lower()
    if mode not in {"high", "low"}:
        raise ValueError(f"Unsupported affine dequantization mode: {mode!r}. Expected 'high' or 'low'.")
    if mode == "low" and qweight.dtype != torch.uint8:
        raise TypeError("qweight must be uint8 for low mode.")
    if mode == "high" and qweight.dtype != torch.int32:
        raise TypeError("qweight must be int32 for high mode.")
    if sub_scales.dtype != torch.int8:
        raise TypeError("sub_scales must be int8.")
    if sub_mins.dtype != torch.int8:
        raise TypeError("sub_mins must be int8.")
    if not torch.is_floating_point(super_scales):
        raise TypeError("super_scales must be floating point.")
    if not torch.is_floating_point(super_mins):
        raise TypeError("super_mins must be floating point.")

    original_shape = tuple(original_shape)
    original_numel = 1
    for dim in original_shape:
        original_numel *= int(dim)

    if original_numel == 0:
        return torch.empty(original_shape, dtype=torch.float16, device=qweight.device)

    if sub_scales.ndim != 2:
        raise ValueError("sub_scales must have shape [num_super_blocks, sub_blocks_per_super].")
    if sub_mins.shape != sub_scales.shape:
        raise ValueError("sub_mins must have the same shape as sub_scales.")

    num_super_blocks, sub_blocks_per_super = sub_scales.shape
    super_block_size = int(sub_blocks_per_super) * SUB_BLOCK_SIZE
    padded_numel = int(num_super_blocks) * super_block_size
    expected_packed_numel = int(num_super_blocks) * (
        super_block_size // 2
        if mode == "low"
        else int(sub_blocks_per_super) * INT5_PACKED_WORDS_PER_SUB_BLOCK
    )

    if original_numel > padded_numel:
        raise ValueError(
            "original_shape contains more values than the quantized blocks: "
            f"expected at most {padded_numel}, got {original_numel}."
        )
    if qweight.numel() != expected_packed_numel:
        raise ValueError(
            "qweight has the wrong number of packed values: "
            f"expected {expected_packed_numel}, got {qweight.numel()}."
        )
    if super_scales.numel() != num_super_blocks:
        raise ValueError("super_scales must contain one value per super-block.")
    if super_mins.numel() != num_super_blocks:
        raise ValueError("super_mins must contain one value per super-block.")

    if mode == "low":
        packed = qweight.contiguous().view(-1)
        unpacked = torch.empty((packed.numel() * 2,), dtype=torch.uint8, device=qweight.device)
        unpacked[0::2] = packed & 0x0F
        unpacked[1::2] = (packed >> 4) & 0x0F
        unpacked = unpacked.view(num_super_blocks, sub_blocks_per_super, SUB_BLOCK_SIZE)
    else:
        unpacked = _unpack_uint5_sub_blocks(qweight, num_super_blocks, sub_blocks_per_super)

    q = unpacked.to(torch.float32)
    real_scales = sub_scales.to(torch.float32) * super_scales.to(torch.float32).view(-1, 1)
    real_mins = sub_mins.to(torch.float32) * super_mins.to(torch.float32).view(-1, 1)

    dequantized = q * real_scales.unsqueeze(-1) + real_mins.unsqueeze(-1)
    flattened = dequantized.reshape(-1)[:original_numel].to(torch.float16)
    return flattened.view(original_shape)
