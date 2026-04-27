import torch

SUB_BLOCK_SIZE = 16
INT6_PACKED_WORDS_PER_SUB_BLOCK = 3


def _unpack_int6_sub_blocks(qweight: torch.Tensor, num_super_blocks: int, sub_blocks_per_super: int) -> torch.Tensor:
    packed = qweight.contiguous().view(num_super_blocks, sub_blocks_per_super, INT6_PACKED_WORDS_PER_SUB_BLOCK)
    words = packed.to(torch.int64) & 0xFFFFFFFF
    unpacked = torch.empty(
        (num_super_blocks, sub_blocks_per_super, SUB_BLOCK_SIZE),
        dtype=torch.int8,
        device=qweight.device,
    )

    for value_index in range(SUB_BLOCK_SIZE):
        bit_offset = value_index * 6
        word_index = bit_offset // 32
        word_shift = bit_offset % 32

        code = words[..., word_index] >> word_shift
        if word_shift > 26:
            code |= words[..., word_index + 1] << (32 - word_shift)
        code &= 0x3F
        unpacked[..., value_index] = ((code & 0x1F) - (code & 0x20)).to(torch.int8)

    return unpacked


def dequantize_from_symmetric(
    qweight: torch.Tensor,
    sub_scales: torch.Tensor,
    super_scales: torch.Tensor,
    original_shape: tuple[int, ...] | torch.Size,
    mode: str = "high",
) -> torch.Tensor:
    """
    Dequantize symmetric block-quantized weights (Q6_K-style) to float.

    Format:
        weights:
            mode="high": int8 (signed)
            mode="low": packed int32 (3 × signed int6 words per 16-weight sub-block)
        sub scales: int8 scale codes (non-negative semantics: 0..127)
        super scales: fp16 d
        blocks: 16 weights per sub-block × 16 sub-blocks = 256 weights per super-block

    Expected shapes:
        qweight:
            [num_super_blocks, 256]

        sub_scales:
            [num_super_blocks, 16]

        super_scales:
            [num_super_blocks]

    Inferred values:
        super_block_size = 256
        sub_blocks_per_super = sub_scales.shape[1] (=16)
        sub_block_size = 16

    Reconstruction:
        q ∈ [-128, 127] for mode="high"
        q ∈ [-32, 31] for mode="low"

        real_scale = sub_scales * super_scales

        weight = q * real_scale

    Behavior:
        - Expands sub-block scales across their 16-weight regions
        - Applies symmetric reconstruction (no offset/min term)
        - Removes any padding introduced during quantization
        - Reshapes to original_shape

    Args:
        qweight: int8 quantized weights
        sub_scales: int8 sub-block scale codes
        super_scales: fp16 super scale factors
        original_shape: original tensor shape before quantization
        mode: "high" for int8 weights, "low" for packed signed int6 weights

    Returns:
        Dequantized float16 tensor with original_shape
    """
    mode = mode.strip().lower()
    if mode not in {"high", "low"}:
        raise ValueError(f"Unsupported symmetric dequantization mode: {mode!r}. Expected 'high' or 'low'.")
    if mode == "high" and qweight.dtype != torch.int8:
        raise TypeError("qweight must be int8 for high mode.")
    if mode == "low" and qweight.dtype != torch.int32:
        raise TypeError("qweight must be int32 for low mode.")
    if sub_scales.dtype != torch.int8:
        raise TypeError("sub_scales must be int8.")
    if not torch.is_floating_point(super_scales):
        raise TypeError("super_scales must be floating point.")

    original_shape = tuple(original_shape)
    original_numel = 1
    for dim in original_shape:
        original_numel *= int(dim)

    if original_numel == 0:
        return torch.empty(original_shape, dtype=torch.float16, device=qweight.device)

    if qweight.ndim != 2:
        raise ValueError("qweight must have shape [num_super_blocks, super_block_size].")
    if sub_scales.ndim != 2:
        raise ValueError("sub_scales must have shape [num_super_blocks, sub_blocks_per_super].")
    if qweight.shape[0] != sub_scales.shape[0]:
        raise ValueError("qweight and sub_scales must have the same number of super-blocks.")

    num_super_blocks, qweight_columns = qweight.shape
    _, sub_blocks_per_super = sub_scales.shape
    super_block_size = int(sub_blocks_per_super) * SUB_BLOCK_SIZE
    expected_qweight_columns = (
        super_block_size
        if mode == "high"
        else int(sub_blocks_per_super) * INT6_PACKED_WORDS_PER_SUB_BLOCK
    )
    padded_numel = int(num_super_blocks) * super_block_size

    if qweight_columns != expected_qweight_columns:
        raise ValueError(
            "qweight block size does not match sub_scales: "
            f"expected {expected_qweight_columns}, got {qweight_columns}."
        )
    if original_numel > padded_numel:
        raise ValueError(
            "original_shape contains more values than the quantized blocks: "
            f"expected at most {padded_numel}, got {original_numel}."
        )
    if super_scales.numel() != num_super_blocks:
        raise ValueError("super_scales must contain one value per super-block.")

    if mode == "high":
        q = qweight.view(num_super_blocks, sub_blocks_per_super, SUB_BLOCK_SIZE).to(torch.float32)
    else:
        q = _unpack_int6_sub_blocks(qweight, num_super_blocks, sub_blocks_per_super).to(torch.float32)
    real_scales = sub_scales.to(torch.float32) * super_scales.to(torch.float32).view(-1, 1)

    dequantized = q * real_scales.unsqueeze(-1)
    flattened = dequantized.reshape(-1)[:original_numel].to(torch.float16)
    return flattened.view(original_shape)
