import torch

SUPER_BLOCK_SIZE = 256
HALF_SUPER_BLOCK_SIZE = 128
SUPPORTED_SUPER_BLOCK_SIZES = (SUPER_BLOCK_SIZE, HALF_SUPER_BLOCK_SIZE)

AFFINE_MODES = {
    "high": {
        "weight_bits": 5,
        "meta_bits": 6,
        "sub_block_size": 32,
    },
    "med": {
        "weight_bits": 4,
        "meta_bits": 6,
        "sub_block_size": 32,
    },
    "low": {
        "weight_bits": 2,
        "meta_bits": 4,
        "sub_block_size": 16,
    },
}


def _packed_words_for_values(value_count: int, bits: int) -> int:
    return (value_count * bits + 31) // 32


def _select_super_block_size(row_size: int, sub_block_size: int) -> int:
    for super_block_size in SUPPORTED_SUPER_BLOCK_SIZES:
        if row_size % super_block_size == 0:
            return super_block_size
    raise ValueError(
        "Unsupported linear weight shape for affine dequantization: "
        f"in_features={row_size}. Input features must be divisible by "
        f"{HALF_SUPER_BLOCK_SIZE} or {SUPER_BLOCK_SIZE}, and by sub-block size {sub_block_size}."
    )


def _unpack_unsigned_values(
    packed: torch.Tensor,
    value_count: int,
    *,
    bits: int,
) -> torch.Tensor:
    words = packed.to(torch.int64) & 0xFFFFFFFF
    unpacked = torch.empty((*packed.shape[:-1], value_count), dtype=torch.uint8, device=packed.device)
    mask = (1 << bits) - 1

    for value_index in range(value_count):
        bit_offset = value_index * bits
        word_index = bit_offset // 32
        word_shift = bit_offset % 32

        code = words[..., word_index] >> word_shift
        if word_shift + bits > 32:
            code |= words[..., word_index + 1] << (32 - word_shift)
        unpacked[..., value_index] = (code & mask).to(torch.uint8)

    return unpacked


def _unpack_signed_values(
    packed: torch.Tensor,
    value_count: int,
    *,
    bits: int,
) -> torch.Tensor:
    words = packed.to(torch.int64) & 0xFFFFFFFF
    unpacked = torch.empty((*packed.shape[:-1], value_count), dtype=torch.int8, device=packed.device)
    mask = (1 << bits) - 1
    sign_bit = 1 << (bits - 1)

    for value_index in range(value_count):
        bit_offset = value_index * bits
        word_index = bit_offset // 32
        word_shift = bit_offset % 32

        code = words[..., word_index] >> word_shift
        if word_shift + bits > 32:
            code |= words[..., word_index + 1] << (32 - word_shift)
        code &= mask
        unpacked[..., value_index] = ((code ^ sign_bit) - sign_bit).to(torch.int8)

    return unpacked


def dequantize_from_affine(
    qweight: torch.Tensor,
    sub_scales: torch.Tensor,
    sub_mins: torch.Tensor,
    super_scales: torch.Tensor,
    super_mins: torch.Tensor,
    original_shape: tuple[int, ...] | torch.Size,
    mode: str = "low",
    bsums: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Dequantize affine linear weights produced by quantize_to_affine.

    mode="high": packed uint5 weights, 6-bit sub-scales/mins, 32-weight sub-blocks.
    mode="med": packed uint4 weights, 6-bit sub-scales/mins, 32-weight sub-blocks.
    mode="low": packed uint2 weights, 4-bit sub-scales/mins, 16-weight sub-blocks.
    weights and sub-scales/mins are bit-packed into int32 words.
    super scales/mins: fp16 d + fp16 dmin.
    blocks: 128/256-weight super-blocks.
    """
    mode = mode.strip().lower()
    if mode not in AFFINE_MODES:
        raise ValueError(
            f"Unsupported affine dequantization mode: {mode!r}. Expected 'high', 'med', or 'low'."
        )
    if qweight.dtype != torch.int32:
        raise TypeError("qweight must be int32.")
    if sub_scales.dtype != torch.int32:
        raise TypeError("sub_scales must be int32.")
    if sub_mins.dtype != torch.int32:
        raise TypeError("sub_mins must be int32.")
    if not torch.is_floating_point(super_scales):
        raise TypeError("super_scales must be floating point.")
    if not torch.is_floating_point(super_mins):
        raise TypeError("super_mins must be floating point.")

    original_shape = tuple(original_shape)
    if len(original_shape) != 2:
        raise ValueError(
            "Affine dequantization expects a 2D linear weight shape "
            f"(out_features, in_features), got {original_shape}."
        )

    row_count = int(original_shape[0])
    row_size = int(original_shape[1])
    original_numel = row_count * row_size

    config = AFFINE_MODES[mode]
    weight_bits = int(config["weight_bits"])
    meta_bits = int(config["meta_bits"])
    sub_block_size = int(config["sub_block_size"])
    super_block_size = _select_super_block_size(row_size, sub_block_size)
    sub_blocks_per_super = super_block_size // sub_block_size
    blocks_per_row = row_size // super_block_size
    expected_num_blocks = row_count * blocks_per_row
    packed_words_per_sub_block = _packed_words_for_values(sub_block_size, weight_bits)
    expected_qweight_columns = sub_blocks_per_super * packed_words_per_sub_block
    expected_sub_metadata_columns = _packed_words_for_values(sub_blocks_per_super, meta_bits)

    if original_numel == 0:
        return torch.empty(original_shape, dtype=torch.float16, device=qweight.device)

    if qweight.shape != (expected_num_blocks, expected_qweight_columns):
        raise ValueError(
            "qweight shape does not match original_shape: "
            f"expected {(expected_num_blocks, expected_qweight_columns)}, got {tuple(qweight.shape)}."
        )
    if sub_scales.shape != (expected_num_blocks, expected_sub_metadata_columns):
        raise ValueError(
            "sub_scales shape does not match original_shape: "
            f"expected {(expected_num_blocks, expected_sub_metadata_columns)}, got {tuple(sub_scales.shape)}."
        )
    if sub_mins.shape != sub_scales.shape:
        raise ValueError("sub_mins must have the same shape as sub_scales.")
    if super_scales.shape != (expected_num_blocks,):
        raise ValueError(
            "super_scales shape does not match original_shape: "
            f"expected {(expected_num_blocks,)}, got {tuple(super_scales.shape)}."
        )
    if super_mins.shape != (expected_num_blocks,):
        raise ValueError(
            "super_mins shape does not match original_shape: "
            f"expected {(expected_num_blocks,)}, got {tuple(super_mins.shape)}."
        )

    packed = qweight.contiguous().view(expected_num_blocks, sub_blocks_per_super, packed_words_per_sub_block)
    q = _unpack_unsigned_values(packed, sub_block_size, bits=weight_bits).to(torch.float32)
    unpacked_sub_scales = _unpack_unsigned_values(
        sub_scales.contiguous(),
        sub_blocks_per_super,
        bits=meta_bits,
    )
    unpacked_sub_mins = _unpack_signed_values(
        sub_mins.contiguous(),
        sub_blocks_per_super,
        bits=meta_bits,
    )
    real_scales = unpacked_sub_scales.to(torch.float32) * super_scales.to(torch.float32).view(-1, 1)
    real_mins = unpacked_sub_mins.to(torch.float32) * super_mins.to(torch.float32).view(-1, 1)

    dequantized_blocks = q * real_scales.unsqueeze(-1) + real_mins.unsqueeze(-1)
    dequantized_rows = dequantized_blocks.reshape(expected_num_blocks, super_block_size)

    if bsums is not None:
        # bsums: [batch, sub_blocks_total] — one bsum per sub-block position
        # real_mins: [expected_num_blocks, sub_blocks_per_super]
        # For each output i, for each super-block sb, for each sub:
        #   correction += bsums[sb * sub_blocks_per_super + sub] * real_mins[sb + i * blocks_per_row, sub]
        # Reshape to: correction = bsums @ real_mins_mtx
        # real_mins_mtx: [sub_blocks_total, out_features]
        sub_blocks_total = expected_num_blocks * sub_blocks_per_super
        real_mins_flat = real_mins.reshape(expected_num_blocks, sub_blocks_per_super)
        # Reshape to [out_features, blocks_per_row, sub_blocks_per_super]:
        out_features = row_count
        rm = real_mins_flat.view(out_features, -1, sub_blocks_per_super).transpose(0, 1).contiguous()
        # rm: [blocks_per_row, out_features, sub_blocks_per_super]
        mins_matrix = rm.reshape(-1, out_features)  # [sub_blocks_total, out_features]
        correction = bsums.to(torch.float32) @ mins_matrix.to(torch.float32).T  # [batch, out_features]
        return (
            dequantized_rows.view(row_count, row_size).contiguous().to(torch.float16),
            correction.to(torch.float16),
        )

    return dequantized_rows.view(row_count, row_size).contiguous().to(torch.float16)
