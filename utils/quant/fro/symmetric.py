import torch

HIGH_BLOCK_SIZE = 32
SUPER_BLOCK_SIZE = 256
HALF_SUPER_BLOCK_SIZE = 128
SUPPORTED_SUPER_BLOCK_SIZES = (SUPER_BLOCK_SIZE, HALF_SUPER_BLOCK_SIZE)
SUB_BLOCK_SIZE = 16


def _packed_words_per_sub_block(bits: int) -> int:
    return (SUB_BLOCK_SIZE * bits + 31) // 32


def _packed_words_for_values(value_count: int, bits: int) -> int:
    return (value_count * bits + 31) // 32


def _select_super_block_size(row_size: int) -> int:
    for super_block_size in SUPPORTED_SUPER_BLOCK_SIZES:
        if row_size % super_block_size == 0:
            return super_block_size
    raise ValueError(
        "Unsupported linear weight shape for symmetric super-block dequantization: "
        f"in_features={row_size}. Input features must be divisible by "
        f"{HALF_SUPER_BLOCK_SIZE} or {SUPER_BLOCK_SIZE}."
    )


def _unpack_signed_sub_blocks(
    qweight: torch.Tensor,
    num_super_blocks: int,
    sub_blocks_per_super: int,
    *,
    bits: int,
) -> torch.Tensor:
    packed_words_per_sub_block = _packed_words_per_sub_block(bits)
    packed = qweight.contiguous().view(num_super_blocks, sub_blocks_per_super, packed_words_per_sub_block)
    words = packed.to(torch.int64) & 0xFFFFFFFF
    unpacked = torch.empty(
        (num_super_blocks, sub_blocks_per_super, SUB_BLOCK_SIZE),
        dtype=torch.int8,
        device=qweight.device,
    )

    mask = (1 << bits) - 1
    sign_bit = 1 << (bits - 1)
    for value_index in range(SUB_BLOCK_SIZE):
        bit_offset = value_index * bits
        word_index = bit_offset // 32
        word_shift = bit_offset % 32

        code = words[..., word_index] >> word_shift
        if word_shift + bits > 32:
            code |= words[..., word_index + 1] << (32 - word_shift)
        code &= mask
        unpacked[..., value_index] = ((code ^ sign_bit) - sign_bit).to(torch.int8)

    return unpacked


def _unpack_unsigned_values(
    packed: torch.Tensor,
    value_count: int,
    *,
    bits: int,
) -> torch.Tensor:
    words = packed.to(torch.int64) & 0xFFFFFFFF
    unpacked = torch.empty((*packed.shape[:-1], value_count), dtype=torch.int8, device=packed.device)
    mask = (1 << bits) - 1

    for value_index in range(value_count):
        bit_offset = value_index * bits
        word_index = bit_offset // 32
        word_shift = bit_offset % 32

        code = words[..., word_index] >> word_shift
        if word_shift + bits > 32:
            code |= words[..., word_index + 1] << (32 - word_shift)
        unpacked[..., value_index] = (code & mask).to(torch.int8)

    return unpacked


def dequantize_from_symmetric(
    qweight: torch.Tensor,
    sub_scales: torch.Tensor,
    super_scales: torch.Tensor | None,
    original_shape: tuple[int, ...] | torch.Size,
    mode: str = "high",
) -> torch.Tensor:
    """
    Dequantize symmetric linear weights produced by quantize_to_symmetric.

    mode="high": int8 weights, 32-weight blocks, fp16 scales, no super scales.
    mode="med": packed signed int6 weights, 128/256-weight super-blocks.
    mode="low": packed signed int3 weights, 128/256-weight super-blocks.
    """
    mode = mode.strip().lower()
    original_shape = tuple(original_shape)
    if len(original_shape) != 2:
        raise ValueError(
            "Symmetric dequantization expects a 2D linear weight shape "
            f"(out_features, in_features), got {original_shape}."
        )

    row_count = int(original_shape[0])
    row_size = int(original_shape[1])
    original_numel = 1
    for dim in original_shape:
        original_numel *= int(dim)

    if original_numel == 0:
        return torch.empty(original_shape, dtype=torch.float16, device=qweight.device)

    if qweight.ndim != 2:
        raise ValueError("qweight must have shape [num_blocks, block_columns].")

    match mode:
        case "high":
            if qweight.dtype != torch.int8:
                raise TypeError("qweight must be int8 for high mode.")
            if not torch.is_floating_point(sub_scales):
                raise TypeError("sub_scales must be floating point for high mode.")
            if sub_scales.ndim != 1:
                raise ValueError("sub_scales must have shape [num_blocks] for high mode.")
            if super_scales is not None:
                raise ValueError("super_scales must be None for high mode.")

            quant_block_size = HIGH_BLOCK_SIZE
            blocks_per_row = (row_size + quant_block_size - 1) // quant_block_size
            padded_row_size = blocks_per_row * quant_block_size
            expected_num_blocks = row_count * blocks_per_row
            if qweight.shape != (expected_num_blocks, quant_block_size):
                raise ValueError(
                    "qweight shape does not match original_shape for high mode: "
                    f"expected {(expected_num_blocks, quant_block_size)}, got {tuple(qweight.shape)}."
                )
            if sub_scales.numel() != expected_num_blocks:
                raise ValueError("sub_scales must contain one scale per high-mode block.")

            dequantized_blocks = qweight.to(torch.float32) * sub_scales.to(torch.float32).view(-1, 1)

        case "med" | "low":
            if qweight.dtype != torch.int32:
                raise TypeError(f"qweight must be int32 for {mode} mode.")
            if super_scales is None or not torch.is_floating_point(super_scales):
                raise TypeError(f"super_scales must be floating point for {mode} mode.")

            weight_bits = 6 if mode == "med" else 3
            super_block_size = _select_super_block_size(row_size)
            sub_blocks_per_super = super_block_size // SUB_BLOCK_SIZE
            blocks_per_row = row_size // super_block_size
            padded_row_size = row_size
            expected_num_blocks = row_count * blocks_per_row
            expected_qweight_columns = sub_blocks_per_super * _packed_words_per_sub_block(weight_bits)
            expected_sub_scale_columns = (
                sub_blocks_per_super
                if mode == "med"
                else _packed_words_for_values(sub_blocks_per_super, bits=6)
            )
            expected_sub_scale_dtype = torch.int8 if mode == "med" else torch.int32

            if qweight.shape != (expected_num_blocks, expected_qweight_columns):
                raise ValueError(
                    f"qweight shape does not match original_shape for {mode} mode: "
                    f"expected {(expected_num_blocks, expected_qweight_columns)}, got {tuple(qweight.shape)}."
                )
            if sub_scales.dtype != expected_sub_scale_dtype:
                raise TypeError(f"sub_scales must be {expected_sub_scale_dtype} for {mode} mode.")
            if sub_scales.ndim != 2:
                raise ValueError(f"sub_scales must be 2D for {mode} mode.")
            if sub_scales.shape != (expected_num_blocks, expected_sub_scale_columns):
                raise ValueError(
                    f"sub_scales shape does not match original_shape for {mode} mode: "
                    f"expected {(expected_num_blocks, expected_sub_scale_columns)}, got {tuple(sub_scales.shape)}."
                )
            if super_scales.shape != (expected_num_blocks,):
                raise ValueError(
                    f"super_scales shape does not match original_shape for {mode} mode: "
                    f"expected {(expected_num_blocks,)}, got {tuple(super_scales.shape)}."
                )

            q = _unpack_signed_sub_blocks(
                qweight,
                expected_num_blocks,
                sub_blocks_per_super,
                bits=weight_bits,
            ).to(torch.float32)
            unpacked_sub_scales = (
                sub_scales
                if mode == "med"
                else _unpack_unsigned_values(sub_scales, sub_blocks_per_super, bits=6)
            )
            real_scales = unpacked_sub_scales.to(torch.float32) * super_scales.to(torch.float32).view(-1, 1)
            dequantized_blocks = (q * real_scales.unsqueeze(-1)).reshape(expected_num_blocks, super_block_size)

        case _:
            raise ValueError(
                f"Unsupported symmetric dequantization mode: {mode!r}. Expected 'high', 'med', or 'low'."
            )

    dequantized_rows = dequantized_blocks.view(row_count, padded_row_size)
    return dequantized_rows[:, :row_size].contiguous().to(torch.float16)
