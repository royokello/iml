import torch
import torch.nn.functional as F

INT6_MIN = -32
INT6_MAX = 31
INT3_MIN = -4
INT3_MAX = 3
INT8_MIN = -128
INT8_MAX = 127
SCALE_CODE_MIN = 0
INT8_SCALE_CODE_MAX = 127
INT6_SCALE_CODE_MAX = 63

HIGH_BLOCK_SIZE = 32
SUPER_BLOCK_SIZE = 256
HALF_SUPER_BLOCK_SIZE = 128
SUPPORTED_SUPER_BLOCK_SIZES = (SUPER_BLOCK_SIZE, HALF_SUPER_BLOCK_SIZE)
SUB_BLOCK_SIZE = 16
_TARGET_FP32_CHUNK_BYTES = 256 * 1024 * 1024 * 2


def _packed_words_per_sub_block(bits: int) -> int:
    return (SUB_BLOCK_SIZE * bits + 31) // 32


def _packed_words_for_values(value_count: int, bits: int) -> int:
    return (value_count * bits + 31) // 32


def _pack_unsigned_values(values: torch.Tensor, *, bits: int) -> torch.Tensor:
    mask = (1 << bits) - 1
    codes = values.to(torch.int64) & mask
    packed = torch.zeros(
        (*codes.shape[:-1], _packed_words_for_values(int(codes.shape[-1]), bits)),
        dtype=torch.int64,
        device=values.device,
    )

    bit_offset = 0
    for value_index in range(int(codes.shape[-1])):
        word_index = bit_offset // 32
        word_shift = bit_offset % 32
        value = codes[..., value_index]

        packed[..., word_index] |= value << word_shift
        if word_shift + bits > 32:
            packed[..., word_index + 1] |= value >> (32 - word_shift)

        bit_offset += bits

    signed_packed = torch.where(packed >= (1 << 31), packed - (1 << 32), packed)
    return signed_packed.to(torch.int32).view(values.shape[0], -1)


def _pack_signed_sub_blocks(qweight: torch.Tensor, *, bits: int) -> torch.Tensor:
    if qweight.shape[-1] != SUB_BLOCK_SIZE:
        raise ValueError(f"{bits}-bit packing expects sub-blocks of {SUB_BLOCK_SIZE} values.")

    mask = (1 << bits) - 1
    codes = qweight.to(torch.int64) & mask
    packed = torch.zeros(
        (*codes.shape[:-1], _packed_words_per_sub_block(bits)),
        dtype=torch.int64,
        device=qweight.device,
    )

    bit_offset = 0
    for value_index in range(SUB_BLOCK_SIZE):
        word_index = bit_offset // 32
        word_shift = bit_offset % 32
        value = codes[..., value_index]

        packed[..., word_index] |= value << word_shift
        if word_shift + bits > 32:
            packed[..., word_index + 1] |= value >> (32 - word_shift)

        bit_offset += bits

    signed_packed = torch.where(packed >= (1 << 31), packed - (1 << 32), packed)
    return signed_packed.to(torch.int32).view(qweight.shape[0], -1)


def _quantize_high_blocks(blocks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    absmax = blocks.abs().amax(dim=1)
    scales = absmax / float(INT8_MAX)
    safe_scales = torch.where(scales > 0, scales, torch.ones_like(scales))

    qweight = torch.round(blocks / safe_scales.unsqueeze(1))
    qweight = torch.where(scales.unsqueeze(1) > 0, qweight, torch.zeros_like(qweight))
    qweight = qweight.clamp(INT8_MIN, INT8_MAX).to(torch.int8)

    return qweight, scales.to(torch.float16)


def _select_super_block_size(row_size: int) -> int:
    for super_block_size in SUPPORTED_SUPER_BLOCK_SIZES:
        if row_size % super_block_size == 0:
            return super_block_size
    raise ValueError(
        "Unsupported linear weight shape for symmetric super-block quantization: "
        f"in_features={row_size}. Input features must be divisible by "
        f"{HALF_SUPER_BLOCK_SIZE} or {SUPER_BLOCK_SIZE}."
    )


def _quantize_super_blocks(
    super_blocks: torch.Tensor,
    *,
    sub_blocks_per_super: int,
    weight_bits: int,
    weight_min: int,
    weight_max: int,
    scale_code_max: int,
    pack_sub_scales: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sub_blocks = super_blocks.view(-1, sub_blocks_per_super, SUB_BLOCK_SIZE)

    absmax = sub_blocks.abs().amax(dim=2)
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
    qweight_out = _pack_signed_sub_blocks(qweight, bits=weight_bits)

    super_scales = sub_scale_values.abs().amax(dim=1) / float(scale_code_max)
    safe_super_scales = torch.where(super_scales > 0, super_scales, torch.ones_like(super_scales))

    sub_scales = torch.round(sub_scale_values / safe_super_scales.unsqueeze(1))
    sub_scales = torch.where(
        super_scales.unsqueeze(1) > 0,
        sub_scales,
        torch.zeros_like(sub_scales),
    )
    sub_scales = sub_scales.clamp(SCALE_CODE_MIN, scale_code_max).to(torch.int8)
    sub_scales_out = _pack_unsigned_values(sub_scales, bits=6) if pack_sub_scales else sub_scales

    return qweight_out, sub_scales_out, super_scales.to(torch.float16)


def quantize_to_symmetric(
    tensor: torch.Tensor,
    mode: str = "high",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    mode="high": int8 weights, 32-weight blocks, fp16 scale, no super scale
    mode="med": packed signed int6 weights, 16-weight sub-blocks, int8 sub-scales,
        8 or 16 sub-block super-blocks, fp16 super-scales
    mode="low": packed signed int3 weights, 16-weight sub-blocks, 6-bit sub-scale codes,
        8 or 16 sub-block super-blocks, fp16 super-scales
    """
    mode = mode.strip().lower()
    if not torch.is_floating_point(tensor):
        raise TypeError("Quantization expects fp16/fp32 tensors.")
    if tensor.ndim != 2:
        raise ValueError(
            "Symmetric quantization expects a 2D linear weight tensor "
            f"with shape (out_features, in_features), got {tuple(tensor.shape)}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Symmetric quantization requires CUDA to process weight chunks.")

    row_size = int(tensor.shape[1])
    match mode:
        case "high":
            sub_block_size = HIGH_BLOCK_SIZE
            quant_block_size = sub_block_size
            qweight_columns = sub_block_size
            qweight_dtype = torch.int8
            sub_scales_shape = lambda block_count: (block_count,)
            sub_scales_dtype = torch.float16
            super_scales_shape = None
            weight_bits = None
            weight_min = None
            weight_max = None
            scale_code_max = None

        case "med":
            weight_bits = 6
            weight_min = INT6_MIN
            weight_max = INT6_MAX
            scale_code_max = INT8_SCALE_CODE_MAX
            pack_sub_scales = False
            super_block_size = _select_super_block_size(row_size)
            sub_blocks_per_super = super_block_size // SUB_BLOCK_SIZE
            quant_block_size = super_block_size
            qweight_columns = sub_blocks_per_super * _packed_words_per_sub_block(weight_bits)
            qweight_dtype = torch.int32
            sub_scales_shape = lambda block_count: (block_count, sub_blocks_per_super)
            sub_scales_dtype = torch.int8
            super_scales_shape = lambda block_count: (block_count,)

        case "low":
            weight_bits = 3
            weight_min = INT3_MIN
            weight_max = INT3_MAX
            scale_code_max = INT6_SCALE_CODE_MAX
            pack_sub_scales = True
            super_block_size = _select_super_block_size(row_size)
            sub_blocks_per_super = super_block_size // SUB_BLOCK_SIZE
            quant_block_size = super_block_size
            qweight_columns = sub_blocks_per_super * _packed_words_per_sub_block(weight_bits)
            qweight_dtype = torch.int32
            packed_scale_words_per_super = _packed_words_for_values(sub_blocks_per_super, bits=6)
            sub_scales_shape = lambda block_count: (block_count, packed_scale_words_per_super)
            sub_scales_dtype = torch.int32
            super_scales_shape = lambda block_count: (block_count,)

        case _:
            raise ValueError(
                f"Unsupported symmetric quantization mode: {mode!r}. Expected 'high', 'med', or 'low'."
            )

    if tensor.numel() == 0:
        return (
            torch.empty((0, qweight_columns), dtype=qweight_dtype, device=tensor.device),
            torch.empty(sub_scales_shape(0), dtype=sub_scales_dtype, device=tensor.device),
            (
                None
                if super_scales_shape is None
                else torch.empty(super_scales_shape(0), dtype=torch.float16, device=tensor.device)
            ),
        )

    pad_len = (quant_block_size - (row_size % quant_block_size)) % quant_block_size
    padded_row_size = row_size + pad_len
    blocks_per_row = padded_row_size // quant_block_size
    num_blocks = int(tensor.shape[0]) * blocks_per_row
    qweight_out = torch.empty(
        (num_blocks, qweight_columns),
        dtype=qweight_dtype,
        device=tensor.device,
    )
    sub_scales_out = torch.empty(
        sub_scales_shape(num_blocks),
        dtype=sub_scales_dtype,
        device=tensor.device,
    )
    super_scales_out = (
        None
        if super_scales_shape is None
        else torch.empty(super_scales_shape(num_blocks), dtype=torch.float16, device=tensor.device)
    )
    rows_per_chunk = max(1, _TARGET_FP32_CHUNK_BYTES // (padded_row_size * 4))

    for start in range(0, int(tensor.shape[0]), rows_per_chunk):
        stop = min(int(tensor.shape[0]), start + rows_per_chunk)
        out_start = start * blocks_per_row
        out_stop = stop * blocks_per_row
        arr32 = tensor[start:stop].detach().to(device="cuda", dtype=torch.float32)
        if pad_len:
            arr32 = F.pad(arr32, (0, pad_len))
        blocks = arr32.contiguous().view(-1, quant_block_size)
        if super_scales_shape is None:
            qweight_chunk, sub_scale_chunk = _quantize_high_blocks(blocks)
        else:
            qweight_chunk, sub_scale_chunk, super_scale_chunk = _quantize_super_blocks(
                blocks,
                sub_blocks_per_super=sub_blocks_per_super,
                weight_bits=weight_bits,
                weight_min=weight_min,
                weight_max=weight_max,
                scale_code_max=scale_code_max,
                pack_sub_scales=pack_sub_scales,
            )
            assert super_scales_out is not None
            super_scales_out[out_start:out_stop].copy_(super_scale_chunk)
        qweight_out[out_start:out_stop].copy_(qweight_chunk)
        sub_scales_out[out_start:out_stop].copy_(sub_scale_chunk)

    return (
        qweight_out,
        sub_scales_out,
        super_scales_out,
    )
