import torch
import torch.nn.functional as F

SCALE_CODE_MIN = 0

SUPER_BLOCK_SIZE = 256
HALF_SUPER_BLOCK_SIZE = 128
SUPPORTED_SUPER_BLOCK_SIZES = (SUPER_BLOCK_SIZE, HALF_SUPER_BLOCK_SIZE)
_TARGET_FP32_CHUNK_BYTES = 256 * 1024 * 1024 * 2

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


def _pack_signed_values(values: torch.Tensor, *, bits: int) -> torch.Tensor:
    return _pack_unsigned_values(values, bits=bits)


def _select_super_block_size(row_size: int, sub_block_size: int) -> int:
    for super_block_size in SUPPORTED_SUPER_BLOCK_SIZES:
        if row_size % super_block_size == 0:
            return super_block_size
    raise ValueError(
        "Unsupported linear weight shape for affine quantization: "
        f"in_features={row_size}. Input features must be divisible by "
        f"{HALF_SUPER_BLOCK_SIZE} or {SUPER_BLOCK_SIZE}, and by sub-block size {sub_block_size}."
    )


def _quantize_super_blocks(
    super_blocks: torch.Tensor,
    *,
    sub_blocks_per_super: int,
    sub_block_size: int,
    weight_bits: int,
    meta_bits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sub_blocks = super_blocks.view(-1, sub_blocks_per_super, sub_block_size)

    sub_mins = sub_blocks.amin(dim=2)
    sub_maxes = sub_blocks.amax(dim=2)
    weight_code_max = (1 << weight_bits) - 1
    scale_code_max = (1 << meta_bits) - 1
    min_code_min = -(1 << (meta_bits - 1))
    min_code_max = (1 << (meta_bits - 1)) - 1
    sub_scales = (sub_maxes - sub_mins) / float(weight_code_max)

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
    quantized = quantized.clamp(0, weight_code_max).to(torch.uint8)
    packed = _pack_unsigned_values(quantized, bits=weight_bits)

    super_scales = sub_scales.abs().amax(dim=1) / float(scale_code_max)
    safe_super_scales = torch.where(super_scales > 0, super_scales, torch.ones_like(super_scales))
    sub_scale_codes = torch.round(sub_scales / safe_super_scales.unsqueeze(1))
    sub_scale_codes = torch.where(
        super_scales.unsqueeze(1) > 0,
        sub_scale_codes,
        torch.zeros_like(sub_scale_codes),
    )
    sub_scale_codes = sub_scale_codes.clamp(SCALE_CODE_MIN, scale_code_max).to(torch.int8)

    super_mins = sub_mins.abs().amax(dim=1) / float(min_code_max)
    safe_super_mins = torch.where(super_mins > 0, super_mins, torch.ones_like(super_mins))
    sub_min_codes = torch.round(sub_mins / safe_super_mins.unsqueeze(1))
    sub_min_codes = torch.where(
        super_mins.unsqueeze(1) > 0,
        sub_min_codes,
        torch.zeros_like(sub_min_codes),
    )
    sub_min_codes = sub_min_codes.clamp(min_code_min, min_code_max).to(torch.int8)

    return (
        packed,
        _pack_unsigned_values(sub_scale_codes, bits=meta_bits),
        _pack_signed_values(sub_min_codes, bits=meta_bits),
        super_scales.to(torch.float16),
        super_mins.to(torch.float16),
    )


def quantize_to_affine(
    tensor: torch.Tensor,
    mode: str = "low",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    mode="high": packed uint5 weights, 6-bit sub-scales/mins, 32-weight sub-blocks
    mode="med": packed uint4 weights, 6-bit sub-scales/mins, 32-weight sub-blocks
    mode="low": packed uint2 weights, 4-bit sub-scales/mins, 16-weight sub-blocks
    weights and sub-scales/mins are bit-packed into int32 words
    superscale: fp16 d + fp16 dmin
    blocks: 128/256-weight super-blocks
    """
    mode = mode.strip().lower()
    if mode not in AFFINE_MODES:
        raise ValueError(
            f"Unsupported affine quantization mode: {mode!r}. Expected 'high', 'med', or 'low'."
        )
    if not torch.is_floating_point(tensor):
        raise TypeError("Quantization expects fp16/fp32 tensors.")
    if tensor.ndim != 2:
        raise ValueError(
            "Affine quantization expects a 2D linear weight tensor "
            f"with shape (out_features, in_features), got {tuple(tensor.shape)}."
        )
    work_device = torch.device("cuda")

    config = AFFINE_MODES[mode]
    weight_bits = int(config["weight_bits"])
    meta_bits = int(config["meta_bits"])
    sub_block_size = int(config["sub_block_size"])
    row_size = int(tensor.shape[1])
    super_block_size = _select_super_block_size(row_size, sub_block_size)
    sub_blocks_per_super = super_block_size // sub_block_size
    qweight_columns = sub_blocks_per_super * _packed_words_for_values(sub_block_size, weight_bits)
    sub_metadata_columns = _packed_words_for_values(sub_blocks_per_super, meta_bits)
    qweight_dtype = torch.int32

    if tensor.numel() == 0:
        return (
            torch.empty((0, qweight_columns), dtype=qweight_dtype, device=tensor.device),
            torch.empty((0, sub_metadata_columns), dtype=torch.int32, device=tensor.device),
            torch.empty((0, sub_metadata_columns), dtype=torch.int32, device=tensor.device),
            torch.empty((0,), dtype=torch.float16, device=tensor.device),
            torch.empty((0,), dtype=torch.float16, device=tensor.device),
        )

    pad_len = (super_block_size - (row_size % super_block_size)) % super_block_size
    padded_row_size = row_size + pad_len
    blocks_per_row = padded_row_size // super_block_size
    num_super_blocks = int(tensor.shape[0]) * blocks_per_row
    packed_out = torch.empty((num_super_blocks, qweight_columns), dtype=qweight_dtype, device=tensor.device)
    sub_scale_codes_out = torch.empty(
        (num_super_blocks, sub_metadata_columns),
        dtype=torch.int32,
        device=tensor.device,
    )
    sub_min_codes_out = torch.empty_like(sub_scale_codes_out)
    super_scales_out = torch.empty((num_super_blocks,), dtype=torch.float16, device=tensor.device)
    super_mins_out = torch.empty_like(super_scales_out)
    rows_per_chunk = max(1, _TARGET_FP32_CHUNK_BYTES // (padded_row_size * 4))

    for start in range(0, int(tensor.shape[0]), rows_per_chunk):
        stop = min(int(tensor.shape[0]), start + rows_per_chunk)
        out_start = start * blocks_per_row
        out_stop = stop * blocks_per_row
        arr32 = tensor[start:stop].detach().to(device=work_device, dtype=torch.float32)
        if pad_len:
            arr32 = F.pad(arr32, (0, pad_len))
        super_blocks = arr32.contiguous().view(-1, super_block_size)
        (
            packed_chunk,
            sub_scale_codes_chunk,
            sub_min_codes_chunk,
            super_scales_chunk,
            super_mins_chunk,
        ) = _quantize_super_blocks(
            super_blocks,
            sub_blocks_per_super=sub_blocks_per_super,
            sub_block_size=sub_block_size,
            weight_bits=weight_bits,
            meta_bits=meta_bits,
        )
        packed_out[out_start:out_stop].copy_(packed_chunk)
        sub_scale_codes_out[out_start:out_stop].copy_(sub_scale_codes_chunk)
        sub_min_codes_out[out_start:out_stop].copy_(sub_min_codes_chunk)
        super_scales_out[out_start:out_stop].copy_(super_scales_chunk)
        super_mins_out[out_start:out_stop].copy_(super_mins_chunk)

    return (
        packed_out,
        sub_scale_codes_out,
        sub_min_codes_out,
        super_scales_out,
        super_mins_out,
    )
