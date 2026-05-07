from __future__ import annotations

import json
import math
from typing import Any

from utils.quant.validators import quant_method_family, quant_method_mode

_SAFETENSORS_HEADER_LEN_BYTES = 8

_DTYPE_BITS = {
    "I8": 8,
    "F16": 16,
    "BF16": 16,
    "I32": 32,
    "F32": 32,
}
_FLOAT_DTYPES = {"F16", "BF16", "F32"}
_FP16_CAST_DTYPES = {"F32", "BF16"}

_SYMMETRIC_HIGH_BLOCK_SIZE = 32
_SYMMETRIC_SUB_BLOCK_SIZE = 16
_SUPER_BLOCK_SIZE = 256
_HALF_SUPER_BLOCK_SIZE = 128
_SUPPORTED_SUPER_BLOCK_SIZES = (_SUPER_BLOCK_SIZE, _HALF_SUPER_BLOCK_SIZE)

_AFFINE_MODES = {
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


def _dtype_bits(dtype: str) -> int:
    try:
        return _DTYPE_BITS[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported safetensors dtype: {dtype!r}") from exc


def _dtype_nbytes(dtype: str, numel: int) -> int:
    return math.ceil(numel * _dtype_bits(dtype) / 8)


def _numel(shape: tuple[int, ...]) -> int:
    return math.prod(shape)


def _packed_words_for_values(value_count: int, bits: int) -> int:
    return (value_count * bits + 31) // 32


def _select_super_block_size(row_size: int, *, family: str, sub_block_size: int | None = None) -> int:
    for super_block_size in _SUPPORTED_SUPER_BLOCK_SIZES:
        if row_size % super_block_size == 0:
            return super_block_size

    if family == "symmetric":
        raise ValueError(
            "Unsupported linear weight shape for symmetric super-block quantization: "
            f"in_features={row_size}. Input features must be divisible by "
            f"{_HALF_SUPER_BLOCK_SIZE} or {_SUPER_BLOCK_SIZE}."
        )

    raise ValueError(
        "Unsupported linear weight shape for affine quantization: "
        f"in_features={row_size}. Input features must be divisible by "
        f"{_HALF_SUPER_BLOCK_SIZE} or {_SUPER_BLOCK_SIZE}, and by sub-block size "
        f"{sub_block_size}."
    )


def _validate_quantized_tensor(name: str, dtype: str, shape: tuple[int, ...], family: str) -> None:
    if dtype not in _FLOAT_DTYPES:
        raise TypeError(f"Quantization expects floating point tensors: {name} has dtype {dtype}.")
    if len(shape) != 2:
        raise ValueError(
            f"{family.title()} quantization expects a 2D linear weight tensor: "
            f"{name} has shape {shape}."
        )


def _estimate_symmetric_tensors(
    name: str,
    dtype: str,
    shape: tuple[int, ...],
    mode: str,
) -> list[tuple[str, str, tuple[int, ...]]]:
    _validate_quantized_tensor(name, dtype, shape, "symmetric")
    out_features, row_size = shape
    metadata_base_name = name.removesuffix(".weight")

    if mode == "high":
        padded_row_size = row_size + (
            _SYMMETRIC_HIGH_BLOCK_SIZE - row_size % _SYMMETRIC_HIGH_BLOCK_SIZE
        ) % _SYMMETRIC_HIGH_BLOCK_SIZE
        block_count = out_features * (padded_row_size // _SYMMETRIC_HIGH_BLOCK_SIZE)
        return [
            (name, "I8", (block_count, _SYMMETRIC_HIGH_BLOCK_SIZE)),
            (f"{metadata_base_name}.sub_scales", "F16", (block_count,)),
        ]

    if mode == "med":
        weight_bits = 6
        super_block_size = _select_super_block_size(row_size, family="symmetric")
        sub_blocks_per_super = super_block_size // _SYMMETRIC_SUB_BLOCK_SIZE
        qweight_columns = sub_blocks_per_super * _packed_words_for_values(
            _SYMMETRIC_SUB_BLOCK_SIZE,
            weight_bits,
        )
        block_count = out_features * (row_size // super_block_size)
        return [
            (name, "I32", (block_count, qweight_columns)),
            (f"{metadata_base_name}.sub_scales", "I8", (block_count, sub_blocks_per_super)),
            (f"{metadata_base_name}.super_scales", "F16", (block_count,)),
        ]

    if mode == "low":
        weight_bits = 3
        super_block_size = _select_super_block_size(row_size, family="symmetric")
        sub_blocks_per_super = super_block_size // _SYMMETRIC_SUB_BLOCK_SIZE
        qweight_columns = sub_blocks_per_super * _packed_words_for_values(
            _SYMMETRIC_SUB_BLOCK_SIZE,
            weight_bits,
        )
        packed_scale_columns = _packed_words_for_values(sub_blocks_per_super, bits=6)
        block_count = out_features * (row_size // super_block_size)
        return [
            (name, "I32", (block_count, qweight_columns)),
            (f"{metadata_base_name}.sub_scales", "I32", (block_count, packed_scale_columns)),
            (f"{metadata_base_name}.super_scales", "F16", (block_count,)),
        ]

    raise ValueError(f"Unsupported symmetric quantization mode: {mode!r}.")


def _estimate_affine_tensors(
    name: str,
    dtype: str,
    shape: tuple[int, ...],
    mode: str,
) -> list[tuple[str, str, tuple[int, ...]]]:
    _validate_quantized_tensor(name, dtype, shape, "affine")
    try:
        config = _AFFINE_MODES[mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported affine quantization mode: {mode!r}.") from exc

    out_features, row_size = shape
    metadata_base_name = name.removesuffix(".weight")
    weight_bits = int(config["weight_bits"])
    meta_bits = int(config["meta_bits"])
    sub_block_size = int(config["sub_block_size"])
    super_block_size = _select_super_block_size(
        row_size,
        family="affine",
        sub_block_size=sub_block_size,
    )
    sub_blocks_per_super = super_block_size // sub_block_size
    qweight_columns = sub_blocks_per_super * _packed_words_for_values(sub_block_size, weight_bits)
    metadata_columns = _packed_words_for_values(sub_blocks_per_super, meta_bits)
    block_count = out_features * (row_size // super_block_size)

    return [
        (name, "I32", (block_count, qweight_columns)),
        (f"{metadata_base_name}.sub_scales", "I32", (block_count, metadata_columns)),
        (f"{metadata_base_name}.sub_mins", "I32", (block_count, metadata_columns)),
        (f"{metadata_base_name}.super_scales", "F16", (block_count,)),
        (f"{metadata_base_name}.super_mins", "F16", (block_count,)),
    ]


def _estimate_output_tensors(
    name: str,
    dtype: str,
    shape: tuple[int, ...],
    method: str | None,
) -> list[tuple[str, str, tuple[int, ...]]]:
    if method is None:
        output_dtype = "F16" if dtype in _FP16_CAST_DTYPES else dtype
        return [(name, output_dtype, shape)]

    family = quant_method_family(method)
    mode = quant_method_mode(method)
    if family == "symmetric":
        return _estimate_symmetric_tensors(name, dtype, shape, mode)
    return _estimate_affine_tensors(name, dtype, shape, mode)


def _tensor_nbytes(tensor: tuple[str, str, tuple[int, ...]]) -> int:
    _, dtype, shape = tensor
    return _dtype_nbytes(dtype, _numel(shape))


def _estimate_safetensors_header_size(tensors: list[tuple[str, str, tuple[int, ...]]]) -> int:
    data_offset = 0
    header: dict[str, Any] = {}
    for name, dtype, shape in tensors:
        data_end = data_offset + _tensor_nbytes((name, dtype, shape))
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [data_offset, data_end],
        }
        data_offset = data_end

    encoded_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return _SAFETENSORS_HEADER_LEN_BYTES + len(encoded_header)


def estimate_quantized_safetensors_size(
    tensors: list[tuple[str, str, tuple[int, ...]]],
    methods_by_name: dict[str, str],
    inclusion_prefix: str | tuple[str, ...] | None,
    exclusion_prefix: str | tuple[str, ...] | None,
) -> int:
    output_tensors: list[tuple[str, str, tuple[int, ...]]] = []
    for name, dtype, shape in tensors:
        if inclusion_prefix is not None and not name.startswith(inclusion_prefix):
            continue
        if exclusion_prefix is not None and name.startswith(exclusion_prefix):
            continue
        output_tensors.extend(
            _estimate_output_tensors(
                name,
                dtype,
                shape,
                methods_by_name.get(name),
            )
        )

    return _estimate_safetensors_header_size(output_tensors) + sum(
        _tensor_nbytes(tensor) for tensor in output_tensors
    )


__all__ = ["estimate_quantized_safetensors_size"]
