#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F

from flux2.quant.denoiser import _build_target_tensors
from utils.loaders.single import safe_load_file as load_safetensors_file

BLOCK_SIZES = (16, 32, 64, 128, 256)
F1_SCALE_DTYPES = ("fp16", "fp32")
DOUBLE_INNER_BLOCK_SIZES = (16, 32, 64)
DOUBLE_OUTER_BLOCK_SIZES = (64, 128, 256)
F2_SUB_SCALE_DTYPES = ("int8", "fp16")
F2_SUPER_SCALE_DTYPES = ("fp16", "fp32")
INT8_MIN = -127
INT8_MAX = 127
INT4_MIN = -7
INT4_MAX = 7
ZERO_SCALE = 1.0

TEXT_ENCODER_LINEAR_NAMES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "lm_head",
)

DENOISER_LINEAR_NAMES = tuple(tensor.removesuffix(".weight") for tensor in _build_target_tensors("4b"))
INCLUDED_LINEAR_NAMES = DENOISER_LINEAR_NAMES + TEXT_ENCODER_LINEAR_NAMES


@dataclass(frozen=True)
class CandidateResult:
    name: str
    format_family: str
    total_bytes: int
    relative_rmse: float


def log_step(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate single-block int8 + fp16-scale quantization for linear safetensors weights."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="A .safetensors file or a folder containing .safetensors files.",
    )
    return parser.parse_args()


def resolve_input_paths(input_path: str | Path) -> list[Path]:
    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    if path.is_file():
        if path.suffix.lower() != ".safetensors":
            raise ValueError(f"Expected a .safetensors file: {path}")
        return [path]

    if not path.is_dir():
        raise ValueError(f"Input must be a file or folder: {path}")

    return sorted(
        (
            candidate
            for candidate in path.rglob("*")
            if candidate.is_file() and candidate.suffix.lower() == ".safetensors"
        ),
        key=lambda candidate: str(candidate.relative_to(path)),
    )


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    log_step(f"loading: {path}")
    return load_safetensors_file(str(path))


def iter_tensors(paths: list[Path]) -> Iterator[tuple[Path, str, torch.Tensor]]:
    for path in paths:
        state_dict = load_state_dict(path)
        for name in sorted(state_dict):
            yield path, name, state_dict[name]
        del state_dict


def is_linear_weight_tensor(name: str, tensor: torch.Tensor) -> bool:
    if not name.endswith(".weight"):
        return False
    if tensor.ndim != 2:
        return False

    module_path = tuple(part for part in name.split(".")[:-1] if part)
    if not module_path:
        return False

    full_name = ".".join(module_path)
    return (
        full_name in INCLUDED_LINEAR_NAMES
        or module_path[-1] in INCLUDED_LINEAR_NAMES
        or (module_path[-1].isdigit() and len(module_path) > 1 and module_path[-2] in INCLUDED_LINEAR_NAMES)
    )


def quantize_format1(
    tensor: torch.Tensor,
    block_size: int,
    scale_dtype: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not torch.is_floating_point(tensor):
        raise TypeError("Format 1 quantization expects a floating-point tensor.")

    tensor_fp16 = tensor.detach().to(dtype=torch.float16, device="cpu")
    original_shape = tensor_fp16.shape
    flattened = tensor_fp16.reshape(-1)

    if flattened.numel() == 0:
        return (
            torch.empty(original_shape, dtype=torch.int8),
            torch.empty((0,), dtype=torch.float16 if scale_dtype == "fp16" else torch.float32),
        )

    pad_len = (block_size - (flattened.numel() % block_size)) % block_size
    if pad_len:
        flattened = F.pad(flattened, (0, pad_len))

    blocks = flattened.view(-1, block_size)
    scales = blocks.abs().amax(dim=1, keepdim=True) / float(INT8_MAX)
    scales = torch.where(scales == 0, torch.full_like(scales, ZERO_SCALE), scales)

    quantized = torch.round(blocks / scales).clamp(INT8_MIN, INT8_MAX).to(torch.int8)
    quantized_flat = quantized.reshape(-1)
    if pad_len:
        quantized_flat = quantized_flat[:-pad_len]

    if scale_dtype == "fp16":
        stored_scales = scales.reshape(-1).to(torch.float16)
    elif scale_dtype == "fp32":
        stored_scales = scales.reshape(-1).to(torch.float32)
    else:
        raise ValueError(f"Unsupported f1 scale dtype: {scale_dtype}")

    return quantized_flat.view(original_shape), stored_scales


def dequantize_format1(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    if quantized.dtype != torch.int8:
        raise TypeError("Format 1 dequantization expects an int8 tensor.")

    original_shape = quantized.shape
    flattened = quantized.reshape(-1).to(dtype=torch.float16, device="cpu")

    if flattened.numel() == 0:
        return torch.empty(original_shape, dtype=torch.float16)

    pad_len = (block_size - (flattened.numel() % block_size)) % block_size
    if pad_len:
        flattened = F.pad(flattened, (0, pad_len))

    blocks = flattened.view(-1, block_size)
    dequantized = blocks * scales.to(dtype=torch.float16, device="cpu").view(-1, 1)
    dequantized_flat = dequantized.reshape(-1)
    if pad_len:
        dequantized_flat = dequantized_flat[:-pad_len]

    return dequantized_flat.view(original_shape)


def total_bytes_format1(numel: int, block_size: int) -> int:
    return numel + 2 * ((numel + block_size - 1) // block_size)


def total_bytes_format1_with_scale_dtype(numel: int, block_size: int, scale_dtype: str) -> int:
    scale_count = (numel + block_size - 1) // block_size
    return numel + scale_count * scale_dtype_bytes(scale_dtype)


def scale_dtype_bytes(scale_dtype: str) -> int:
    if scale_dtype == "int8":
        return 1
    if scale_dtype == "fp16":
        return 2
    if scale_dtype == "fp32":
        return 4
    raise ValueError(f"Unsupported scale dtype: {scale_dtype}")


def total_bytes_format2(
    numel: int,
    inner_block_size: int,
    outer_block_size: int,
    sub_scale_dtype: str,
    super_scale_dtype: str,
) -> int:
    weight_bytes = (numel + 1) // 2
    inner_scale_count = (numel + inner_block_size - 1) // inner_block_size
    outer_scale_count = (numel + outer_block_size - 1) // outer_block_size
    return (
        weight_bytes
        + inner_scale_count * scale_dtype_bytes(sub_scale_dtype)
        + outer_scale_count * scale_dtype_bytes(super_scale_dtype)
    )


def total_bytes_format3(numel: int, inner_block_size: int) -> int:
    weight_bytes = (numel + 1) // 2
    inner_scale_count = (numel + inner_block_size - 1) // inner_block_size
    return weight_bytes + inner_scale_count + 4


def relative_rmse(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    squared_error_sum = torch.sum((reconstructed - original).square(), dtype=torch.float32)
    squared_original_sum = torch.sum(original.square(), dtype=torch.float32)
    numel = original.numel()
    rmse = torch.sqrt(squared_error_sum / numel)
    rms = torch.sqrt(squared_original_sum / numel)
    if rms.item() == 0:
        return 0.0 if rmse.item() == 0 else float("inf")
    return float((rmse / rms).item())


def pack_int4(values: torch.Tensor) -> torch.Tensor:
    flattened = values.reshape(-1).to(dtype=torch.int16, device="cpu")
    if flattened.numel() % 2:
        flattened = F.pad(flattened, (0, 1))

    low = flattened[0::2] & 0x0F
    high = (flattened[1::2] & 0x0F) << 4
    return (low | high).to(torch.uint8)


def unpack_int4(packed: torch.Tensor, num_values: int) -> torch.Tensor:
    packed_flat = packed.reshape(-1).to(dtype=torch.int16, device="cpu")
    unpacked = torch.empty((packed_flat.numel() * 2,), dtype=torch.int16)
    unpacked[0::2] = packed_flat & 0x0F
    unpacked[1::2] = (packed_flat >> 4) & 0x0F
    unpacked = torch.where(unpacked >= 8, unpacked - 16, unpacked)
    return unpacked[:num_values].to(torch.int8)


def quantize_format2_block(
    tensor: torch.Tensor,
    inner_block_size: int,
    outer_block_size: int,
    sub_scale_dtype: str,
    super_scale_dtype: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...], int]:
    tensor_fp32 = tensor.detach().to(dtype=torch.float32, device="cpu")
    original_shape = tensor_fp32.shape
    flattened = tensor_fp32.reshape(-1)
    original_numel = flattened.numel()

    if original_numel == 0:
        return (
            torch.empty((0,), dtype=torch.uint8),
            torch.empty((0,), dtype=torch.int8),
            torch.empty((0,), dtype=torch.float16),
            original_shape,
            0,
        )

    pad_len = (outer_block_size - (original_numel % outer_block_size)) % outer_block_size
    if pad_len:
        flattened = F.pad(flattened, (0, pad_len))

    outer_blocks = flattened.view(-1, outer_block_size)
    outer_scales = outer_blocks.abs().amax(dim=1, keepdim=True) / float(INT8_MAX)
    outer_scales = torch.where(outer_scales == 0, torch.full_like(outer_scales, ZERO_SCALE), outer_scales)

    inner_blocks = outer_blocks.view(-1, outer_block_size // inner_block_size, inner_block_size)
    normalized_inner = inner_blocks / outer_scales.view(-1, 1, 1)
    raw_inner_scales = normalized_inner.abs().amax(dim=2) / float(INT4_MAX)
    raw_inner_scales = torch.where(
        raw_inner_scales == 0,
        torch.full_like(raw_inner_scales, ZERO_SCALE),
        raw_inner_scales,
    )

    if sub_scale_dtype == "int8":
        inner_scales = torch.round(raw_inner_scales).clamp(1, INT8_MAX).to(torch.int8)
    elif sub_scale_dtype == "fp16":
        inner_scales = raw_inner_scales.to(torch.float16)
    else:
        raise ValueError(f"Unsupported f2 sub-scale dtype: {sub_scale_dtype}")

    if super_scale_dtype == "fp16":
        stored_outer_scales = outer_scales.reshape(-1).to(torch.float16)
    elif super_scale_dtype == "fp32":
        stored_outer_scales = outer_scales.reshape(-1).to(torch.float32)
    else:
        raise ValueError(f"Unsupported f2 super-scale dtype: {super_scale_dtype}")

    effective_inner_scales = inner_scales.to(torch.float32)
    quantized = torch.round(
        inner_blocks / (outer_scales.view(-1, 1, 1) * effective_inner_scales.unsqueeze(-1))
    ).clamp(INT4_MIN, INT4_MAX).to(torch.int8)

    packed = pack_int4(quantized.reshape(-1))
    return (
        packed,
        inner_scales.reshape(-1),
        stored_outer_scales,
        original_shape,
        original_numel,
    )


def dequantize_format2_block(
    packed_weights: torch.Tensor,
    inner_scales: torch.Tensor,
    outer_scales: torch.Tensor,
    inner_block_size: int,
    outer_block_size: int,
    original_shape: tuple[int, ...],
    original_numel: int,
) -> torch.Tensor:
    if original_numel == 0:
        return torch.empty(original_shape, dtype=torch.float16)

    unpacked = unpack_int4(packed_weights, inner_scales.numel() * inner_block_size).to(torch.float32)
    inner_blocks = unpacked.view(-1, inner_block_size)
    dequantized_inner = inner_blocks * inner_scales.to(torch.float32).unsqueeze(1)

    num_inner_per_outer = outer_block_size // inner_block_size
    dequantized_outer = dequantized_inner.view(-1, num_inner_per_outer, inner_block_size)
    dequantized = dequantized_outer * outer_scales.to(torch.float32).view(-1, 1, 1)
    flattened = dequantized.reshape(-1)[:original_numel]
    return flattened.view(original_shape).to(torch.float16)


def quantize_format3_tensor(
    tensor: torch.Tensor,
    inner_block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...], int]:
    tensor_fp32 = tensor.detach().to(dtype=torch.float32, device="cpu")
    original_shape = tensor_fp32.shape
    flattened = tensor_fp32.reshape(-1)
    original_numel = flattened.numel()

    if original_numel == 0:
        return (
            torch.empty((0,), dtype=torch.uint8),
            torch.empty((0,), dtype=torch.int8),
            torch.tensor(1.0, dtype=torch.float32),
            original_shape,
            0,
        )

    pad_len = (inner_block_size - (original_numel % inner_block_size)) % inner_block_size
    if pad_len:
        flattened = F.pad(flattened, (0, pad_len))

    tensor_scale = flattened.abs().amax() / float(INT8_MAX)
    if float(tensor_scale.item()) == 0.0:
        tensor_scale = torch.tensor(ZERO_SCALE, dtype=torch.float32)

    inner_blocks = flattened.view(-1, inner_block_size)
    raw_inner_scales = (inner_blocks / tensor_scale).abs().amax(dim=1) / float(INT4_MAX)
    raw_inner_scales = torch.where(
        raw_inner_scales == 0,
        torch.full_like(raw_inner_scales, ZERO_SCALE),
        raw_inner_scales,
    )

    inner_scales = torch.round(raw_inner_scales).clamp(1, INT8_MAX).to(torch.int8)
    effective_inner_scales = inner_scales.to(torch.float32)
    quantized = torch.round(
        inner_blocks / (tensor_scale * effective_inner_scales.unsqueeze(1))
    ).clamp(INT4_MIN, INT4_MAX).to(torch.int8)

    packed = pack_int4(quantized.reshape(-1))
    return packed, inner_scales, tensor_scale.to(torch.float32), original_shape, original_numel


def dequantize_format3_tensor(
    packed_weights: torch.Tensor,
    inner_scales: torch.Tensor,
    tensor_scale: torch.Tensor,
    inner_block_size: int,
    original_shape: tuple[int, ...],
    original_numel: int,
) -> torch.Tensor:
    if original_numel == 0:
        return torch.empty(original_shape, dtype=torch.float16)

    unpacked = unpack_int4(packed_weights, inner_scales.numel() * inner_block_size).to(torch.float32)
    inner_blocks = unpacked.view(-1, inner_block_size)
    dequantized = inner_blocks * inner_scales.to(torch.float32).unsqueeze(1) * tensor_scale.to(torch.float32)
    flattened = dequantized.reshape(-1)[:original_numel]
    return flattened.view(original_shape).to(torch.float16)


def evaluate_format1_candidate(original: torch.Tensor, block_size: int, scale_dtype: str) -> CandidateResult:
    quantized, scales = quantize_format1(original, block_size, scale_dtype)
    reconstructed = dequantize_format1(quantized, scales, block_size)
    return CandidateResult(
        name=f"f1_i8_{scale_dtype}b{block_size}",
        format_family="f1",
        total_bytes=total_bytes_format1_with_scale_dtype(original.numel(), block_size, scale_dtype),
        relative_rmse=relative_rmse(original, reconstructed),
    )


def evaluate_format2_candidate(
    original: torch.Tensor,
    inner_block_size: int,
    outer_block_size: int,
    sub_scale_dtype: str,
    super_scale_dtype: str,
) -> CandidateResult:
    packed, inner_scales, outer_scales, original_shape, original_numel = quantize_format2_block(
        original,
        inner_block_size,
        outer_block_size,
        sub_scale_dtype,
        super_scale_dtype,
    )
    reconstructed = dequantize_format2_block(
        packed,
        inner_scales,
        outer_scales,
        inner_block_size,
        outer_block_size,
        original_shape,
        original_numel,
    )
    return CandidateResult(
        name=f"f2_i4_{sub_scale_dtype}b{inner_block_size}_{super_scale_dtype}b{outer_block_size}",
        format_family="f2",
        total_bytes=total_bytes_format2(
            original.numel(),
            inner_block_size,
            outer_block_size,
            sub_scale_dtype,
            super_scale_dtype,
        ),
        relative_rmse=relative_rmse(original, reconstructed),
    )


def evaluate_format3_candidate(original: torch.Tensor, inner_block_size: int) -> CandidateResult:
    packed, inner_scales, tensor_scale, original_shape, original_numel = quantize_format3_tensor(
        original,
        inner_block_size,
    )
    reconstructed = dequantize_format3_tensor(
        packed,
        inner_scales,
        tensor_scale,
        inner_block_size,
        original_shape,
        original_numel,
    )
    return CandidateResult(
        name=f"f3_i4_i8b{inner_block_size}_fp32tensor",
        format_family="f3",
        total_bytes=total_bytes_format3(original.numel(), inner_block_size),
        relative_rmse=relative_rmse(original, reconstructed),
    )


def evaluate_tensor_candidates(tensor: torch.Tensor) -> list[CandidateResult]:
    if tensor.ndim != 2 or tensor.numel() == 0 or not torch.is_floating_point(tensor):
        return []

    original = tensor.detach().to(dtype=torch.float16, device="cpu")
    candidates: list[CandidateResult] = []

    for scale_dtype in F1_SCALE_DTYPES:
        for block_size in BLOCK_SIZES:
            candidates.append(evaluate_format1_candidate(original, block_size, scale_dtype))

    for inner_block_size in DOUBLE_INNER_BLOCK_SIZES:
        for outer_block_size in DOUBLE_OUTER_BLOCK_SIZES:
            if outer_block_size <= inner_block_size or outer_block_size % inner_block_size != 0:
                continue
            for sub_scale_dtype in F2_SUB_SCALE_DTYPES:
                for super_scale_dtype in F2_SUPER_SCALE_DTYPES:
                    candidates.append(
                        evaluate_format2_candidate(
                            original,
                            inner_block_size,
                            outer_block_size,
                            sub_scale_dtype,
                            super_scale_dtype,
                        )
                    )

    for inner_block_size in DOUBLE_INNER_BLOCK_SIZES:
        candidates.append(evaluate_format3_candidate(original, inner_block_size))

    return sorted(candidates, key=lambda item: (item.total_bytes, item.relative_rmse, item.name))


def normalize_value(value: float, minimum: float, maximum: float) -> float:
    if maximum == minimum:
        return 0.0
    return (value - minimum) / (maximum - minimum)


def detect_knee(candidates: list[CandidateResult]) -> CandidateResult | None:
    if not candidates:
        return None

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 2:
        return candidates[len(candidates) // 2]

    x_values = [candidate.total_bytes for candidate in candidates]
    y_values = [candidate.relative_rmse for candidate in candidates]
    x_min = float(min(x_values))
    x_max = float(max(x_values))
    y_min = float(min(y_values))
    y_max = float(max(y_values))

    normalized_points = [
        (
            normalize_value(float(candidate.total_bytes), x_min, x_max),
            normalize_value(candidate.relative_rmse, y_min, y_max),
        )
        for candidate in candidates
    ]

    x1, y1 = normalized_points[0]
    x2, y2 = normalized_points[-1]
    denominator = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5
    if denominator == 0.0:
        return candidates[len(candidates) // 2]

    knee_index = 1
    max_distance = -1.0
    for index in range(1, len(normalized_points) - 1):
        x0, y0 = normalized_points[index]
        distance = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / denominator
        if distance > max_distance:
            max_distance = distance
            knee_index = index

    return candidates[knee_index]


def select_knee_window(candidates: list[CandidateResult]) -> list[CandidateResult]:
    if not candidates:
        return []

    knee = detect_knee(candidates)
    if knee is None:
        return []

    knee_index = candidates.index(knee)
    if len(candidates) <= 3:
        return candidates

    start = max(0, knee_index - 1)
    end = start + 3
    if end > len(candidates):
        end = len(candidates)
        start = end - 3
    return candidates[start:end]


def format_candidate_group(candidates: list[CandidateResult]) -> str:
    knee = detect_knee(candidates)
    labels = [
        f"{candidate.name}(k)" if knee is not None and candidate == knee else candidate.name
        for candidate in candidates
    ]
    return f"({', '.join(labels)})"


def evaluate_tensor(tensor: torch.Tensor) -> tuple[dict[str, list[CandidateResult]], CandidateResult] | None:
    candidates = evaluate_tensor_candidates(tensor)
    if not candidates:
        return None

    window_by_family: dict[str, list[CandidateResult]] = {}
    for family in ("f1", "f2", "f3"):
        family_candidates = [candidate for candidate in candidates if candidate.format_family == family]
        family_window = select_knee_window(family_candidates)
        if not family_window:
            return None
        window_by_family[family] = family_window

    overall_candidates = sorted(
        [candidate for family in ("f1", "f2", "f3") for candidate in window_by_family[family]],
        key=lambda item: (item.total_bytes, item.relative_rmse, item.name),
    )
    overall_knee = detect_knee(overall_candidates)
    if overall_knee is None:
        return None

    return window_by_family, overall_knee


def main() -> None:
    args = parse_args()
    log_step(f"resolving input: {args.input}")
    paths = resolve_input_paths(args.input)
    log_step(f"found {len(paths)} safetensors file(s)")

    with torch.inference_mode():
        current_path: Path | None = None
        file_index = 0
        file_eligible = 0
        file_evaluated = 0
        total_evaluated = 0
        overall_name_counts: Counter[str] = Counter()

        for path, name, tensor in iter_tensors(paths):
            if current_path != path:
                if current_path is not None:
                    log_step(
                        f"finished [{file_index}/{len(paths)}]: eligible={file_eligible}, evaluated={file_evaluated}"
                    )
                current_path = path
                file_index += 1
                file_eligible = 0
                file_evaluated = 0
                log_step(f"scanning [{file_index}/{len(paths)}]: {path}")

            if not is_linear_weight_tensor(name, tensor):
                continue

            file_eligible += 1
            evaluation = evaluate_tensor(tensor)
            if evaluation is None:
                continue

            window_by_family, overall_knee = evaluation
            file_evaluated += 1
            total_evaluated += 1
            overall_name_counts[overall_knee.name] += 1
            print(
                f"{total_evaluated}: shape={tuple(tensor.shape)}, "
                f"{format_candidate_group(window_by_family['f1'])}, "
                f"{format_candidate_group(window_by_family['f2'])}, "
                f"{format_candidate_group(window_by_family['f3'])}, "
                f"overall:{overall_knee.name}",
                flush=True,
            )

        if current_path is not None:
            log_step(
                f"finished [{file_index}/{len(paths)}]: eligible={file_eligible}, evaluated={file_evaluated}"
            )

    print(f"files_scanned={len(paths)}")
    print(f"tensors_evaluated={total_evaluated}")
    if overall_name_counts:
        print("overall_variant_tally:")
        for knee_name, count in overall_name_counts.most_common():
            print(f"  {knee_name}={count}")


if __name__ == "__main__":
    main()
