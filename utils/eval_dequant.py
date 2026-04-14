from __future__ import annotations

import gc
import importlib
import importlib.util
import statistics
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

from utils.dequantize import dequantize_from_block
from utils.quantize import quantize_to_block

LINEAR_SHAPES = (
    (6144, 128),
    (6144, 15360),
    # (6144, 6144),
    # (36864, 6144),
    # (6144, 18432),
    # (55296, 6144),
    # (6144, 24576),
)

QUANT_PRECISIONS = ("int4", "int8")
SCALE_PRECISION = "fp16"
BLOCK_SIZES = (32, 64, 128)

METHODS = (
    "int4_reference_dequant",
    "int4_cuda_dequant_kernel",
    "int8_reference_dequant",
    "int8_cuda_dequant_kernel",
)

METHOD_COMPATIBILITY = {
    "int4": ("int4_reference_dequant", "int4_cuda_dequant_kernel"),
    "int8": ("int8_reference_dequant", "int8_cuda_dequant_kernel"),
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DTYPE = torch.float16
REPEATS = 100
WARMUP = 10
SEED = 19930625


@dataclass(frozen=True)
class BenchmarkResult:
    shape: tuple[int, int]
    quantization_precision: str
    block_size: int
    method: str
    mean_ms: float
    median_ms: float
    best_ms: float
    max_abs_diff: float
    passed: bool
    input_bytes: int
    output_bytes: int


@dataclass
class QuantizedCase:
    tensor: torch.Tensor
    scales: torch.Tensor
    original_numel: int
    shape: tuple[int, int]


def _shape_label(shape: tuple[int, int]) -> str:
    return f"{shape[0]}x{shape[1]}"


def _numel(shape: tuple[int, int]) -> int:
    return shape[0] * shape[1]


def _cleanup_device(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _case_seed(shape: tuple[int, int], block_size: int) -> int:
    return SEED + shape[0] * 100_003 + shape[1] * 1_009 + block_size


def _build_dense_weight(shape: tuple[int, int], *, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=OUTPUT_DTYPE)


@lru_cache(maxsize=1)
def _load_prebuilt_int4_dequant_module():
    module_name = "int4_dequant_cuda"
    cuda_dir = Path(__file__).resolve().parent / "cuda" / "int4_dequant"

    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        pass

    suffixes = importlib.machinery.EXTENSION_SUFFIXES
    for suffix in suffixes:
        matches = sorted(cuda_dir.glob(f"{module_name}*{suffix}"))
        if not matches:
            continue
        module_path = matches[0]
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    raise ModuleNotFoundError(
        "Prebuilt module 'int4_dequant_cuda' not found. Build it first with "
        "'python setup.py build_ext --inplace' in utils/cuda/int4_dequant."
    )


@lru_cache(maxsize=1)
def _load_prebuilt_int8_dequant_module():
    module_name = "int8_dequant_cuda"
    cuda_dir = Path(__file__).resolve().parent / "cuda" / "int8_dequant"

    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        pass

    suffixes = importlib.machinery.EXTENSION_SUFFIXES
    for suffix in suffixes:
        matches = sorted(cuda_dir.glob(f"{module_name}*{suffix}"))
        if not matches:
            continue
        module_path = matches[0]
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    raise ModuleNotFoundError(
        "Prebuilt module 'int8_dequant_cuda' not found. Build it first with "
        "'python setup.py build_ext --inplace' in utils/cuda/int8_dequant."
    )


def _prepare_quantized_case(
    shape: tuple[int, int],
    *,
    quantization_precision: str,
    block_size: int,
    device: torch.device,
) -> QuantizedCase:
    dense_weight = _build_dense_weight(shape, seed=_case_seed(shape, block_size))
    quantized, scales = quantize_to_block(
        dense_weight,
        block_size=block_size,
        quantization_precision=quantization_precision,
        scaling_precision=SCALE_PRECISION,
    )
    del dense_weight
    return QuantizedCase(
        tensor=quantized.to(device=device),
        scales=scales.to(device=device),
        original_numel=_numel(shape),
        shape=shape,
    )


def _reference_dequant(
    quantized_case: QuantizedCase,
    *,
    quantization_precision: str,
    block_size: int,
) -> torch.Tensor:
    return dequantize_from_block(
        quantized_case.tensor,
        quantized_case.scales,
        block_size=block_size,
        quantization_precision=quantization_precision,
        scaling_precision=SCALE_PRECISION,
        output_dtype=OUTPUT_DTYPE,
        output_shape=quantized_case.shape,
        use_int4_cuda_kernel=False,
    )


def _cuda_int4_dequant(
    quantized_case: QuantizedCase,
    *,
    block_size: int,
    output_fp16: torch.Tensor,
) -> torch.Tensor:
    module = _load_prebuilt_int4_dequant_module()
    module.dequantize_int4_fp16(
        quantized_case.tensor,
        quantized_case.scales,
        output_fp16,
        quantized_case.original_numel,
        block_size,
    )
    return output_fp16


def _cuda_int8_dequant(
    quantized_case: QuantizedCase,
    *,
    block_size: int,
    output_fp16: torch.Tensor,
) -> torch.Tensor:
    module = _load_prebuilt_int8_dequant_module()
    module.dequantize_int8_fp16(
        quantized_case.tensor,
        quantized_case.scales,
        output_fp16,
        quantized_case.original_numel,
        block_size,
    )
    return output_fp16


def _run_method(
    quantized_case: QuantizedCase,
    *,
    quantization_precision: str,
    block_size: int,
    method: str,
    output_fp16: torch.Tensor,
) -> torch.Tensor:
    if method == "int4_reference_dequant":
        return _reference_dequant(
            quantized_case,
            quantization_precision="int4",
            block_size=block_size,
        )
    if method == "int4_cuda_dequant_kernel":
        return _cuda_int4_dequant(
            quantized_case,
            block_size=block_size,
            output_fp16=output_fp16,
        )
    if method == "int8_reference_dequant":
        return _reference_dequant(
            quantized_case,
            quantization_precision="int8",
            block_size=block_size,
        )
    if method == "int8_cuda_dequant_kernel":
        return _cuda_int8_dequant(
            quantized_case,
            block_size=block_size,
            output_fp16=output_fp16,
        )
    raise ValueError(f"Unsupported method: {method}")


def _validate_method(
    quantized_case: QuantizedCase,
    *,
    quantization_precision: str,
    block_size: int,
    method: str,
    output_fp16: torch.Tensor,
) -> tuple[float, bool]:
    with torch.inference_mode():
        output = _run_method(
            quantized_case,
            quantization_precision=quantization_precision,
            block_size=block_size,
            method=method,
            output_fp16=output_fp16,
        )
        reference = _reference_dequant(
            quantized_case,
            quantization_precision=quantization_precision,
            block_size=block_size,
        )
        max_abs_diff = (output - reference).abs().max().item()
        passed = max_abs_diff == 0.0
        del output
        del reference
    return float(max_abs_diff), passed


def _print_progress(result: BenchmarkResult) -> None:
    print(
        f"[{result.quantization_precision}] "
        f"[b{result.block_size}] "
        f"[{_shape_label(result.shape)}] "
        f"[{result.method}] "
        f"median={result.median_ms:.3f} ms "
        f"mean={result.mean_ms:.3f} ms "
        f"best={result.best_ms:.3f} ms "
        f"max_abs_diff={result.max_abs_diff:.6f} "
        f"pass={'yes' if result.passed else 'no'}"
    )


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    header_row = "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    divider = "  ".join("-" * widths[index] for index in range(len(headers)))
    print(header_row)
    print(divider)
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def _print_full_summary(results: list[BenchmarkResult]) -> None:
    print("\nFull Results")
    rows = [
        [
            _shape_label(result.shape),
            result.quantization_precision,
            str(result.block_size),
            result.method,
            f"{result.median_ms:.3f}",
            f"{result.mean_ms:.3f}",
            f"{result.best_ms:.3f}",
            f"{result.max_abs_diff:.6f}",
            "yes" if result.passed else "no",
        ]
        for result in sorted(
            results,
            key=lambda item: (
                item.quantization_precision,
                item.shape,
                item.block_size,
                item.method,
            ),
        )
    ]
    _print_table(
        ["shape", "quant", "block", "method", "median_ms", "mean_ms", "best_ms", "max_abs_diff", "pass"],
        rows,
    )


def _print_best_per_case(results: list[BenchmarkResult]) -> None:
    print("\nBest Method Per Case")
    grouped: dict[tuple[tuple[int, int], str, int], list[BenchmarkResult]] = {}
    for result in results:
        grouped.setdefault((result.shape, result.quantization_precision, result.block_size), []).append(result)

    rows: list[list[str]] = []
    for key in sorted(grouped):
        shape, quantization_precision, block_size = key
        winner = min(grouped[key], key=lambda item: item.median_ms)
        rows.append(
            [
                _shape_label(shape),
                quantization_precision,
                str(block_size),
                winner.method,
                f"{winner.median_ms:.3f}",
                f"{winner.mean_ms:.3f}",
                f"{winner.best_ms:.3f}",
                f"{winner.max_abs_diff:.6f}",
            ]
        )

    _print_table(
        ["shape", "quant", "block", "winner", "median_ms", "mean_ms", "best_ms", "max_abs_diff"],
        rows,
    )


def _print_method_averages(results: list[BenchmarkResult]) -> None:
    print("\nMethod Averages")
    grouped: dict[str, list[BenchmarkResult]] = {}
    for result in results:
        grouped.setdefault(result.method, []).append(result)

    rows: list[list[str]] = []
    for method in METHODS:
        method_results = grouped.get(method, [])
        if not method_results:
            continue
        avg_median_ms = sum(item.median_ms for item in method_results) / len(method_results)
        avg_mean_ms = sum(item.mean_ms for item in method_results) / len(method_results)
        avg_best_ms = sum(item.best_ms for item in method_results) / len(method_results)
        max_diff = max(item.max_abs_diff for item in method_results)
        rows.append(
            [
                method,
                str(len(method_results)),
                f"{avg_median_ms:.3f}",
                f"{avg_mean_ms:.3f}",
                f"{avg_best_ms:.3f}",
                f"{max_diff:.6f}",
            ]
        )

    _print_table(
        ["method", "cases", "avg_median_ms", "avg_mean_ms", "avg_best_ms", "max_abs_diff"],
        rows,
    )


def _benchmark_case(
    quantized_case: QuantizedCase,
    *,
    quantization_precision: str,
    block_size: int,
    method: str,
    device: torch.device,
) -> BenchmarkResult:
    output_fp16 = torch.empty(quantized_case.shape, device=device, dtype=OUTPUT_DTYPE)

    with torch.inference_mode():
        for _ in range(WARMUP):
            _ = _run_method(
                quantized_case,
                quantization_precision=quantization_precision,
                block_size=block_size,
                method=method,
                output_fp16=output_fp16,
            )
        torch.cuda.synchronize(device)

        elapsed_ms: list[float] = []
        for _ in range(REPEATS):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            _ = _run_method(
                quantized_case,
                quantization_precision=quantization_precision,
                block_size=block_size,
                method=method,
                output_fp16=output_fp16,
            )
            torch.cuda.synchronize(device)
            elapsed_ms.append((time.perf_counter() - start) * 1000.0)

    mean_ms = sum(elapsed_ms) / len(elapsed_ms)
    median_ms = statistics.median(elapsed_ms)
    best_ms = min(elapsed_ms)
    max_abs_diff, passed = _validate_method(
        quantized_case,
        quantization_precision=quantization_precision,
        block_size=block_size,
        method=method,
        output_fp16=output_fp16,
    )

    return BenchmarkResult(
        shape=quantized_case.shape,
        quantization_precision=quantization_precision,
        block_size=block_size,
        method=method,
        mean_ms=mean_ms,
        median_ms=median_ms,
        best_ms=best_ms,
        max_abs_diff=max_abs_diff,
        passed=passed,
        input_bytes=(
            quantized_case.tensor.numel() * quantized_case.tensor.element_size()
            + quantized_case.scales.numel() * quantized_case.scales.element_size()
        ),
        output_bytes=quantized_case.original_numel * torch.tensor([], dtype=OUTPUT_DTYPE).element_size(),
    )


def main() -> None:
    print("INT4/INT8 full dequant benchmark")
    print(f"Device: {DEVICE}")
    print(f"Scale precision: {SCALE_PRECISION}")
    print(f"Repeats: {REPEATS} | Warmup: {WARMUP}")
    print("This benchmark times full dequant to fp16, including scale application.\n")

    results: list[BenchmarkResult] = []
    total_cases = (
        sum(len(METHOD_COMPATIBILITY[precision]) for precision in QUANT_PRECISIONS)
        * len(LINEAR_SHAPES)
        * len(BLOCK_SIZES)
    )
    case_index = 0

    for shape in LINEAR_SHAPES:
        for block_size in BLOCK_SIZES:
            for quantization_precision in QUANT_PRECISIONS:
                quantized_case = _prepare_quantized_case(
                    shape,
                    quantization_precision=quantization_precision,
                    block_size=block_size,
                    device=DEVICE,
                )

                for method in METHOD_COMPATIBILITY[quantization_precision]:
                    case_index += 1
                    print(
                        f"[{case_index}/{total_cases}] "
                        f"shape={_shape_label(shape)} "
                        f"quant={quantization_precision} "
                        f"block={block_size} "
                        f"method={method}"
                    )
                    result = _benchmark_case(
                        quantized_case,
                        quantization_precision=quantization_precision,
                        block_size=block_size,
                        method=method,
                        device=DEVICE,
                    )
                    results.append(result)
                    _print_progress(result)

                del quantized_case
                _cleanup_device(DEVICE)

    _print_full_summary(results)
    _print_best_per_case(results)
    _print_method_averages(results)


if __name__ == "__main__":
    main()
