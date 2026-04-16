from __future__ import annotations

import gc
import statistics
import time
from dataclasses import dataclass

import torch

from utils.quant.double import dequantize_from_double_block, quantize_to_double_block
from utils.quant.single import dequantize_from_single_block, quantize_to_single_block

LINEAR_SHAPES = (
    (6144, 128),
    (6144, 15360),
    (6144, 6144),
    (36864, 6144),
    (6144, 18432),
    (55296, 6144),
    (6144, 24576),
)

METHODS = ("single_block_dequant", "double_block_dequant")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DTYPE = torch.float16
REPEATS = 100
WARMUP = 10
SEED = 19930625


@dataclass(frozen=True)
class BenchmarkResult:
    shape: tuple[int, int]
    method: str
    mean_ms: float
    median_ms: float
    best_ms: float
    max_abs_diff: float
    mean_abs_diff: float
    input_bytes: int
    output_bytes: int


@dataclass
class QuantizedCase:
    method: str
    shape: tuple[int, int]
    output_shape: tuple[int, ...]
    reference: torch.Tensor
    tensor: torch.Tensor
    scales: torch.Tensor | None = None
    sub_scales: torch.Tensor | None = None
    super_scales: torch.Tensor | None = None


def _shape_label(shape: tuple[int, int]) -> str:
    return f"{shape[0]}x{shape[1]}"


def _numel(shape: tuple[int, int]) -> int:
    return shape[0] * shape[1]


def _cleanup_device(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _case_seed(shape: tuple[int, int]) -> int:
    return SEED + shape[0] * 100_003 + shape[1] * 1_009


def _build_dense_weight(shape: tuple[int, int], *, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=OUTPUT_DTYPE)


def _prepare_single_case(
    shape: tuple[int, int],
    *,
    device: torch.device,
) -> QuantizedCase:
    dense_weight = _build_dense_weight(shape, seed=_case_seed(shape))
    quantized, scales = quantize_to_single_block(dense_weight)
    return QuantizedCase(
        method="single_block_dequant",
        shape=shape,
        output_shape=shape,
        reference=dense_weight.to(device=device),
        tensor=quantized.to(device=device),
        scales=scales.to(device=device),
    )


def _prepare_double_case(
    shape: tuple[int, int],
    *,
    device: torch.device,
) -> QuantizedCase:
    dense_weight = _build_dense_weight(shape, seed=_case_seed(shape)).to(device=device)
    packed, sub_scales, super_scales = quantize_to_double_block(dense_weight)
    return QuantizedCase(
        method="double_block_dequant",
        shape=shape,
        output_shape=shape,
        reference=dense_weight,
        tensor=packed,
        sub_scales=sub_scales,
        super_scales=super_scales,
    )


def _prepare_quantized_case(
    shape: tuple[int, int],
    *,
    method: str,
    device: torch.device,
) -> QuantizedCase:
    if method == "single_block_dequant":
        return _prepare_single_case(shape, device=device)
    if method == "double_block_dequant":
        if device.type != "cuda":
            raise RuntimeError("double_block_dequant benchmark requires CUDA.")
        return _prepare_double_case(shape, device=device)
    raise ValueError(f"Unsupported method: {method}")


def _run_method(quantized_case: QuantizedCase) -> torch.Tensor:
    if quantized_case.method == "single_block_dequant":
        if quantized_case.scales is None:
            raise ValueError("single_block_dequant case is missing scales.")
        return dequantize_from_single_block(quantized_case.tensor, quantized_case.scales)

    if quantized_case.method == "double_block_dequant":
        if quantized_case.sub_scales is None or quantized_case.super_scales is None:
            raise ValueError("double_block_dequant case is missing sub_scales or super_scales.")
        return dequantize_from_double_block(
            quantized_case.tensor,
            quantized_case.sub_scales,
            quantized_case.super_scales,
        ).view(quantized_case.output_shape)

    raise ValueError(f"Unsupported method: {quantized_case.method}")


def _input_bytes(quantized_case: QuantizedCase) -> int:
    total = quantized_case.tensor.numel() * quantized_case.tensor.element_size()
    if quantized_case.scales is not None:
        total += quantized_case.scales.numel() * quantized_case.scales.element_size()
    if quantized_case.sub_scales is not None:
        total += quantized_case.sub_scales.numel() * quantized_case.sub_scales.element_size()
    if quantized_case.super_scales is not None:
        total += quantized_case.super_scales.numel() * quantized_case.super_scales.element_size()
    return total


def _accuracy_metrics(quantized_case: QuantizedCase) -> tuple[float, float]:
    with torch.inference_mode():
        output = _run_method(quantized_case)
        diff = (output - quantized_case.reference).abs()
        max_abs_diff = float(diff.max().item())
        mean_abs_diff = float(diff.mean().item())
        del diff
        del output
    return max_abs_diff, mean_abs_diff


def _print_progress(result: BenchmarkResult) -> None:
    print(
        f"[{result.method}] "
        f"[{_shape_label(result.shape)}] "
        f"median={result.median_ms:.3f} ms "
        f"mean={result.mean_ms:.3f} ms "
        f"best={result.best_ms:.3f} ms "
        f"max_abs_diff={result.max_abs_diff:.6f} "
        f"mean_abs_diff={result.mean_abs_diff:.6f}"
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
            result.method,
            f"{result.median_ms:.3f}",
            f"{result.mean_ms:.3f}",
            f"{result.best_ms:.3f}",
            f"{result.max_abs_diff:.6f}",
            f"{result.mean_abs_diff:.6f}",
            str(result.input_bytes),
            str(result.output_bytes),
        ]
        for result in sorted(results, key=lambda item: (item.shape, item.method))
    ]
    _print_table(
        [
            "shape",
            "method",
            "median_ms",
            "mean_ms",
            "best_ms",
            "max_abs_diff",
            "mean_abs_diff",
            "input_bytes",
            "output_bytes",
        ],
        rows,
    )


def _print_best_per_shape(results: list[BenchmarkResult]) -> None:
    print("\nBest Method Per Shape")
    grouped: dict[tuple[int, int], list[BenchmarkResult]] = {}
    for result in results:
        grouped.setdefault(result.shape, []).append(result)

    rows: list[list[str]] = []
    for shape in sorted(grouped):
        winner = min(grouped[shape], key=lambda item: item.median_ms)
        rows.append(
            [
                _shape_label(shape),
                winner.method,
                f"{winner.median_ms:.3f}",
                f"{winner.mean_ms:.3f}",
                f"{winner.best_ms:.3f}",
                f"{winner.max_abs_diff:.6f}",
                f"{winner.mean_abs_diff:.6f}",
            ]
        )

    _print_table(
        ["shape", "winner", "median_ms", "mean_ms", "best_ms", "max_abs_diff", "mean_abs_diff"],
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
        rows.append(
            [
                method,
                str(len(method_results)),
                f"{sum(item.median_ms for item in method_results) / len(method_results):.3f}",
                f"{sum(item.mean_ms for item in method_results) / len(method_results):.3f}",
                f"{sum(item.best_ms for item in method_results) / len(method_results):.3f}",
                f"{max(item.max_abs_diff for item in method_results):.6f}",
                f"{sum(item.mean_abs_diff for item in method_results) / len(method_results):.6f}",
            ]
        )

    _print_table(
        ["method", "cases", "avg_median_ms", "avg_mean_ms", "avg_best_ms", "max_abs_diff", "avg_mean_abs_diff"],
        rows,
    )


def _benchmark_case(
    quantized_case: QuantizedCase,
    *,
    device: torch.device,
) -> BenchmarkResult:
    with torch.inference_mode():
        for _ in range(WARMUP):
            output = _run_method(quantized_case)
            del output

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        elapsed_ms: list[float] = []
        for _ in range(REPEATS):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            output = _run_method(quantized_case)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed_ms.append((time.perf_counter() - start) * 1000.0)
            del output

    max_abs_diff, mean_abs_diff = _accuracy_metrics(quantized_case)

    return BenchmarkResult(
        shape=quantized_case.shape,
        method=quantized_case.method,
        mean_ms=sum(elapsed_ms) / len(elapsed_ms),
        median_ms=statistics.median(elapsed_ms),
        best_ms=min(elapsed_ms),
        max_abs_diff=max_abs_diff,
        mean_abs_diff=mean_abs_diff,
        input_bytes=_input_bytes(quantized_case),
        output_bytes=_numel(quantized_case.shape) * torch.tensor([], dtype=OUTPUT_DTYPE).element_size(),
    )


def main() -> None:
    if DEVICE.type != "cuda":
        raise RuntimeError("utils.quant.eval requires CUDA to benchmark both single and double block dequant.")

    print("Single/Double block dequant benchmark")
    print(f"Device: {DEVICE}")
    print(f"Repeats: {REPEATS} | Warmup: {WARMUP}")
    print("This benchmark times dequant to fp16 for the current utils.quant single/double formats.\n")

    results: list[BenchmarkResult] = []
    total_cases = len(LINEAR_SHAPES) * len(METHODS)
    case_index = 0

    for shape in LINEAR_SHAPES:
        for method in METHODS:
            case_index += 1
            print(f"[{case_index}/{total_cases}] shape={_shape_label(shape)} method={method}")
            quantized_case = _prepare_quantized_case(shape, method=method, device=DEVICE)
            result = _benchmark_case(quantized_case, device=DEVICE)
            results.append(result)
            _print_progress(result)
            del quantized_case
            _cleanup_device(DEVICE)

    _print_full_summary(results)
    _print_best_per_shape(results)
    _print_method_averages(results)


if __name__ == "__main__":
    main()
