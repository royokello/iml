from __future__ import annotations

import gc
import importlib
import statistics
import time
from dataclasses import dataclass
from functools import lru_cache

import torch

from utils.quant.cuda.affine_high import dequantize_from_affine_high as dequantize_from_affine_high_cuda
from utils.quant.cuda.affine_low import dequantize_from_affine_low as dequantize_from_affine_low_cuda
from utils.quant.cuda.symmetric_high import dequantize_from_symmetric_high as dequantize_from_symmetric_high_cuda
from utils.quant.cuda.symmetric_low import dequantize_from_symmetric_low as dequantize_from_symmetric_low_cuda
from utils.quant.to.affine import quantize_to_affine
from utils.quant.to.symmetric import quantize_to_symmetric

dequantize_from_affine = importlib.import_module("utils.quant.from.affine").dequantize_from_affine
dequantize_from_symmetric = importlib.import_module("utils.quant.from.symmetric").dequantize_from_symmetric

LINEAR_SHAPES = (
    (6144, 128),
    # (6144, 15360),
    (6144, 6144),
    # (36864, 6144),
    # (6144, 18432),
    # (55296, 6144),
    # (6144, 24576),
)

METHODS = (
    "affine_low_dequant",
    "affine_low_cuda_dequant",
    "affine_high_dequant",
    "affine_high_cuda_dequant",
    "symmetric_low_dequant",
    "symmetric_low_cuda_dequant",
    "symmetric_high_dequant",
    "symmetric_high_cuda_dequant",
)

DEVICE = torch.device("cuda")
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
    sub_scales: torch.Tensor | None = None
    sub_mins: torch.Tensor | None = None
    super_scales: torch.Tensor | None = None
    super_mins: torch.Tensor | None = None


def _shape_label(shape: tuple[int, int]) -> str:
    return f"{shape[0]}x{shape[1]}"


def _numel(shape: tuple[int, int]) -> int:
    return shape[0] * shape[1]


def _cleanup_device(device: torch.device) -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _case_seed(shape: tuple[int, int]) -> int:
    return SEED + shape[0] * 100_003 + shape[1] * 1_009


@lru_cache(maxsize=None)
def _build_dense_weight(shape: tuple[int, int], *, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=OUTPUT_DTYPE)


def _prepare_affine_case(
    shape: tuple[int, int],
    *,
    device: torch.device,
    mode: str,
    use_kernel: bool,
) -> QuantizedCase:
    dense_weight_cpu = _build_dense_weight(shape, seed=_case_seed(shape))
    dense_weight = dense_weight_cpu.to(device=device)
    qweight, sub_scales, sub_mins, super_scales, super_mins = quantize_to_affine(dense_weight, mode=mode)
    return QuantizedCase(
        method=f"affine_{mode}_{'cuda_' if use_kernel else ''}dequant",
        shape=shape,
        output_shape=shape,
        reference=dense_weight,
        tensor=qweight,
        sub_scales=sub_scales,
        sub_mins=sub_mins,
        super_scales=super_scales,
        super_mins=super_mins,
    )


def _prepare_symmetric_case(
    shape: tuple[int, int],
    *,
    device: torch.device,
    mode: str,
    use_kernel: bool,
) -> QuantizedCase:
    dense_weight_cpu = _build_dense_weight(shape, seed=_case_seed(shape))
    dense_weight = dense_weight_cpu.to(device=device)
    qweight, sub_scales, super_scales = quantize_to_symmetric(dense_weight, mode=mode)
    return QuantizedCase(
        method=f"symmetric_{mode}_{'cuda_' if use_kernel else ''}dequant",
        shape=shape,
        output_shape=shape,
        reference=dense_weight,
        tensor=qweight,
        sub_scales=sub_scales,
        super_scales=super_scales,
    )


def _prepare_quantized_case(
    shape: tuple[int, int],
    *,
    method: str,
    device: torch.device,
) -> QuantizedCase:
    if method == "affine_high_cuda_dequant":
        return _prepare_affine_case(shape, device=device, mode="high", use_kernel=True)
    if method == "affine_high_dequant":
        return _prepare_affine_case(shape, device=device, mode="high", use_kernel=False)
    if method == "affine_low_cuda_dequant":
        return _prepare_affine_case(shape, device=device, mode="low", use_kernel=True)
    if method == "affine_low_dequant":
        return _prepare_affine_case(shape, device=device, mode="low", use_kernel=False)
    if method == "symmetric_high_cuda_dequant":
        return _prepare_symmetric_case(shape, device=device, mode="high", use_kernel=True)
    if method == "symmetric_high_dequant":
        return _prepare_symmetric_case(shape, device=device, mode="high", use_kernel=False)
    if method == "symmetric_low_cuda_dequant":
        return _prepare_symmetric_case(shape, device=device, mode="low", use_kernel=True)
    if method == "symmetric_low_dequant":
        return _prepare_symmetric_case(shape, device=device, mode="low", use_kernel=False)
    raise ValueError(f"Unsupported method: {method}")


def _run_method(quantized_case: QuantizedCase) -> torch.Tensor:
    if quantized_case.method in {
        "affine_high_cuda_dequant",
        "affine_high_dequant",
        "affine_low_cuda_dequant",
        "affine_low_dequant",
    }:
        if (
            quantized_case.sub_scales is None
            or quantized_case.sub_mins is None
            or quantized_case.super_scales is None
            or quantized_case.super_mins is None
        ):
            raise ValueError(f"{quantized_case.method} case is missing affine metadata.")
        mode = "high" if "_high_" in quantized_case.method else "low"
        use_kernel = "_cuda_" in quantized_case.method
        if use_kernel:
            dequantize = dequantize_from_affine_high_cuda if mode == "high" else dequantize_from_affine_low_cuda
        else:
            dequantize = dequantize_from_affine
        return dequantize(
            quantized_case.tensor,
            quantized_case.sub_scales,
            quantized_case.sub_mins,
            quantized_case.super_scales,
            quantized_case.super_mins,
            quantized_case.output_shape,
            **({} if use_kernel else {"mode": mode}),
        )

    if quantized_case.method in {
        "symmetric_high_cuda_dequant",
        "symmetric_high_dequant",
        "symmetric_low_cuda_dequant",
        "symmetric_low_dequant",
    }:
        if quantized_case.sub_scales is None or quantized_case.super_scales is None:
            raise ValueError(f"{quantized_case.method} case is missing sub_scales or super_scales.")
        mode = "high" if "_high_" in quantized_case.method else "low"
        use_kernel = "_cuda_" in quantized_case.method
        if use_kernel:
            dequantize = (
                dequantize_from_symmetric_high_cuda
                if mode == "high"
                else dequantize_from_symmetric_low_cuda
            )
        else:
            dequantize = dequantize_from_symmetric
        return dequantize(
            quantized_case.tensor,
            quantized_case.sub_scales,
            quantized_case.super_scales,
            quantized_case.output_shape,
            **({} if use_kernel else {"mode": mode}),
        )

    raise ValueError(f"Unsupported method: {quantized_case.method}")


def _input_bytes(quantized_case: QuantizedCase) -> int:
    total = quantized_case.tensor.numel() * quantized_case.tensor.element_size()
    if quantized_case.sub_scales is not None:
        total += quantized_case.sub_scales.numel() * quantized_case.sub_scales.element_size()
    if quantized_case.sub_mins is not None:
        total += quantized_case.sub_mins.numel() * quantized_case.sub_mins.element_size()
    if quantized_case.super_scales is not None:
        total += quantized_case.super_scales.numel() * quantized_case.super_scales.element_size()
    if quantized_case.super_mins is not None:
        total += quantized_case.super_mins.numel() * quantized_case.super_mins.element_size()
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
    method_order = {method: index for index, method in enumerate(METHODS)}
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
        for result in sorted(results, key=lambda item: (item.shape, method_order[item.method]))
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


def _print_kernel_speedups(results: list[BenchmarkResult]) -> None:
    print("\nKernel Speedups")
    by_method: dict[str, list[BenchmarkResult]] = {}
    for result in results:
        by_method.setdefault(result.method, []).append(result)

    rows: list[list[str]] = []
    for method in METHODS:
        if "_cuda_dequant" not in method:
            continue
        baseline_method = method.replace("_cuda_dequant", "_dequant")
        baseline_results = by_method.get(baseline_method, [])
        kernel_results = by_method.get(method, [])
        if not baseline_results or not kernel_results:
            continue

        baseline_median = sum(item.median_ms for item in baseline_results) / len(baseline_results)
        kernel_median = sum(item.median_ms for item in kernel_results) / len(kernel_results)
        rows.append([method, f"x{baseline_median / kernel_median:.1f}"])

    _print_table(["method", "speedup"], rows)


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

        torch.cuda.synchronize(device)

        elapsed_ms: list[float] = []
        for _ in range(REPEATS):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            output = _run_method(quantized_case)
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


def _active_methods(device: torch.device) -> tuple[str, ...]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("utils.quant.eval benchmarks CUDA tensors only.")
    return METHODS


def main() -> None:
    methods = _active_methods(DEVICE)

    print("Affine/Symmetric CUDA tensor dequant benchmark")
    print(f"Device: {DEVICE}")
    print(f"Repeats: {REPEATS} | Warmup: {WARMUP}")
    print("This benchmark times CUDA kernels and non-kernel torch dequantizers on CUDA tensors.")
    print()

    results: list[BenchmarkResult] = []
    total_cases = len(LINEAR_SHAPES) * len(methods)
    case_index = 0

    for shape in LINEAR_SHAPES:
        for method in methods:
            case_index += 1
            print(f"[{case_index}/{total_cases}] shape={_shape_label(shape)} method={method}")
            quantized_case = _prepare_quantized_case(shape, method=method, device=DEVICE)
            result = _benchmark_case(quantized_case, device=DEVICE)
            results.append(result)
            _print_progress(result)
            del quantized_case
            _cleanup_device(DEVICE)

    _print_full_summary(results)
    _print_kernel_speedups(results)
    _print_method_averages(results)


if __name__ == "__main__":
    main()
