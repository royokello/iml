import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

BATCH = 2
TOKENS = 256
# Largest SDXL UNet FFN expansion: 1280 -> 5120
IN_FEATURES = 1280
OUT_FEATURES = 5120
WARMUP = 10
ITERS = 50

_INT8_LINEAR_MODULE = None


def load_int8_linear_extension():
    global _INT8_LINEAR_MODULE
    if _INT8_LINEAR_MODULE is None:
        repo_root = Path(__file__).resolve().parents[1]
        cu_src = repo_root / "cuda" / "int8_linear.cu"
        cpp_bind = repo_root / "cuda" / "int8_linear_bindings.cpp"

        extra_ldflags: list[str] = []
        cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        if os.name == "nt":
            if cuda_home:
                lib_dir = Path(cuda_home) / "lib" / "x64"
                extra_ldflags.append(f"/LIBPATH:{lib_dir}")
            extra_ldflags += ["cublasLt.lib", "cublas.lib"]
        else:
            extra_ldflags += ["-lcublasLt", "-lcublas"]

        _INT8_LINEAR_MODULE = load(
            name="int8_linear_ops",
            sources=[str(cu_src), str(cpp_bind)],
            extra_cuda_cflags=["-O3"],
            extra_cflags=["-O3"],
            extra_ldflags=extra_ldflags,
        )
    return _INT8_LINEAR_MODULE


def linear_flops(batch, tokens, cin, cout) -> float:
    return float(batch) * tokens * cin * cout * 2.0


def quantize_to_int8(t: torch.Tensor):
    max_abs = t.abs().max()
    scale = max_abs / 127.0 + 1e-8
    q = torch.clamp((t / scale).round(), -128, 127).to(torch.int8)
    return q, scale


def benchmark_fp16(x, w, b, flops):
    for _ in range(WARMUP):
        F.linear(x, w, b)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    out = None
    for _ in range(ITERS):
        start.record()
        out = F.linear(x, w, b)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    avg_ms = sum(times) / len(times)
    tflops = flops / 1e12 / (avg_ms / 1e3)
    return avg_ms, tflops, out.detach()


def benchmark_int8(x_q, w_q, bias, scale_factor, apply_scale, flops):
    module = load_int8_linear_extension()

    for _ in range(WARMUP):
        module.int8_linear(x_q, w_q, bias, float(scale_factor), apply_scale)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    out = None
    for _ in range(ITERS):
        start.record()
        out = module.int8_linear(x_q, w_q, bias, float(scale_factor), apply_scale)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    avg_ms = sum(times) / len(times)
    tops = flops / 1e12 / (avg_ms / 1e3)
    return avg_ms, tops, out.detach()


def run_linear_benchmark():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required")

    device = torch.device("cuda")
    torch.manual_seed(0)

    x = torch.randn(
        BATCH, TOKENS, IN_FEATURES, dtype=torch.float16, device=device
    )
    w = torch.randn(
        OUT_FEATURES, IN_FEATURES, dtype=torch.float16, device=device
    )
    bias = torch.randn(OUT_FEATURES, dtype=torch.float16, device=device)

    flops = linear_flops(BATCH, TOKENS, IN_FEATURES, OUT_FEATURES)

    print("=== INT8 Linear Benchmark (FP16 vs INT8) ===")
    print(f"x shape: {[BATCH, TOKENS, IN_FEATURES]}")
    print(f"w shape: {[OUT_FEATURES, IN_FEATURES]}")

    fp16_ms, fp16_tflops, fp16_out = benchmark_fp16(x, w, bias, flops)
    print("FP16 Torch linear")
    print(f"  avg latency: {fp16_ms:.3f} ms")
    print(f"  throughput: {fp16_tflops:.2f} TFLOP/s")

    x_q, scale_x = quantize_to_int8(x)
    w_q, scale_w = quantize_to_int8(w)
    scale_x = float(scale_x.item())
    scale_w = float(scale_w.item())
    scale_product = scale_x * scale_w

    scenarios = [
        ("INT8 (raw accum)", 1.0, False),
        ("INT8 (scaled)", scale_product, True),
    ]

    for label, scale_factor, apply_scale in scenarios:
        int8_ms, int8_tops, int8_out = benchmark_int8(
            x_q,
            w_q,
            bias,
            scale_factor,
            apply_scale,
            flops,
        )
        print(label)
        print(f"  avg latency: {int8_ms:.3f} ms")
        print(f"  effective throughput: {int8_tops:.2f} TOPS")
        if apply_scale:
            print(
                f"  scales: Sx={scale_x:.3e}, Sw={scale_w:.3e}, eff={scale_factor:.3e}"
            )
        int8_out_fp32 = int8_out.float()
        diff = (int8_out_fp32 - fp16_out.float()).abs()
        max_err = diff.max().item()
        mean_err = diff.mean().item()
        rmse = diff.pow(2).mean().sqrt().item()
        print(
            f"  error: max={max_err:.3e}, mean={mean_err:.3e}, rmse={rmse:.3e}"
        )
        if int8_ms > 0:
            print(f"  Speedup (FP16 / INT8): {fp16_ms / int8_ms:.2f}x")


def main():
    run_linear_benchmark()


if __name__ == "__main__":
    main()
