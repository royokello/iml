import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

BATCH = 1
C_IN = 1280
C_OUT = 1280
H = 32
W = 32
K_H = 1
K_W = 1
STRIDE = (1, 1)
PADDING = (0, 0)
DILATION = (1, 1)

WARMUP = 10
ITERS = 50

_INT8_MODULE = None


def load_int8_extension():
    global _INT8_MODULE
    if _INT8_MODULE is None:
        repo_root = Path(__file__).resolve().parents[1]
        cu_1x1 = repo_root / "cuda" / "int8_conv2d_1x1.cu"
        cu_3x3 = repo_root / "cuda" / "int8_conv2d_3x3_im2col.cu"
        cpp_src = repo_root / "cuda" / "int8_conv2d_bindings.cpp"

        extra_ldflags = []
        cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        if os.name == "nt":
            if cuda_home:
                lib_dir = Path(cuda_home) / "lib" / "x64"
                extra_ldflags.append(f"/LIBPATH:{lib_dir}")
            extra_ldflags += ["cublasLt.lib", "cublas.lib"]
        else:
            extra_ldflags += ["-lcublasLt", "-lcublas"]

        _INT8_MODULE = load(
            name="int8_conv2d_im2col",
            sources=[str(cu_1x1), str(cu_3x3), str(cpp_src)],
            extra_cuda_cflags=["-O3"],
            extra_cflags=["-O3"],
            extra_ldflags=extra_ldflags,
        )
    return _INT8_MODULE


def conv_flops(batch, cout, hout, wout, cin_k) -> float:
    return float(batch) * cout * hout * wout * cin_k * 2.0


def quantize_to_int8(t: torch.Tensor):
    max_abs = t.abs().max()
    scale = max_abs / 127.0 + 1e-8
    q = torch.clamp((t / scale).round(), -128, 127).to(torch.int8)
    return q, scale


def benchmark_fp16(x, w, stride, padding, dilation, flops):
    for _ in range(WARMUP):
        F.conv2d(x, w, stride=stride, padding=padding, dilation=dilation)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    out = None
    for _ in range(ITERS):
        start.record()
        out = F.conv2d(x, w, stride=stride, padding=padding, dilation=dilation)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    avg_ms = sum(times) / len(times)
    tflops = flops / 1e12 / (avg_ms / 1e3)
    return avg_ms, tflops, out.detach()


def benchmark_int8(
    x_q,
    w_q,
    bias_fp16,
    scale_factor,
    apply_scale,
    stride,
    padding,
    dilation,
    flops,
    use_1x1_kernel,
):
    module = load_int8_extension()
    conv_fn = module.int8_conv2d_1x1 if use_1x1_kernel else module.int8_conv2d_3x3_im2col
    scale_tensor = torch.full(
        (w_q.size(0),),
        float(scale_factor),
        dtype=torch.float32,
        device=x_q.device,
    )

    for _ in range(WARMUP):
        conv_fn(
            x_q,
            w_q,
            bias_fp16,
            scale_tensor,
            apply_scale,
            stride[0],
            stride[1],
            padding[0],
            padding[1],
            dilation[0],
            dilation[1],
            1,
        )
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    out = None
    for _ in range(ITERS):
        start.record()
        out = conv_fn(
            x_q,
            w_q,
            bias_fp16,
            scale_tensor,
            apply_scale,
            stride[0],
            stride[1],
            padding[0],
            padding[1],
            dilation[0],
            dilation[1],
            1,
        )
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))

    avg_ms = sum(times) / len(times)
    tops = flops / 1e12 / (avg_ms / 1e3)
    return avg_ms, tops, out.detach()


def run_case(name, kh, kw, stride, pad, dilation):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required")

    device = torch.device("cuda")
    torch.manual_seed(0)

    x = torch.randn(BATCH, C_IN, H, W, dtype=torch.float16, device=device)
    w = torch.randn(C_OUT, C_IN, kh, kw, dtype=torch.float16, device=device)
    bias = torch.randn(C_OUT, dtype=torch.float16, device=device)

    hout = (H + 2 * pad[0] - dilation[0] * (kh - 1) - 1) // stride[0] + 1
    wout = (W + 2 * pad[1] - dilation[1] * (kw - 1) - 1) // stride[1] + 1
    flops = conv_flops(BATCH, C_OUT, hout, wout, C_IN * kh * kw)

    print(f"=== {name} Conv Benchmark (FP16 vs INT8) ===")
    print(f"X shape: {[BATCH, C_IN, H, W]}")
    print(f"W shape: {[C_OUT, C_IN, kh, kw]}")

    fp16_ms, fp16_tflops, fp16_out = benchmark_fp16(x, w, stride, pad, dilation, flops)
    print("FP16 PyTorch conv2d")
    print(f"  avg latency: {fp16_ms:.3f} ms")
    print(f"  throughput: {fp16_tflops:.2f} TFLOP/s")

    use_1x1 = (kh == 1 and kw == 1 and stride == (1, 1) and pad == (0, 0) and dilation == (1, 1))
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
            stride,
            pad,
            dilation,
            flops,
            use_1x1,
        )
        print(label)
        print(f"  avg latency: {int8_ms:.3f} ms")
        print(f"  effective throughput: {int8_tops:.2f} TOPS")
        if apply_scale:
            print(f"  scales: Sx={scale_x:.3e}, Sw={scale_w:.3e}, eff={scale_factor:.3e}")
        # Respect "raw" strictly: do NOT scale on host when apply_scale=False.
        # This will produce large errors versus FP16 because units differ.
        int8_out_fp32 = int8_out.float()
        with torch.no_grad():
            diff = (int8_out_fp32 - fp16_out.float()).abs()
            max_err = diff.max().item()
            mean_err = diff.mean().item()
            rmse = diff.pow(2).mean().sqrt().item()
        print(f"  error: max={max_err:.3e}, mean={mean_err:.3e}, rmse={rmse:.3e}")
        if int8_ms > 0:
            print(f"  Speedup (FP16 / INT8): {fp16_ms / int8_ms:.2f}x")


def main():
    run_case("1x1", 1, 1, STRIDE, PADDING, DILATION)
    print()
    run_case("3x3", 3, 3, (1, 1), (1, 1), (1, 1))


if __name__ == "__main__":
    main()
