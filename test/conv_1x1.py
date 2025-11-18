# import os
# from pathlib import Path

# import torch
# import torch.nn.functional as F
# from torch.utils.cpp_extension import load
# C_IN = 1280
# C_OUT = 1280
# HEIGHT = 64
# WIDTH = 64
# WARMUP = 64
# ITERS = 256

# _CONV_1X1_MODULE = None


# def load_conv_1x1_extension():
#     global _CONV_1X1_MODULE
#     if _CONV_1X1_MODULE is not None:
#         return _CONV_1X1_MODULE

#     repo_root = Path(__file__).resolve().parents[1]
#     cu_src = repo_root / "cuda" / "conv_1x1.cu"
#     cpp_src = repo_root / "cuda" / "bindings.cpp"

#     extra_ldflags: list[str] = []
#     cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
#     if os.name == "nt":
#         if cuda_home:
#             lib_dir = Path(cuda_home) / "lib" / "x64"
#             extra_ldflags.append(f"/LIBPATH:{lib_dir}")
#         extra_ldflags += ["cublasLt.lib", "cublas.lib"]
#     else:
#         extra_ldflags += ["-lcublasLt", "-lcublas"]

#     _CONV_1X1_MODULE = load(
#         name="conv_1x1_cuda_test",
#         sources=[str(cu_src), str(cpp_src)],
#         extra_cuda_cflags=["-O3"],
#         extra_cflags=["-O3"],
#         extra_ldflags=extra_ldflags,
#     )
#     return _CONV_1X1_MODULE


# def quantize_tensor_per_tensor(tensor: torch.Tensor):
#     max_abs = tensor.abs().amax()
#     scale = torch.clamp(max_abs / 127.0, min=1e-8).to(torch.float32)
#     quantized = torch.clamp((tensor / scale).round(), -128, 127).to(torch.int8)
#     return quantized, scale


# def quantize_weight_per_channel(weight: torch.Tensor):
#     max_abs = weight.abs().amax(dim=1)
#     scale = torch.clamp(max_abs / 127.0, min=1e-8).to(torch.float32)
#     quantized = torch.clamp((weight / scale[:, None]).round(), -128, 127).to(
#         torch.int8
#     )
#     return quantized, scale


# def conv_flops(cout: int, hout: int, wout: int, cin_k: int) -> float:
#     return float(cout) * hout * wout * cin_k * 2.0


# def benchmark_fp16(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor, flops: float):
#     for _ in range(WARMUP):
#         F.conv2d(x, w, bias)
#     torch.cuda.synchronize()

#     start = torch.cuda.Event(enable_timing=True)
#     end = torch.cuda.Event(enable_timing=True)
#     times = []
#     out = None
#     for _ in range(ITERS):
#         start.record()
#         out = F.conv2d(x, w, bias)
#         end.record()
#         end.synchronize()
#         times.append(start.elapsed_time(end))
#     avg_ms = sum(times) / len(times)
#     tflops = flops / 1e12 / (avg_ms / 1e3)
#     return avg_ms, tflops, out.detach()


# def benchmark_int8(x_q, w_q, bias, scale_factor, module, flops: float):
#     scale_tensor = scale_factor.contiguous()
#     for _ in range(WARMUP):
#         module.conv_1x1(x_q, w_q, scale_tensor, bias)
#     torch.cuda.synchronize()

#     start = torch.cuda.Event(enable_timing=True)
#     end = torch.cuda.Event(enable_timing=True)
#     times = []
#     out = None
#     for _ in range(ITERS):
#         start.record()
#         out = 
#         end.record()
#         end.synchronize()
#         times.append(start.elapsed_time(end))
#     avg_ms = sum(times) / len(times)
#     tops = flops / 1e12 / (avg_ms / 1e3)
#     return avg_ms, tops, out.detach()


# def run_conv_1x1_cuda_test():
#     if not torch.cuda.is_available():
#         raise RuntimeError("CUDA device required for conv_1x1_cuda test")

#     device = torch.device("cuda")
#     torch.manual_seed(0)

#     x_fp16 = torch.randn(C_IN, HEIGHT, WIDTH, dtype=torch.float16, device=device)
#     w_fp16 = torch.randn(C_OUT, C_IN, dtype=torch.float16, device=device)
#     bias_fp16 = torch.randn(C_OUT, dtype=torch.float16, device=device)

#     x_q, scale_x = quantize_tensor_per_tensor(x_fp16.to(torch.float32))
#     w_q, scale_w = quantize_weight_per_channel(w_fp16.to(torch.float32))
#     scale_combined = (scale_w * scale_x).contiguous()

#     module = load_conv_1x1_extension()
#     flops = conv_flops(C_OUT, HEIGHT, WIDTH, C_IN)

#     x_fp16_batch = x_fp16.unsqueeze(0)
#     w_fp16_kernel = w_fp16.view(C_OUT, C_IN, 1, 1)

#     fp16_ms, fp16_tflops, fp16_out = benchmark_fp16(x_fp16_batch, w_fp16_kernel, bias_fp16, flops)

#     print("FP16 PyTorch conv2d")
#     print(f"  avg latency: {fp16_ms:.3f} ms")
#     print(f"  throughput: {fp16_tflops:.2f} TFLOP/s")

#     bias_fp32 = bias_fp16.to(torch.float32)

#     int8_ms, int8_tops, output = benchmark_int8(
#         x_q, w_q, bias_fp32, scale_combined, module, flops
#     )

#     expected = fp16_out[0].float()

#     diff = output - expected
#     abs_diff = diff.abs()
#     max_err = abs_diff.max().item()
#     mean_err = abs_diff.mean().item()
#     rmse = diff.pow(2).mean().sqrt().item()

#     print(f"  avg latency: {int8_ms:.3f} ms")
#     print(f"  effective throughput: {int8_tops:.2f} TOPS")
#     print(f"  error: max={max_err:.3e}, mean={mean_err:.3e}, rmse={rmse:.3e}")
#     if int8_ms > 0:
#         print(f"  Speedup: {fp16_ms / int8_ms:.2f}x")


# def main():
#     run_conv_1x1_cuda_test()


# if __name__ == "__main__":
#     main()
