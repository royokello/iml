"""
Fused int8 x quantized-weight GEMM — benchmark + autotune.

Steps:
  1. Autotune all flux2 (M, N, K) combinations (5 tile configs, 30 iters each, deduped by M-bucket).
  2. Measure throughput (ms, GFLOPS) for each.
  3. Compare against fp16 cuBLAS baseline (F.linear).

Usage: python -m utils.quant.cuda.gemm.benchmark
"""

import time
import torch
import torch.nn.functional as F

from utils.quant.to.intermediate import quantize_to_intermediate
from utils.quant.to.symmetric import quantize_to_symmetric
from utils.quant.cuda.gemm import fused_gemm, autotune, TILE_CONFIGS, _cache_key, _load_autotune_cache

SUPER_BLOCK_SIZE = 256
SUB_BLOCK_SIZE = 16

# Flux 2 problem sizes: (M, N, K, label)
PROBLEM_SIZES = [
    (64,   6144,  6144,  "double_attn_small"),
    (256,  6144,  6144,  "double_attn_med"),
    (1024, 6144,  6144,  "double_attn_large"),
    (2048, 6144,  6144,  "double_attn_xl"),
    (4608, 6144,  6144,  "double_attn_full"),
    (4608, 6144,  128,   "proj_out"),
    (4608, 6144,  15360, "embed"),
    (4608, 6144,  18432, "qkv_mlp"),
    (4608, 6144,  24576, "single_attn"),
    (4608, 36864, 6144,  "double_attn_input"),
    (4608, 55296, 6144,  "single_attn_input"),
    (1,    6144,  6144,  "single_token"),
    (16,   6144,  6144,  "batch_16"),
    (128,  6144,  6144,  "batch_128"),
]

CUDA_EVENTS = True  # use cuda.Event for precise timing; fallback to time.perf_counter


def benchmark_fused(M, N, K, fmt, force_config=None, n_warmup=10, n_iters=100):
    """Time fused_gemm, return ms per iteration and effective GFLOPS."""
    device = torch.device("cuda")

    # Random data
    act = torch.randn(M, K, device=device, dtype=torch.float16)
    weight = torch.randn(N, K, device=device, dtype=torch.float16)

    # Quantize
    q_blocks, bsums_blocks, act_scales = quantize_to_intermediate(act)
    sbs = SUPER_BLOCK_SIZE if K % SUPER_BLOCK_SIZE == 0 else K
    q_act = q_blocks.view(M, K)
    bsums = bsums_blocks.view(M, K // SUB_BLOCK_SIZE)
    act_scales_2d = act_scales.view(M, K // sbs)

    family, mode = fmt.split("-")
    if family == "sym":
        q_weight, sub_scales, super_scales = quantize_to_symmetric(weight, mode=mode)
        sub_mins, super_mins = None, None
    else:
        res = quantize_to_affine(weight, mode=mode)
        q_weight, sub_scales, sub_mins, super_scales, super_mins = res

    # Warmup
    for _ in range(n_warmup):
        _ = fused_gemm(q_act, bsums, act_scales_2d, q_weight, sub_scales, sub_mins,
                       super_scales, super_mins, N, fmt, force_config=force_config)
    torch.cuda.synchronize()

    # Timed run
    if CUDA_EVENTS:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_iters):
            _ = fused_gemm(q_act, bsums, act_scales_2d, q_weight, sub_scales, sub_mins,
                           super_scales, super_mins, N, fmt, force_config=force_config)
        end.record()
        torch.cuda.synchronize()
        total_ms = start.elapsed_time(end)
    else:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = fused_gemm(q_act, bsums, act_scales_2d, q_weight, sub_scales, sub_mins,
                           super_scales, super_mins, N, fmt, force_config=force_config)
        torch.cuda.synchronize()
        total_ms = (time.perf_counter() - t0) * 1000

    ms_per_iter = total_ms / n_iters
    gflops = 2.0 * M * N * K / (ms_per_iter * 1e6)
    return ms_per_iter, gflops


def benchmark_cublas(M, N, K, n_warmup=10, n_iters=100):
    """Time fp16 F.linear as cuBLAS baseline."""
    device = torch.device("cuda")
    act = torch.randn(M, K, device=device, dtype=torch.float16)
    weight = torch.randn(N, K, device=device, dtype=torch.float16)

    for _ in range(n_warmup):
        _ = F.linear(act, weight)
    torch.cuda.synchronize()

    if CUDA_EVENTS:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_iters):
            _ = F.linear(act, weight)
        end.record()
        torch.cuda.synchronize()
        total_ms = start.elapsed_time(end)
    else:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = F.linear(act, weight)
        torch.cuda.synchronize()
        total_ms = (time.perf_counter() - t0) * 1000

    ms_per_iter = total_ms / n_iters
    gflops = 2.0 * M * N * K / (ms_per_iter * 1e6)
    return ms_per_iter, gflops


def run_benchmark():
    if not torch.cuda.is_available():
        print("ERROR: CUDA required")
        return

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"CUDA: {torch.version.cuda}")
    print()

    # ── Step 1: Autotune ──────────────────────────────────────────────
    print("=== Autotuning ===")
    cache = _load_autotune_cache()
    for M, N, K, label in PROBLEM_SIZES:
        key = _cache_key(M, N, K)
        if key in cache:
            print(f"  ({M:>5}, {N:>5}, {K:>5})  {label}  → cached: {cache[key]}")
            continue
        print(f"  Autotuning ({M:>5}, {N:>5}, {K:>5})  {label}")
        try:
            device = torch.device("cuda")
            act = torch.randn(M, K, device=device, dtype=torch.float16)
            weight = torch.randn(N, K, device=device, dtype=torch.float16)

            q_blocks, bsums_blocks, act_scales = quantize_to_intermediate(act)
            sbs = SUPER_BLOCK_SIZE if K % SUPER_BLOCK_SIZE == 0 else K
            q_act = q_blocks.view(M, K)
            bsums = bsums_blocks.view(M, K // SUB_BLOCK_SIZE)
            act_scales_2d = act_scales.view(M, K // sbs)

            q_weight, sub_scales, super_scales = quantize_to_symmetric(weight, mode="high")

            best = autotune(q_act, bsums, act_scales_2d, q_weight,
                            sub_scales, None, super_scales, None, N, "sym-high")
            print(f"    → best config: {best}")
        except Exception as e:
            print(f"    → ERROR: {e}")
    print()

    # ── Step 2: Benchmark table ───────────────────────────────────────
    print("=== Throughput (sym-high, autotuned config) ===")
    header = (f"{'Label':<22} {'M':>6} {'N':>6} {'K':>6} "
              f"{'Fused ms':>10} {'Fused GFLOPS':>14} "
              f"{'cuBLAS ms':>10} {'cuBLAS GFLOPS':>14} "
              f"{'Speedup':>8}")
    print(header)
    print("-" * len(header))

    for M, N, K, label in PROBLEM_SIZES:
        try:
            ms_f, gf_f = benchmark_fused(M, N, K, fmt="sym-high")
            ms_c, gf_c = benchmark_cublas(M, N, K)
            speedup = ms_c / ms_f if ms_f > 0 else float("inf")
            print(f"{label:<22} {M:>6} {N:>6} {K:>6} "
                  f"{ms_f:>10.3f} {gf_f:>14.1f} "
                  f"{ms_c:>10.3f} {gf_c:>14.1f} "
                  f"{speedup:>8.2f}x")
        except Exception as e:
            print(f"{label:<22} {M:>6} {N:>6} {K:>6}  ERROR: {e}")

    # ── Step 3: M sweep (fixed N=K=6144) ──────────────────────────────
    print("\n=== M sweep (N=K=6144, sym-high) ===")
    print(f"{'M':>8} {'ms':>10} {'GFLOPS':>12}")
    print("-" * 32)
    for M in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 4608]:
        try:
            ms, gf = benchmark_fused(M, 6144, 6144, fmt="sym-high", n_iters=50)
            print(f"{M:>8} {ms:>10.3f} {gf:>12.1f}")
        except Exception as e:
            print(f"{M:>8} ERROR: {e}")


if __name__ == "__main__":
    run_benchmark()
