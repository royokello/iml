"""
Fused int8 x quantized-weight GEMM — correctness evaluator.

Tests all 6 packed formats against the Python dequantize + F.linear reference.
Requires CUDA. Outputs per-format/shape pass/fail table, summary line.

Usage: python -m utils.quant.cuda.gemm.evaluator
"""

import torch
import torch.nn.functional as F

from utils.quant.to.intermediate import quantize_to_intermediate
from utils.quant.to.symmetric import quantize_to_symmetric
from utils.quant.to.affine import quantize_to_affine
from utils.quant.fro.intermediate import dequantize_from_intermediate
from utils.quant.fro.symmetric import dequantize_from_symmetric
from utils.quant.fro.affine import dequantize_from_affine
from utils.quant.cuda.gemm import fused_gemm

# ── Constants ─────────────────────────────────────────────────────────────

SUPER_BLOCK_SIZE = 256
SUB_BLOCK_SIZE = 16

# sub_block_size per format (weight side)
FORMAT_SUB_SIZES = {
    "sym-high": 32,
    "sym-med": 16,
    "sym-low": 16,
    "aff-high": 32,
    "aff-med": 32,
    "aff-low": 16,
}

# ── Helpers ───────────────────────────────────────────────────────────────

def _is_compatible(K: int, fmt: str) -> bool:
    """Check if K is compatible with format's sub-block and super-block."""
    sub = FORMAT_SUB_SIZES[fmt]
    if K % sub != 0:
        return False
    # super-block check (sym-med, sym-low, all aff require K%128==0 or K%256==0)
    family, mode = fmt.split("-")
    if family == "sym" and mode != "high":
        if K % 128 != 0:
            return False
    if family == "aff":
        if K % 128 != 0:
            return False
    return True


def _quant_act(act: torch.Tensor, M: int, K: int):
    """Quantize activations and reshape for fused_gemm()."""
    q_blocks, bsums_blocks, act_scales = quantize_to_intermediate(act)
    sbs = SUPER_BLOCK_SIZE if K % SUPER_BLOCK_SIZE == 0 else K
    q_act = q_blocks.view(M, K)
    bsums = bsums_blocks.view(M, K // SUB_BLOCK_SIZE)
    act_scales_2d = act_scales.view(M, K // sbs)
    return q_act, bsums, act_scales_2d, q_blocks, bsums_blocks, act_scales


def _quant_weight(weight: torch.Tensor, fmt: str):
    """Quantize weight; return (q_weight, sub_scales, sub_mins, super_scales, super_mins)."""
    family, mode = fmt.split("-")
    if family == "sym":
        qw, ss, sps = quantize_to_symmetric(weight, mode=mode)
        return qw, ss, None, sps, None
    qw, ss, sm, sps, spm = quantize_to_affine(weight, mode=mode)
    return qw, ss, sm, sps, spm


def _dequant_weight(q_weight, sub_scales, sub_mins, super_scales, super_mins,
                     original_shape, fmt: str):
    """Dequantize weight for reference comparison."""
    family, mode = fmt.split("-")
    if family == "sym":
        return dequantize_from_symmetric(
            q_weight, sub_scales, super_scales, original_shape, mode=mode
        )
    return dequantize_from_affine(
        q_weight, sub_scales, sub_mins, super_scales, super_mins,
        original_shape, mode=mode
    )


# ── Single test ───────────────────────────────────────────────────────────

def test_one(M: int, N: int, K: int, fmt: str,
             atol: float = 1.0, rtol: float = 0.02):
    """Run one format/shape test. Returns (passed, max_error, msg)."""
    device = torch.device("cuda")

    # Random data
    act = torch.randn(M, K, device=device, dtype=torch.float16)
    weight = torch.randn(N, K, device=device, dtype=torch.float16)

    # Quantize
    q_act, bsums, act_scales_2d, q_blocks, bsums_blocks, act_scales = _quant_act(act, M, K)
    q_weight, sub_scales, sub_mins, super_scales, super_mins = _quant_weight(weight, fmt)

    # Reference: dequantize → F.linear (compute in fp32)
    w_float = _dequant_weight(q_weight, sub_scales, sub_mins,
                               super_scales, super_mins, weight.shape, fmt).float()
    act_float = dequantize_from_intermediate(q_blocks, act_scales, act.shape).float()
    ref = F.linear(act_float, w_float).cpu()

    # Kernel (output is fp16)
    out = fused_gemm(q_act, bsums, act_scales_2d,
                     q_weight, sub_scales, sub_mins,
                     super_scales, super_mins, N, fmt).float().cpu()

    # Compare in fp32 on CPU
    diff = (out - ref).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    passed = torch.allclose(out, ref, atol=atol, rtol=rtol)

    return passed, max_err, mean_err


# ── Test cases ────────────────────────────────────────────────────────────

TEST_CASES = [
    (4, 32, 256,  "small aligned"),
    (8, 64, 256,  "batch 8 aligned"),
    (2, 64, 256,  "batch 2"),
    (4, 33, 256,  "odd N"),
    (1, 128, 256, "single token"),
    (4, 128, 512, "mid size"),
    (64, 6144, 6144, "batch 64 double attn"),
    (4608, 6144, 6144, "full double attn"),
    (4608, 6144, 128,  "proj out"),
    (4608, 6144, 15360,"embed"),
    (4608, 6144, 18432,"qkv / mlp double"),
    (4608, 6144, 24576,"single attn"),
    (4608, 36864, 6144,"double attn input"),
    (4608, 55296, 6144,"single attn input"),
]

FORMATS = ["sym-high", "sym-med", "sym-low",
           "aff-high", "aff-med", "aff-low"]


def run_tests():
    if not torch.cuda.is_available():
        print("ERROR: CUDA required — evaluator must run on Windows with GPU")
        return

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print()

    header = f"{'Format':<12} {'Shape':<22} {'Result':<8} {'MaxErr':<10} {'MeanErr':<10}"
    print(header)
    print("-" * len(header))

    total, passed, skipped = 0, 0, 0
    for M, N, K, desc in TEST_CASES:
        for fmt in FORMATS:
            total += 1
            if not _is_compatible(K, fmt):
                print(f"{fmt:<12} {f'({M},{N},{K})':<22} {'SKIP':<8}")
                skipped += 1
                continue
            try:
                ok, max_err, mean_err = test_one(M, N, K, fmt)
                status = "PASS" if ok else "FAIL"
                if ok:
                    passed += 1
                print(f"{fmt:<12} {f'({M},{N},{K})':<22} {status:<8} {max_err:<10.4f} {mean_err:<10.6f}")
            except Exception as e:
                print(f"{fmt:<12} {f'({M},{N},{K})':<22} {'ERROR':<8} {str(e)[:60]}")

    run = total - skipped
    print(f"\nResults: {passed}/{run} passed ({skipped} skipped, {run - passed} failed)")


if __name__ == "__main__":
    run_tests()
