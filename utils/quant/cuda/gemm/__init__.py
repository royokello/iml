from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

HIGH_BLOCK_SIZE = 32
SUB_BLOCK_SIZE = 16
SUPER_BLOCK_SIZE = 256
HALF_SUPER_BLOCK_SIZE = 128

# ── Tile configs (must match gemm_fused.cu) ──────────────────────────────

TILE_CONFIGS = {
    "balanced":   (64, 64, 128),
    "tall_m":     (128, 64, 128),
    "wide_n":     (64, 128, 128),
    "large":      (128, 128, 64),
    "small_m":    (32, 64, 128),
}


def _select_super_block_size(row_size: int) -> int:
    for sbs in (SUPER_BLOCK_SIZE, HALF_SUPER_BLOCK_SIZE):
        if row_size % sbs == 0:
            return sbs
    raise ValueError(
        f"in_features={row_size} not divisible by "
        f"{HALF_SUPER_BLOCK_SIZE} or {SUPER_BLOCK_SIZE}"
    )


# ── Module loading ───────────────────────────────────────────────────────

_CUDA_MODULE: Any = None

def _get_cuda_module():
    global _CUDA_MODULE
    if _CUDA_MODULE is not None:
        return _CUDA_MODULE
    module_name = "gemm_fused_cuda"
    cuda_dir = Path(__file__).resolve().parent

    try:
        _CUDA_MODULE = __import__(module_name)
        return _CUDA_MODULE
    except ImportError:
        pass

    suffixes = [".pyd", ".so"]
    for suffix in suffixes:
        matches = sorted(cuda_dir.glob(f"{module_name}*{suffix}"))
        if not matches:
            continue
        import importlib.util
        import sys
        spec = importlib.util.spec_from_file_location(module_name, matches[0])
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        _CUDA_MODULE = module
        return _CUDA_MODULE

    raise ModuleNotFoundError(
        f"Prebuilt module '{module_name}' not found. Build first with "
        "'python setup.py build_ext --inplace' in utils/quant/cuda/gemm."
    )


# ── Autotune cache ──────────────────────────────────────────────────────

AUTOTUNE_CACHE_PATH = Path(__file__).resolve().parent / ".autotune_cache.json"


def _bucket_m(m: int) -> str:
    if m <= 32:
        return "1-32"
    if m <= 256:
        return "33-256"
    if m <= 1024:
        return "257-1024"
    if m <= 2048:
        return "1025-2048"
    if m <= 4096:
        return "2049-4096"
    return "4097+"


def _load_autotune_cache() -> dict[str, str]:
    if AUTOTUNE_CACHE_PATH.exists():
        try:
            with open(AUTOTUNE_CACHE_PATH) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_autotune_cache(cache: dict[str, str]) -> None:
    AUTOTUNE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(AUTOTUNE_CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def _cache_key(m: int, n: int, k: int) -> str:
    return f"{_bucket_m(m)}_{n}_{k}"


def _lookup_best_config(m: int, n: int, k: int) -> tuple[int, int, int] | None:
    cache = _load_autotune_cache()
    key = _cache_key(m, n, k)
    name = cache.get(key)
    if name and name in TILE_CONFIGS:
        return TILE_CONFIGS[name]
    return None


def _store_best_config(m: int, n: int, k: int, config_name: str) -> None:
    cache = _load_autotune_cache()
    key = _cache_key(m, n, k)
    cache[key] = config_name
    _save_autotune_cache(cache)


# ── Main API ─────────────────────────────────────────────────────────────

def fused_gemm(
    q_act: torch.Tensor,
    bsums: torch.Tensor,
    act_scales: torch.Tensor,
    q_weight: torch.Tensor,
    sub_scales: torch.Tensor,
    sub_mins: torch.Tensor | None,
    super_scales: torch.Tensor | None,
    super_mins: torch.Tensor | None,
    out_features: int,
    weight_format: str,
    force_config: str | None = None,
) -> torch.Tensor:
    """Fused int8 activation × quantized weight GEMM with bsums correction.

    Args:
        q_act: int8 activations [M, in_features]
        bsums: int16 sub-block sums [M, num_sub_blocks]
        act_scales: float32 super-block scales [M, num_super_blocks]
        q_weight: packed quantized weights (format-dependent)
        sub_scales: sub-block scale metadata (format-dependent)
        sub_mins: sub-block min metadata (None for symmetric)
        super_scales: super-block scales (None for sym-high)
        super_mins: super-block mins (None for symmetric)
        out_features: output dimension N
        weight_format: one of "sym-high", "sym-med", "sym-low",
                       "aff-high", "aff-med", "aff-low"
        force_config: force a specific tile config name (for autotuning)

    Returns:
        float16 output tensor [M, out_features]
    """
    M, in_features = q_act.shape
    N = out_features
    K = in_features

    super_block_size = _select_super_block_size(in_features)

    output = torch.empty(
        (M, N), dtype=torch.float16, device=q_act.device
    )

    module = _get_cuda_module()

    # Determine tile config
    if force_config and force_config in TILE_CONFIGS:
        bm, bn, bk = TILE_CONFIGS[force_config]
    else:
        cached = _lookup_best_config(M, N, K)
        if cached:
            bm, bn, bk = cached
        else:
            bm, bn, bk = (0, 0, 0)  # let C++ select_launch pick default

    module.fused_gemm_fp16(
        q_act,
        bsums,
        act_scales,
        q_weight,
        sub_scales,
        sub_mins,
        super_scales,
        super_mins,
        output,
        M, N, K,
        super_block_size,
        weight_format,
        bm, bn, bk,
    )
    return output


# ── Autotune helper ──────────────────────────────────────────────────────

def autotune(
    q_act: torch.Tensor,
    bsums: torch.Tensor,
    act_scales: torch.Tensor,
    q_weight: torch.Tensor,
    sub_scales: torch.Tensor,
    sub_mins: torch.Tensor | None,
    super_scales: torch.Tensor | None,
    super_mins: torch.Tensor | None,
    out_features: int,
    weight_format: str,
    n_warmup: int = 10,
    n_iters: int = 30,
) -> str:
    """Benchmark all tile configs and cache the fastest one.

    Returns the name of the fastest config.
    """
    M = q_act.shape[0]
    K = q_act.shape[1]
    N = out_features

    # Single warmup across all configs (GPU reaches steady state once)
    for _ in range(n_warmup):
        _ = fused_gemm(q_act, bsums, act_scales, q_weight,
                       sub_scales, sub_mins, super_scales, super_mins,
                       out_features, weight_format)

    results: dict[str, float] = {}
    for name in TILE_CONFIGS:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_iters):
            _ = fused_gemm(q_act, bsums, act_scales, q_weight,
                           sub_scales, sub_mins, super_scales, super_mins,
                           out_features, weight_format, force_config=name)
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end) / n_iters  # ms per iteration
        results[name] = elapsed

    best = min(results, key=results.__getitem__)
    _store_best_config(M, N, K, best)
    return best


__all__ = ["fused_gemm", "autotune", "TILE_CONFIGS"]
