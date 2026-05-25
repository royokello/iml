from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import load_file as safe_load_file

from utils.quant.cuda.affine_high import dequantize_from_affine_high
from utils.quant.cuda.affine_low import dequantize_from_affine_low
from utils.quant.cuda.affine_med import dequantize_from_affine_med
from utils.quant.cuda.symmetric_high import dequantize_from_symmetric_high
from utils.quant.cuda.symmetric_low import dequantize_from_symmetric_low
from utils.quant.cuda.symmetric_med import dequantize_from_symmetric_med
from utils.quant.linear import QuantizedLinear
from utils.quant.to.affine import quantize_to_affine
from utils.quant.to.symmetric import quantize_to_symmetric
from utils.quant.validators import quant_method_family, quant_method_mode
from .model import TrainableLoraLinear

_LORA_A_SUFFIX = ".lora_A.weight"
_LORA_B_SUFFIX = ".lora_B.weight"
_ALPHA_SUFFIX = ".alpha"


def apply_lora(
    transformer: nn.Module,
    lora_path: Mapping[str | Path, float],
    scale: float = 1.0,
) -> nn.Module:
    """Merge a LoRA checkpoint into `transformer` in place.

    `lora_path` must be a mapping of `checkpoint_path -> strength`.
    Deltas are summed per target module and merged once.

    Supported tensor format inside each checkpoint:
    - `<module>.alpha`
    - `<module>.lora_A.weight` + `<module>.lora_B.weight`
    """
    modules = dict(transformer.named_modules())

    for checkpoint_path, strength in _normalize_lora_sources(lora_path):
        state_dict = safe_load_file(str(checkpoint_path), device="cpu")

        for module_key, lora_a, lora_b, alpha in _iter_lora_checkpoint(state_dict):
            module_name, module = _resolve_target_module(module_key, modules)
            delta = _build_lora_delta(lora_a, lora_b, alpha, module_name, scale=scale * strength)
            _merge_lora_delta(module, delta, module_name)

    return transformer


def load_checkpoint(transformer: torch.nn.Module, checkpoint_path: Path) -> None:
    checkpoint_state = safe_load_file(str(checkpoint_path), device="cpu")
    for module_name, child in transformer.named_modules():
        if not isinstance(child, TrainableLoraLinear):
            continue

        lora_a_key = f"{module_name}.lora_A.weight"
        lora_b_key = f"{module_name}.lora_B.weight"
        if lora_a_key not in checkpoint_state and lora_b_key not in checkpoint_state:
            child.lora_A.data.zero_()
            child.lora_B.data.zero_()
            continue
        if lora_a_key not in checkpoint_state or lora_b_key not in checkpoint_state:
            raise KeyError(f"Incomplete LoRA weights for {module_name} in checkpoint {checkpoint_path}")

        child.lora_A.data.copy_(
            checkpoint_state[lora_a_key].to(device=child.lora_A.device, dtype=child.lora_A.dtype)
        )
        child.lora_B.data.copy_(
            checkpoint_state[lora_b_key].to(device=child.lora_B.device, dtype=child.lora_B.dtype)
        )


def _normalize_lora_sources(
    lora_path: Mapping[str | Path, float],
) -> tuple[tuple[str | Path, float], ...]:
    sources: list[tuple[str | Path, float]] = []
    for path, strength in lora_path.items():
        if not isinstance(strength, (int, float)):
            raise TypeError(f"LoRA strength for {path} must be numeric, got {type(strength).__name__}")
        sources.append((path, float(strength)))
    return tuple(sources)


def _iter_lora_checkpoint(state_dict: dict[str, Any]) -> Iterator[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]]:
    keys = sorted(state_dict)
    if any("lora" not in key and "alpha" not in key for key in keys):
        raise ValueError("LoRA checkpoint contains non-LoRA tensor names.")
    if len(keys) % 3 != 0:
        raise ValueError("LoRA checkpoint must contain alpha, lora_A, and lora_B for each module.")

    for index in range(0, len(keys), 3):
        alpha_key, lora_a_key, lora_b_key = keys[index : index + 3]
        if not alpha_key.endswith(_ALPHA_SUFFIX):
            raise ValueError(f"Expected LoRA alpha tensor, got {alpha_key}")
        if not lora_a_key.endswith(_LORA_A_SUFFIX):
            raise ValueError(f"Expected LoRA A tensor, got {lora_a_key}")
        if not lora_b_key.endswith(_LORA_B_SUFFIX):
            raise ValueError(f"Expected LoRA B tensor, got {lora_b_key}")

        module_key = alpha_key[: -len(_ALPHA_SUFFIX)]
        if module_key != lora_a_key[: -len(_LORA_A_SUFFIX)] or module_key != lora_b_key[: -len(_LORA_B_SUFFIX)]:
            raise ValueError(f"Mismatched LoRA tensor group: {alpha_key}, {lora_a_key}, {lora_b_key}")

        alpha = state_dict[alpha_key].detach().to(dtype=torch.float32, device="cpu")
        lora_a = state_dict[lora_a_key].detach().to(dtype=torch.float32, device="cpu")
        lora_b = state_dict[lora_b_key].detach().to(dtype=torch.float32, device="cpu")

        rank_from_a = lora_a.shape[0]
        rank_from_b = lora_b.shape[1]
        if rank_from_a != rank_from_b:
            raise ValueError(
                f"Shape mismatch for {module_key}: "
                f"A has rank {rank_from_a} but B expects rank {rank_from_b}"
            )

        inferred_rank = rank_from_a
        if inferred_rank <= 0:
            raise ValueError(f"Invalid LoRA rank for {module_key}: {inferred_rank}")

        yield module_key, lora_a, lora_b, alpha


def _resolve_target_module(module_key: str, modules: dict[str, nn.Module]) -> tuple[str, nn.Module]:
    module = modules[module_key]
    if not isinstance(module, (nn.Linear, QuantizedLinear)):
        raise TypeError(
            f"Unsupported LoRA target module for {module_key}: "
            f"expected nn.Linear or QuantizedLinear, got {type(module).__name__}"
        )
    return module_key, module


def _build_lora_delta(
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    alpha: torch.Tensor,
    module_name: str,
    *,
    scale: float,
) -> torch.Tensor:
    rank = lora_a.shape[0]
    if rank <= 0:
        raise ValueError(f"Invalid LoRA rank for {module_name}: {rank}")

    delta = torch.matmul(lora_b, lora_a)
    delta = delta.mul(alpha / float(rank))
    delta = delta.mul(float(scale))
    return delta.contiguous()


def _merge_lora_delta(module: nn.Module, delta: torch.Tensor, module_name: str) -> None:
    if isinstance(module, nn.Linear):
        _merge_linear_delta(module, delta, module_name)
        return
    if isinstance(module, QuantizedLinear):
        _merge_quantized_linear_delta(module, delta, module_name)
        return
    raise TypeError(f"Unsupported LoRA target module for {module_name}: {type(module).__name__}")


def _merge_linear_delta(module: nn.Linear, delta: torch.Tensor, module_name: str) -> None:
    weight = module.weight
    expected_shape = tuple(weight.shape)
    if tuple(delta.shape) != expected_shape:
        raise ValueError(
            f"LoRA shape mismatch for {module_name}: delta {tuple(delta.shape)} vs weight {expected_shape}"
        )
    with torch.no_grad():
        weight.add_(delta.to(device=weight.device, dtype=weight.dtype))


def _merge_quantized_linear_delta(module: QuantizedLinear, delta: torch.Tensor, module_name: str) -> None:
    quant_method = getattr(module, "method", None)
    if quant_method is None:
        raise ValueError(f"QuantizedLinear target {module_name} is missing its quantization method.")
    family = quant_method_family(quant_method)
    mode = quant_method_mode(quant_method)

    module.to(device="cuda")

    quantized_weight = module.weight

    original_shape = (module.out_features, module.in_features)

    if family == "symmetric":
        if mode == "high":
            dequantize = dequantize_from_symmetric_high
        elif mode == "med":
            dequantize = dequantize_from_symmetric_med
        elif mode == "low":
            dequantize = dequantize_from_symmetric_low
        else:
            raise ValueError(f"Unsupported symmetric quantization mode for {module_name}: {mode}")
        if mode == "high":
            weight = dequantize(
                quantized_weight,
                module.sub_scales,
                original_shape=original_shape,
            ).to(dtype=torch.float32)
        else:
            weight = dequantize(
                quantized_weight,
                module.sub_scales,
                module.super_scales,
                original_shape=original_shape,
            ).to(dtype=torch.float32)
    elif family == "affine":
        if mode == "high":
            dequantize = dequantize_from_affine_high
        elif mode == "med":
            dequantize = dequantize_from_affine_med
        elif mode == "low":
            dequantize = dequantize_from_affine_low
        else:
            raise ValueError(f"Unsupported affine quantization mode for {module_name}: {mode}")
        weight = dequantize(
            quantized_weight,
            module.sub_scales,
            module.sub_mins,
            module.super_scales,
            module.super_mins,
            original_shape=original_shape,
        ).to(dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported quantized linear method for {module_name}: {quant_method}")

    expected_shape = tuple(weight.shape)
    if tuple(delta.shape) != expected_shape:
        raise ValueError(
            f"LoRA shape mismatch for {module_name}: delta {tuple(delta.shape)} vs weight {expected_shape}"
        )

    merged_weight = weight.add(delta.to(device="cuda", dtype=torch.float32))

    with torch.no_grad():
        if family == "symmetric":
            qweight, sub_scales, super_scales = quantize_to_symmetric(merged_weight, mode=mode)
            module.weight.copy_(qweight.to(device="cuda"))
            module.sub_scales.copy_(sub_scales.to(device="cuda"))
            if super_scales is not None:
                module.super_scales.copy_(super_scales.to(device="cuda"))
        else:
            qweight, sub_scales, sub_mins, super_scales, super_mins = quantize_to_affine(
                merged_weight,
                mode=mode,
            )
            module.weight.copy_(qweight.to(device="cuda"))
            module.sub_scales.copy_(sub_scales.to(device="cuda"))
            module.sub_mins.copy_(sub_mins.to(device="cuda"))
            module.super_scales.copy_(super_scales.to(device="cuda"))
            module.super_mins.copy_(super_mins.to(device="cuda"))

    module.to(device="cpu")


__all__ = ["apply_lora", "load_checkpoint"]
