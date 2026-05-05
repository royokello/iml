from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
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

_WEIGHT_SUFFIXES = {
    ".lora_A.weight": "a",
    ".lora_B.weight": "b",
}
_ALPHA_SUFFIX = ".alpha"


@dataclass
class _LoraSpec:
    module_key: str
    a: torch.Tensor | None = None
    b: torch.Tensor | None = None
    alpha: float | None = None


def apply_lora(
    transformer: nn.Module,
    lora_path: Mapping[str | Path, float],
    scale: float = 1.0,
    strict: bool = False,
) -> nn.Module:
    """Merge a LoRA checkpoint into `transformer` in place.

    `lora_path` must be a mapping of `checkpoint_path -> strength`.
    Deltas are summed per target module and merged once.

    Supported tensor formats inside each checkpoint:
    - `<module>.lora_A.weight` + `<module>.lora_B.weight`
    - optional `<module>.alpha`

    When `strict` is false, unrelated checkpoint entries are ignored. Invalid
    LoRA entries still raise because they indicate a broken merge request.
    """
    modules = dict(transformer.named_modules())

    for checkpoint_path, strength in _iter_lora_sources(lora_path):
        state_dict = _load_lora_state_dict(checkpoint_path)
        specs = _normalize_lora_checkpoint(state_dict, strict=strict)

        for spec in specs:
            module_name, module = _resolve_target_module(spec.module_key, modules)
            delta = _build_lora_delta(spec, module_name, scale=scale * strength)
            _merge_lora_delta(module, delta, module_name)

    return transformer


def load_checkpoint(transformer: torch.nn.Module, checkpoint_path: Path) -> None:
    checkpoint_state = safe_load_file(str(checkpoint_path), device="cpu")
    for module_name, child in transformer.named_modules():
        if not isinstance(child, TrainableLoraLinear):
            continue

        lora_a_key = f"{module_name}.lora_A.weight"
        lora_b_key = f"{module_name}.lora_B.weight"
        if lora_a_key not in checkpoint_state or lora_b_key not in checkpoint_state:
            raise KeyError(f"Missing LoRA weights for {module_name} in checkpoint {checkpoint_path}")

        child.lora_A.data.copy_(
            checkpoint_state[lora_a_key].to(device=child.lora_A.device, dtype=child.lora_A.dtype)
        )
        child.lora_B.data.copy_(
            checkpoint_state[lora_b_key].to(device=child.lora_B.device, dtype=child.lora_B.dtype)
        )


def _iter_lora_sources(
    lora_path: Mapping[str | Path, float],
) -> tuple[tuple[str | Path, float], ...]:
    if not isinstance(lora_path, Mapping):
        raise TypeError("lora_path must be a mapping of checkpoint path to strength.")
    if not lora_path:
        raise ValueError("No LoRA checkpoints provided.")

    sources: list[tuple[str | Path, float]] = []
    for path, strength in lora_path.items():
        if not isinstance(strength, (int, float)):
            raise TypeError(f"LoRA strength for {path} must be numeric, got {type(strength).__name__}")
        sources.append((path, float(strength)))
    return tuple(sources)


def _load_lora_state_dict(lora_path: str | Path) -> dict[str, Any]:
    path = Path(lora_path)
    if not path.is_file():
        raise FileNotFoundError(f"LoRA checkpoint not found: {path}")
    if path.suffix.lower() != ".safetensors":
        raise ValueError(f"Unsupported LoRA checkpoint extension for {path}: expected .safetensors")

    state_dict = safe_load_file(str(path), device="cpu")

    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format in {path}: expected a tensor state dict.")

    return state_dict


def _normalize_lora_checkpoint(state_dict: dict[str, Any], *, strict: bool) -> list[_LoraSpec]:
    specs: dict[str, _LoraSpec] = {}
    ignored_keys: list[str] = []
    ignored_tensor_keys: list[str] = []

    for key, value in state_dict.items():
        handled = False

        for suffix, attr in _WEIGHT_SUFFIXES.items():
            if not key.endswith(suffix):
                continue
            module_key = key[: -len(suffix)]
            spec = specs.setdefault(module_key, _LoraSpec(module_key=module_key))
            tensor = _as_2d_lora_tensor(value, key)
            current = getattr(spec, attr)
            if current is not None:
                raise ValueError(f"Duplicate LoRA tensor for {module_key}: {key}")
            setattr(spec, attr, tensor)
            handled = True
            break

        if handled:
            continue

        if key.endswith(_ALPHA_SUFFIX):
            module_key = key[: -len(_ALPHA_SUFFIX)]
            spec = specs.setdefault(module_key, _LoraSpec(module_key=module_key))
            if spec.alpha is not None:
                raise ValueError(f"Duplicate LoRA alpha for {module_key}: {key}")
            spec.alpha = _as_scalar(value, key, kind="alpha")
            handled = True

        if not handled:
            ignored_keys.append(key)
            if isinstance(value, torch.Tensor):
                ignored_tensor_keys.append(key)

    if not specs:
        suffixes = ", ".join(sorted(_WEIGHT_SUFFIXES))
        raise ValueError(
            "Unsupported checkpoint format: no LoRA weights found. "
            f"Expected keys ending with one of: {suffixes}"
        )

    if strict and ignored_keys:
        preview = ", ".join(sorted(ignored_keys)[:8])
        if len(ignored_keys) > 8:
            preview += ", ..."
        raise ValueError(f"Unsupported checkpoint format: unrecognized keys: {preview}")

    if ignored_tensor_keys:
        preview = ", ".join(sorted(ignored_tensor_keys)[:8])
        if len(ignored_tensor_keys) > 8:
            preview += ", ..."
        raise ValueError(
            "LoRA checkpoint contains tensor entries that were not merged: "
            f"{preview}"
        )

    normalized_specs: list[_LoraSpec] = []
    for module_key, spec in sorted(specs.items()):
        if spec.a is None or spec.b is None:
            missing = []
            if spec.a is None:
                missing.append("A")
            if spec.b is None:
                missing.append("B")
            missing_text = ", ".join(missing)
            raise ValueError(f"Unsupported checkpoint format for {module_key}: missing LoRA tensors: {missing_text}")

        rank_from_a = spec.a.shape[0]
        rank_from_b = spec.b.shape[1]
        if rank_from_a != rank_from_b:
            raise ValueError(
                f"Shape mismatch for {module_key}: "
                f"A has rank {rank_from_a} but B expects rank {rank_from_b}"
            )

        inferred_rank = rank_from_a
        if inferred_rank <= 0:
            raise ValueError(f"Invalid LoRA rank for {module_key}: {inferred_rank}")

        spec.alpha = float(inferred_rank if spec.alpha is None else spec.alpha)
        normalized_specs.append(spec)

    return normalized_specs


def _as_2d_lora_tensor(value: Any, key: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"Unsupported checkpoint format for {key}: expected a tensor.")

    tensor = value.detach().to(dtype=torch.float32, device="cpu")
    if tensor.ndim == 2:
        return tensor
    if tensor.ndim == 4 and tensor.shape[2:] == (1, 1):
        return tensor[:, :, 0, 0].contiguous()

    raise ValueError(
        f"Unsupported LoRA tensor shape for {key}: expected 2D or 4D 1x1, got {tuple(tensor.shape)}"
    )


def _as_scalar(value: Any, key: str, *, kind: str) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"Unsupported {kind} tensor for {key}: expected a scalar, got shape {tuple(value.shape)}")
        return float(value.detach().cpu().item())
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"Unsupported {kind} value for {key}: expected a scalar tensor or number.")


def _resolve_target_module(module_key: str, modules: dict[str, nn.Module]) -> tuple[str, nn.Module]:
    candidates = list(_candidate_module_keys(module_key))
    unsupported_matches: dict[str, str] = {}

    for candidate in candidates:
        module = modules.get(candidate)
        if module is None:
            continue
        if _is_supported_target_module(module):
            return candidate, module
        unsupported_matches[candidate] = type(module).__name__

    ambiguous_matches: dict[str, list[str]] = {}
    for candidate in candidates:
        matches = [
            name
            for name, module in modules.items()
            if (name == candidate or name.endswith(f".{candidate}")) and _is_supported_target_module(module)
        ]
        if len(matches) == 1:
            match_name = matches[0]
            return match_name, modules[match_name]
        if len(matches) > 1:
            ambiguous_matches[candidate] = sorted(matches)

    if ambiguous_matches:
        candidate, matches = next(iter(ambiguous_matches.items()))
        preview = ", ".join(matches[:6])
        if len(matches) > 6:
            preview += ", ..."
        raise KeyError(f"Ambiguous LoRA target for {module_key} via {candidate}: {preview}")

    if unsupported_matches:
        candidate, module_type = next(iter(unsupported_matches.items()))
        raise TypeError(
            f"Unsupported LoRA target module for {module_key} via {candidate}: "
            f"expected nn.Linear or QuantizedLinear, got {module_type}"
        )

    raise KeyError(f"Missing module match for LoRA target: {module_key}")


def _candidate_module_keys(module_key: str) -> tuple[str, ...]:
    base_candidates: list[str] = []
    current = module_key
    while current:
        base_candidates.append(current)
        if "." not in current:
            break
        current = current.split(".", 1)[1]

    candidates: list[str] = []
    for candidate in base_candidates:
        candidates.append(candidate)
        normalized = ".".join(part for part in candidate.split(".") if not part.isdigit())
        if normalized != candidate:
            candidates.append(normalized)

    return tuple(dict.fromkeys(candidates))


def _is_supported_target_module(module: nn.Module) -> bool:
    return isinstance(module, (nn.Linear, QuantizedLinear))


def _build_lora_delta(spec: _LoraSpec, module_name: str, *, scale: float) -> torch.Tensor:
    assert spec.a is not None
    assert spec.b is not None
    assert spec.alpha is not None

    rank = spec.a.shape[0]
    if rank <= 0:
        raise ValueError(f"Invalid LoRA rank for {module_name}: {rank}")

    delta = torch.matmul(spec.b, spec.a)
    delta = delta.mul(float(spec.alpha) / float(rank))
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

    original_device = module.weight.device
    if original_device.type == "cuda":
        merge_device = original_device
    else:
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Cannot merge LoRA into quantized module {module_name}: CUDA is required for dequantization."
            )
        merge_device = torch.device("cuda")
        module.to(device=merge_device)

    quantized_weight = module.weight

    original_shape = (module.out_features, module.in_features)

    if family == "symmetric":
        if mode == "high":
            dequantize = dequantize_from_symmetric_high
        elif mode == "med":
            dequantize = dequantize_from_symmetric_med
        else:
            dequantize = dequantize_from_symmetric_low
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
        else:
            dequantize = dequantize_from_affine_low
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

    merged_weight = weight.add(delta.to(device=merge_device, dtype=torch.float32))

    with torch.no_grad():
        if family == "symmetric":
            qweight, sub_scales, super_scales = quantize_to_symmetric(merged_weight, mode=mode)
            module.weight.copy_(qweight.to(device=merge_device))
            module.sub_scales.copy_(sub_scales.to(device=merge_device))
            if super_scales is not None:
                module.super_scales.copy_(super_scales.to(device=merge_device))
        else:
            qweight, sub_scales, sub_mins, super_scales, super_mins = quantize_to_affine(
                merged_weight,
                mode=mode,
            )
            module.weight.copy_(qweight.to(device=merge_device))
            module.sub_scales.copy_(sub_scales.to(device=merge_device))
            module.sub_mins.copy_(sub_mins.to(device=merge_device))
            module.super_scales.copy_(super_scales.to(device=merge_device))
            module.super_mins.copy_(super_mins.to(device=merge_device))

    if original_device != merge_device:
        module.to(device=original_device)


__all__ = ["apply_lora", "load_checkpoint"]
