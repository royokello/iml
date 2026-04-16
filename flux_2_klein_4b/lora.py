from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import load_file as safe_load_file

from utils.quant.double import dequantize_from_double_block, quantize_to_double_block
from utils.quant.linear import QuantizedLinear
from utils.quant.single import dequantize_from_single_block, quantize_to_single_block

_WEIGHT_SUFFIXES = {
    ".lora_A.weight": "a",
    ".lora_B.weight": "b",
    ".lora_down.weight": "a",
    ".lora_up.weight": "b",
}
_ALPHA_SUFFIXES = (".alpha", ".lora_alpha")
_RANK_SUFFIXES = (".rank",)
_LORA_DOWN_SUFFIX = ".lora_down.weight"
_LORA_UP_SUFFIX = ".lora_up.weight"
_DIRECT_FLUX_KEY_RE = re.compile(r"^lora_unet_(double|single)_blocks_(\d+)_(.+)$")
_FLUX_SIMPLE_KEY_MAP = {
    "lora_unet_img_in": "x_embedder",
    "lora_unet_txt_in": "context_embedder",
    "lora_unet_time_in_in_layer": "time_guidance_embed.timestep_embedder.linear_1",
    "lora_unet_time_in_out_layer": "time_guidance_embed.timestep_embedder.linear_2",
    "lora_unet_final_layer_linear": "proj_out",
}


@dataclass
class _LoraSpec:
    module_key: str
    a: torch.Tensor | None = None
    b: torch.Tensor | None = None
    alpha: float | None = None
    rank: int | None = None


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
    - `<module>.lora_down.weight` + `<module>.lora_up.weight`
    - optional `<module>.alpha`, `<module>.lora_alpha`, or `<module>.rank`

    When `strict` is false, unrelated checkpoint entries are ignored. Invalid
    LoRA entries still raise because they indicate a broken merge request.
    """
    modules = dict(transformer.named_modules())

    for checkpoint_path, strength in _iter_lora_sources(lora_path):
        state_dict = _load_lora_state_dict(checkpoint_path)
        state_dict = _convert_flux_unet_lora_state_dict(state_dict)
        specs = _normalize_lora_checkpoint(state_dict, strict=strict)

        for spec in specs:
            module_name, module = _resolve_target_module(spec.module_key, modules)
            delta = _build_lora_delta(spec, module_name, scale=scale * strength)
            _merge_lora_delta(module, delta, module_name)

    return transformer


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


def _convert_flux_unet_lora_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    if not any(key.startswith("lora_unet_") for key in state_dict):
        return state_dict

    remaining = dict(state_dict)
    converted: dict[str, Any] = {}

    direct_mappings = {
        "img_attn_proj": lambda i: (f"transformer_blocks.{i}.attn.to_out.0",),
        "txt_attn_proj": lambda i: (f"transformer_blocks.{i}.attn.to_add_out",),
        "img_mlp_0": lambda i: (f"transformer_blocks.{i}.ff.linear_in",),
        "img_mlp_2": lambda i: (f"transformer_blocks.{i}.ff.linear_out",),
        "txt_mlp_0": lambda i: (f"transformer_blocks.{i}.ff_context.linear_in",),
        "txt_mlp_2": lambda i: (f"transformer_blocks.{i}.ff_context.linear_out",),
        "linear1": lambda i: (f"single_transformer_blocks.{i}.attn.to_qkv_mlp_proj",),
        "linear2": lambda i: (f"single_transformer_blocks.{i}.attn.to_out",),
    }
    cat_mappings = {
        "img_attn_qkv": lambda i: (
            f"transformer_blocks.{i}.attn.to_q",
            f"transformer_blocks.{i}.attn.to_k",
            f"transformer_blocks.{i}.attn.to_v",
        ),
        "txt_attn_qkv": lambda i: (
            f"transformer_blocks.{i}.attn.add_q_proj",
            f"transformer_blocks.{i}.attn.add_k_proj",
            f"transformer_blocks.{i}.attn.add_v_proj",
        ),
    }

    for source_prefix, target_prefix in _FLUX_SIMPLE_KEY_MAP.items():
        _move_flux_lora_simple(remaining, converted, source_prefix, target_prefix)

    candidate_prefixes = {
        key.removesuffix(_LORA_DOWN_SUFFIX)
        for key in remaining
        if key.endswith(_LORA_DOWN_SUFFIX)
    }
    candidate_prefixes.update(
        key.removesuffix(_LORA_UP_SUFFIX)
        for key in remaining
        if key.endswith(_LORA_UP_SUFFIX)
    )
    candidate_prefixes.update(
        key.removesuffix(".alpha")
        for key in remaining
        if key.endswith(".alpha")
    )

    for key in sorted(candidate_prefixes):
        match = _DIRECT_FLUX_KEY_RE.match(key)
        if match is None:
            continue

        _, block_index_text, tail = match.groups()
        block_index = int(block_index_text)

        if tail in direct_mappings:
            _move_flux_lora_simple(remaining, converted, key, direct_mappings[tail](block_index)[0])
            continue

        if tail in cat_mappings:
            _move_flux_lora_cat(remaining, converted, key, cat_mappings[tail](block_index))

    converted.update(remaining)
    return converted


def _move_flux_lora_simple(
    source: dict[str, Any],
    destination: dict[str, Any],
    source_prefix: str,
    target_prefix: str,
) -> None:
    down_key = source_prefix + _LORA_DOWN_SUFFIX
    up_key = source_prefix + _LORA_UP_SUFFIX
    alpha_key = source_prefix + ".alpha"

    if down_key not in source and up_key not in source and alpha_key not in source:
        return

    down_weight = source.pop(down_key, None)
    up_weight = source.pop(up_key, None)
    alpha = source.pop(alpha_key, None)

    if down_weight is not None:
        if not isinstance(down_weight, torch.Tensor):
            raise ValueError(f"Unsupported checkpoint format for {down_key}: expected a tensor.")
        if alpha is not None:
            down_weight, up_weight = _scale_flux_lora_pair(source_prefix, down_weight, up_weight, alpha)
        destination[target_prefix + _LORA_DOWN_SUFFIX] = down_weight

    if up_weight is not None:
        if not isinstance(up_weight, torch.Tensor):
            raise ValueError(f"Unsupported checkpoint format for {up_key}: expected a tensor.")
        destination[target_prefix + _LORA_UP_SUFFIX] = up_weight


def _move_flux_lora_cat(
    source: dict[str, Any],
    destination: dict[str, Any],
    source_prefix: str,
    target_prefixes: tuple[str, ...],
) -> None:
    down_key = source_prefix + _LORA_DOWN_SUFFIX
    if down_key not in source:
        return

    down_weight = source.pop(down_key)
    if not isinstance(down_weight, torch.Tensor):
        raise ValueError(f"Unsupported checkpoint format for {down_key}: expected a tensor.")

    up_key = source_prefix + _LORA_UP_SUFFIX
    if up_key not in source:
        for target_prefix in target_prefixes:
            destination[target_prefix + _LORA_DOWN_SUFFIX] = down_weight
        alpha_key = source_prefix + ".alpha"
        if alpha_key in source:
            alpha = source.pop(alpha_key)
            for target_prefix in target_prefixes:
                destination[target_prefix + ".alpha"] = alpha
        return

    up_weight = source.pop(up_key)
    if not isinstance(up_weight, torch.Tensor):
        raise ValueError(f"Unsupported checkpoint format for {up_key}: expected a tensor.")

    alpha_key = source_prefix + ".alpha"
    alpha = source.pop(alpha_key, None)
    if alpha is not None:
        down_weight, up_weight = _scale_flux_lora_pair(source_prefix, down_weight, up_weight, alpha)

    split_weights = _split_flux_cat_lora(source_prefix, down_weight, up_weight, len(target_prefixes))
    for target_prefix, split_down, split_up in zip(target_prefixes, split_weights[0], split_weights[1]):
        destination[target_prefix + _LORA_DOWN_SUFFIX] = split_down
        destination[target_prefix + _LORA_UP_SUFFIX] = split_up


def _split_flux_cat_lora(
    source_prefix: str,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    num_splits: int,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    if num_splits <= 0:
        raise ValueError(f"Invalid Flux LoRA split count for {source_prefix}: {num_splits}")
    if up_weight.ndim != 2:
        raise ValueError(
            f"Unsupported Flux LoRA up tensor shape for {source_prefix}: expected 2D, got {tuple(up_weight.shape)}"
        )
    if down_weight.ndim != 2:
        raise ValueError(
            f"Unsupported Flux LoRA down tensor shape for {source_prefix}: expected 2D, got {tuple(down_weight.shape)}"
        )
    if up_weight.shape[0] % num_splits != 0:
        raise ValueError(
            f"Unsupported Flux fused LoRA shape for {source_prefix}: output dimension {up_weight.shape[0]} "
            f"is not divisible by {num_splits}"
        )

    dims = [up_weight.shape[0] // num_splits] * num_splits
    rank = down_weight.shape[0]

    is_sparse = False
    if rank % num_splits == 0:
        split_rank = rank // num_splits
        is_sparse = True
        offset = 0
        for j, dim in enumerate(dims):
            for k in range(num_splits):
                if j == k:
                    continue
                block = up_weight[offset : offset + dim, k * split_rank : (k + 1) * split_rank]
                is_sparse = bool(is_sparse and torch.all(block == 0))
            offset += dim

    if not is_sparse:
        down_parts = tuple(down_weight for _ in range(num_splits))
        up_parts = tuple(torch.split(up_weight, dims, dim=0))
        return down_parts, up_parts

    split_rank = rank // num_splits
    down_parts = tuple(torch.chunk(down_weight, num_splits, dim=0))
    up_parts_list: list[torch.Tensor] = []
    offset = 0
    for index, dim in enumerate(dims):
        up_parts_list.append(
            up_weight[offset : offset + dim, index * split_rank : (index + 1) * split_rank].contiguous()
        )
        offset += dim
    return down_parts, tuple(up_parts_list)


def _scale_flux_lora_pair(
    source_prefix: str,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor | None,
    alpha: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    alpha_value = _as_scalar(alpha, source_prefix + ".alpha", kind="alpha")
    rank = down_weight.shape[0]
    if rank <= 0:
        raise ValueError(f"Invalid LoRA rank for {source_prefix}: {rank}")

    scale = float(alpha_value) / float(rank)
    return down_weight * scale, up_weight


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

        for suffix in _ALPHA_SUFFIXES:
            if not key.endswith(suffix):
                continue
            module_key = key[: -len(suffix)]
            spec = specs.setdefault(module_key, _LoraSpec(module_key=module_key))
            if spec.alpha is not None:
                raise ValueError(f"Duplicate LoRA alpha for {module_key}: {key}")
            spec.alpha = _as_scalar(value, key, kind="alpha")
            handled = True
            break

        if handled:
            continue

        for suffix in _RANK_SUFFIXES:
            if not key.endswith(suffix):
                continue
            module_key = key[: -len(suffix)]
            spec = specs.setdefault(module_key, _LoraSpec(module_key=module_key))
            rank_value = int(_as_scalar(value, key, kind="rank"))
            if rank_value <= 0:
                raise ValueError(f"Invalid LoRA rank for {module_key}: {rank_value}")
            if spec.rank is not None and spec.rank != rank_value:
                raise ValueError(
                    f"Conflicting LoRA rank metadata for {module_key}: {spec.rank} vs {rank_value}"
                )
            spec.rank = rank_value
            handled = True
            break

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
                missing.append("A/down")
            if spec.b is None:
                missing.append("B/up")
            missing_text = ", ".join(missing)
            raise ValueError(f"Unsupported checkpoint format for {module_key}: missing LoRA tensors: {missing_text}")

        rank_from_a = spec.a.shape[0]
        rank_from_b = spec.b.shape[1]
        if rank_from_a != rank_from_b:
            raise ValueError(
                f"Shape mismatch for {module_key}: "
                f"A/down has rank {rank_from_a} but B/up expects rank {rank_from_b}"
            )

        inferred_rank = rank_from_a
        if inferred_rank <= 0:
            raise ValueError(f"Invalid LoRA rank for {module_key}: {inferred_rank}")

        if spec.rank is not None and spec.rank != inferred_rank:
            raise ValueError(
                f"Rank metadata mismatch for {module_key}: metadata says {spec.rank}, tensors imply {inferred_rank}"
            )

        spec.rank = inferred_rank
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

    expanded: list[str] = []
    for candidate in base_candidates:
        expanded.append(candidate)
        if not candidate.endswith(".0"):
            expanded.append(f"{candidate}.0")

    ordered: list[str] = []
    seen: set[str] = set()
    for candidate in expanded:
        if candidate not in seen:
            seen.add(candidate)
            ordered.append(candidate)
    return tuple(ordered)


def _build_lora_delta(spec: _LoraSpec, module_name: str, *, scale: float) -> torch.Tensor:
    assert spec.a is not None
    assert spec.b is not None
    assert spec.rank is not None
    assert spec.alpha is not None

    delta = torch.matmul(spec.b, spec.a)
    delta = delta.mul(float(scale) * float(spec.alpha) / float(spec.rank))
    if not torch.isfinite(delta).all():
        raise ValueError(f"Invalid LoRA delta for {module_name}: non-finite values after merge construction.")
    return delta


def _merge_lora_delta(module: nn.Module, delta: torch.Tensor, module_name: str) -> None:
    if isinstance(module, nn.Linear):
        _merge_linear_lora(module, delta, module_name)
        return
    if isinstance(module, QuantizedLinear):
        _merge_quantized_linear_lora(module, delta, module_name)
        return

    raise TypeError(
        f"Unsupported LoRA target module for {module_name}: expected nn.Linear or QuantizedLinear, "
        f"got {type(module).__name__}"
    )


def _is_supported_target_module(module: nn.Module) -> bool:
    return isinstance(module, (nn.Linear, QuantizedLinear))


def _merge_linear_lora(module: nn.Linear, delta: torch.Tensor, module_name: str) -> None:
    target_shape = tuple(module.weight.shape)
    if tuple(delta.shape) != target_shape:
        raise ValueError(
            f"Shape mismatch for {module_name}: LoRA delta shape {tuple(delta.shape)} "
            f"does not match weight shape {target_shape}"
        )

    with torch.no_grad():
        module.weight.add_(delta.to(device=module.weight.device, dtype=module.weight.dtype))


def _merge_quantized_linear_lora(module: QuantizedLinear, delta: torch.Tensor, module_name: str) -> None:
    target_shape = (module.out_features, module.in_features)
    if tuple(delta.shape) != target_shape:
        raise ValueError(
            f"Shape mismatch for {module_name}: LoRA delta shape {tuple(delta.shape)} "
            f"does not match weight shape {target_shape}"
        )

    device = module.weight.device

    if module.method == "single":
        dense_weight = dequantize_from_single_block(module.weight, module.scales).view(target_shape)
    elif module.method == "double":
        dense_weight = dequantize_from_double_block(module.weight, module.scales, module.super_scales).view(target_shape)
    else:
        raise ValueError(f"Unsupported quantization method for {module_name}: {module.method!r}")

    with torch.no_grad():
        dense_weight.add_(delta.to(device=device, dtype=torch.float16))
        if module.method == "single":
            qweight, qscales = quantize_to_single_block(dense_weight)
            module.weight.copy_(qweight.to(device=device, dtype=module.weight.dtype))
            module.scales.copy_(qscales.to(device=module.scales.device, dtype=module.scales.dtype))
        else:
            qweight, qscales, qsuper_scales = quantize_to_double_block(dense_weight)
            module.weight.copy_(qweight.to(device=device, dtype=module.weight.dtype))
            module.scales.copy_(qscales.to(device=module.scales.device, dtype=module.scales.dtype))
            module.super_scales.copy_(qsuper_scales.to(device=module.super_scales.device, dtype=module.super_scales.dtype))


__all__ = ["apply_lora"]
