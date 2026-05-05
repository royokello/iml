from __future__ import annotations

import gc
import json
import time
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open

from gemma4.models.text.config import Gemma4TextConfig
from gemma4.models.text.main import Gemma4TextModel
from gemma4.models.text.rotary_embedding import Gemma4TextRotaryEmbedding
from gemma4.models.text.scaled_word_embedding import Gemma4TextScaledWordEmbedding
from gemma4.quant import _GEMMA4_QUANT_CONFIGS, _build_language_targets
from utils.quant.embedding import QuantizedTextScaledWordEmbedding
from utils.quant.replace import replace_targeted_linear_modules, target_tensors_to_linear_names

_MODEL_DIR = "gemma4"
_CHECKPOINT_NAME = "model.safetensors"
_LANGUAGE_PREFIX = "model.language_model."


def _resolve_model_dir(path: str | Path) -> Path:
    model_path = Path(path).expanduser().resolve()
    if model_path.is_file():
        return model_path.parent
    if (model_path / _MODEL_DIR).is_dir() and not (model_path / "config.json").is_file():
        return model_path / _MODEL_DIR
    return model_path


def _strip_language_prefix(name: str) -> str:
    if name.startswith(_LANGUAGE_PREFIX):
        return name[len(_LANGUAGE_PREFIX) :]
    return name


def _strip_language_prefixes(names: set[str]) -> set[str]:
    return {_strip_language_prefix(name) for name in names}


def _load_text_config(model_dir: Path) -> Gemma4TextConfig:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Gemma4 config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    text_config = config.get("text_config", config)
    return Gemma4TextConfig(**text_config)


def _list_checkpoint_keys(
    checkpoint_path: Path,
    *,
    key_transform: Callable[[str], str] | None = None,
) -> set[str]:
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        if key_transform is None:
            return set(handle.keys())
        return {key_transform(name) for name in handle.keys()}


def _load_safetensors_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    *,
    key_transform: Callable[[str], str] | None = None,
    inclusion_prefix: str | tuple[str, ...] | None = None,
    skip_mismatched_shapes: bool = False,
):
    load_start = time.perf_counter()
    state_dict: dict[str, torch.Tensor] = {}
    model_state = model.state_dict() if skip_mismatched_shapes else None
    skipped_prefix = 0
    skipped_unknown = 0
    skipped_shape = 0

    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        for name in handle.keys():
            if inclusion_prefix is not None and not name.startswith(inclusion_prefix):
                skipped_prefix += 1
                continue

            key = name if key_transform is None else key_transform(name)
            if model_state is not None:
                expected = model_state.get(key)
                if expected is None:
                    skipped_unknown += 1
                    continue

                tensor_shape = handle.get_slice(name).get_shape()
                if tuple(tensor_shape) != tuple(expected.shape):
                    skipped_shape += 1
                    continue

            state_dict[key] = handle.get_tensor(name)

    load_seconds = time.perf_counter() - load_start
    apply_start = time.perf_counter()
    incompatible = model.load_state_dict(state_dict, strict=False, assign=True)
    apply_seconds = time.perf_counter() - apply_start
    total_seconds = time.perf_counter() - load_start
    del state_dict
    gc.collect()

    print(
        f"    [1/1] "
        f"load={load_seconds:.3f}s "
        f"apply={apply_seconds:.3f}s "
        f"total={total_seconds:.3f}s"
    )
    if skipped_prefix or skipped_unknown or skipped_shape:
        print(
            "    skipped "
            f"prefix={skipped_prefix} "
            f"unknown={skipped_unknown} "
            f"shape={skipped_shape}"
        )
    return incompatible


def _materialize_meta_tensors(model: nn.Module, *, validate: bool = True) -> None:
    for module in model.modules():
        if not isinstance(module, Gemma4TextRotaryEmbedding):
            continue

        for layer_type, rope_init_fn in module.rope_init_fns.items():
            rope_type = module.rope_type[layer_type]
            rope_init_fn_kwargs = {"device": "cpu", "layer_type": layer_type}
            if layer_type == "full_attention" and rope_type == "proportional":
                rope_init_fn_kwargs["head_dim_key"] = "global_head_dim"

            inv_freq, _ = rope_init_fn(module.config, **rope_init_fn_kwargs)
            module.register_buffer(f"{layer_type}_inv_freq", inv_freq, persistent=False)
            module.register_buffer(f"{layer_type}_original_inv_freq", inv_freq.clone(), persistent=False)

    if not validate:
        return

    unresolved_parameters = [name for name, parameter in model.named_parameters() if getattr(parameter, "is_meta", False)]
    if unresolved_parameters:
        raise RuntimeError("Checkpoint load left parameters on meta: " + ", ".join(unresolved_parameters))

    unresolved_buffers = [name for name, buffer in model.named_buffers() if getattr(buffer, "is_meta", False)]
    if unresolved_buffers:
        raise RuntimeError("Checkpoint load left buffers on meta: " + ", ".join(unresolved_buffers))


def _build_target_linear_names_by_method(method: str) -> dict[str, tuple[str, ...]]:
    try:
        quant_config = _GEMMA4_QUANT_CONFIGS[method]
    except KeyError as exc:
        allowed = ", ".join(sorted(_GEMMA4_QUANT_CONFIGS))
        raise ValueError(f"Unsupported Gemma4 quantization method: {method!r}. Expected one of: {allowed}.") from exc

    targets_by_method = _build_language_targets(quant_config)
    embedding_target_names = set(_build_embedding_targets_by_method(method))
    return {
        quant_method: target_tensors_to_linear_names(
            (
                target_tensor
                for target_tensor in target_tensors
                if _strip_language_prefix(target_tensor).removesuffix(".weight") not in embedding_target_names
            ),
            key_transform=_strip_language_prefix,
        )
        for quant_method, target_tensors in targets_by_method.items()
    }


def _build_embedding_targets_by_method(method: str) -> dict[str, str]:
    try:
        quant_config = _GEMMA4_QUANT_CONFIGS[method]
    except KeyError as exc:
        allowed = ", ".join(sorted(_GEMMA4_QUANT_CONFIGS))
        raise ValueError(f"Unsupported Gemma4 quantization method: {method!r}. Expected one of: {allowed}.") from exc

    embedding_targets: dict[str, str] = {}
    for target_name, tensor_name in (
        ("token_embed", "embed_tokens"),
        ("per_layer_token_embed", "embed_tokens_per_layer"),
    ):
        quant_method = quant_config.get(target_name)
        if quant_method is not None:
            embedding_targets[tensor_name] = quant_method
    return embedding_targets


def _replace_quantized_embeddings(
    model: nn.Module,
    *,
    quant_method: str,
    quantize_weights: bool,
    checkpoint_keys: set[str] | None = None,
    embedding_targets: dict[str, str] | None = None,
) -> None:
    if embedding_targets is None:
        embedding_targets = _build_embedding_targets_by_method(quant_method)
    if checkpoint_keys is not None:
        checkpoint_keys = _strip_language_prefixes(checkpoint_keys)

    for module_path, tensor_method in embedding_targets.items():
        parent = model
        parts = module_path.split(".")
        module_name = parts[-1]

        for part in parts[:-1]:
            if not hasattr(parent, part):
                print(f"Warning: Could not find parent module {part} in path {module_path}")
                break
            parent = getattr(parent, part)
        else:
            if not hasattr(parent, module_name):
                print(f"Warning: Module {module_path} not found in model")
                continue

            original_embedding = getattr(parent, module_name)
            if not isinstance(original_embedding, (nn.Embedding, Gemma4TextScaledWordEmbedding)):
                continue

            if checkpoint_keys is not None and f"{module_path}.sub_scales" not in checkpoint_keys:
                continue

            if quantize_weights:
                quant_embedding = QuantizedTextScaledWordEmbedding(
                    original_embedding,
                    method=tensor_method,
                )
            else:
                quant_embedding = QuantizedTextScaledWordEmbedding.from_prequantized(
                    original_embedding,
                    method=tensor_method,
                )

            setattr(parent, module_name, quant_embedding)


def _replace_quantized_modules(
    model: nn.Module,
    *,
    quant_method: str,
    quantize_weights: bool,
    checkpoint_keys: set[str] | None = None,
) -> None:
    embedding_targets = _build_embedding_targets_by_method(quant_method)
    _replace_quantized_embeddings(
        model,
        quant_method=quant_method,
        quantize_weights=quantize_weights,
        checkpoint_keys=checkpoint_keys,
        embedding_targets=embedding_targets,
    )

    for tensor_method, target_linear_names in _build_target_linear_names_by_method(quant_method).items():
        replace_targeted_linear_modules(
            model,
            method=tensor_method,
            target_linear_names=target_linear_names,
            quantize_weights=quantize_weights,
            checkpoint_keys=checkpoint_keys,
        )


def _unsupported_prequantized_targets(model: nn.Module, checkpoint_keys: set[str], quant_method: str) -> list[str]:
    checkpoint_keys = _strip_language_prefixes(checkpoint_keys)
    linear_names = {name for name, module in model.named_modules() if isinstance(module, nn.Linear)}
    unsupported: list[str] = []
    for target_linear_names in _build_target_linear_names_by_method(quant_method).values():
        for name in target_linear_names:
            if name not in linear_names and f"{name}.sub_scales" in checkpoint_keys:
                unsupported.append(name)

    embedding_names = {
        name for name, module in model.named_modules() if isinstance(module, (nn.Embedding, Gemma4TextScaledWordEmbedding))
    }
    for name in _build_embedding_targets_by_method(quant_method):
        if name not in embedding_names and f"{name}.sub_scales" in checkpoint_keys:
            unsupported.append(name)

    return unsupported


def load_gemma4_text_model(
    path: str | Path,
    *,
    quant_method: str | None = None,
) -> Gemma4TextModel:
    model_dir = _resolve_model_dir(path)
    config = _load_text_config(model_dir)

    default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float16)
        config.dtype = torch.float16
        with torch.device("meta"):
            model = Gemma4TextModel(config)
    finally:
        torch.set_default_dtype(default_dtype)

    checkpoint_path = model_dir / _CHECKPOINT_NAME
    quantized_checkpoint_path = (
        None
        if quant_method is None
        else model_dir / f"language_{quant_method.strip().lower()}_quant.safetensors"
    )

    if quant_method is None or quantized_checkpoint_path is None or not quantized_checkpoint_path.is_file():
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Gemma4 checkpoint not found: {checkpoint_path}")

        _load_safetensors_checkpoint(
            model,
            checkpoint_path,
            key_transform=_strip_language_prefix,
            inclusion_prefix=_LANGUAGE_PREFIX,
            skip_mismatched_shapes=True,
        )
        _materialize_meta_tensors(model)
        if quant_method is not None:
            _replace_quantized_modules(
                model,
                quant_method=quant_method.strip().lower(),
                quantize_weights=True,
            )
    else:
        quant_method = quant_method.strip().lower()
        checkpoint_keys = _list_checkpoint_keys(quantized_checkpoint_path, key_transform=_strip_language_prefix)
        unsupported_targets = _unsupported_prequantized_targets(model, checkpoint_keys, quant_method)

        if unsupported_targets:
            if not checkpoint_path.is_file():
                unsupported = ", ".join(sorted(unsupported_targets))
                raise RuntimeError(
                    "Saved Gemma4 quant checkpoint contains quantized non-linear targets that cannot be "
                    f"materialized without the original checkpoint: {unsupported}. "
                    f"Expected original checkpoint at {checkpoint_path}."
                )

            _load_safetensors_checkpoint(
                model,
                checkpoint_path,
                key_transform=_strip_language_prefix,
                inclusion_prefix=_LANGUAGE_PREFIX,
                skip_mismatched_shapes=True,
            )
            _materialize_meta_tensors(model)

        _replace_quantized_modules(
            model,
            quant_method=quant_method,
            quantize_weights=False,
            checkpoint_keys=checkpoint_keys,
        )
        _load_safetensors_checkpoint(
            model,
            quantized_checkpoint_path,
            key_transform=_strip_language_prefix,
            skip_mismatched_shapes=True,
        )
        _materialize_meta_tensors(model)

    model.eval()
    return model


load_gemma4_language_model = load_gemma4_text_model


__all__ = ["load_gemma4_language_model", "load_gemma4_text_model"]
