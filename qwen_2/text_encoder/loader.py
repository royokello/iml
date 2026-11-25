from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLTextConfig,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLModel,
    Qwen3VLTextModel,
    Qwen3VLTextRotaryEmbedding,
    Qwen3VLVisionRotaryEmbedding,
)

from krea_2.text_encoder.embedding import QuantizedEmbedding
from qwen_2.config import parse_quant_method
from qwen_2.text_encoder.keys import _strip_model_key, _strip_text_key
from qwen_2.text_encoder.target import _build_qwen2_text_encoder_targets
from utils.quant.linear import QuantizedLinear
from utils.quant.name import convert_quant_ranks
from utils.quant.replace import replace_targeted_linear_modules, target_tensors_to_linear_names
from utils.safetensors_io import TensorFile
from utils.text import replace_hyphens_with_underscores

__all__ = ["load_qwen2_text_encoder"]

_PLAIN_METHODS = ("fp16", "fp32", "bf16")
_TORCH_DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}
_SKIP_TEXT = ("model.visual.", "lm_head.")
_SKIP_FULL = ("lm_head.",)


def _read_config(model_dir: Path) -> dict:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Text encoder config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _text_config_kwargs(full_config: dict) -> dict:
    text_config = dict(full_config["text_config"])
    rope_scaling = dict(text_config.pop("rope_scaling", None) or {})
    text_config["rope_parameters"] = {
        "rope_type": rope_scaling.get("rope_type", "default"),
        "rope_theta": text_config.get("rope_theta", 5000000),
        "mrope_section": rope_scaling.get("mrope_section", [24, 20, 20]),
        "mrope_interleaved": rope_scaling.get("mrope_interleaved", True),
    }
    return text_config


def _build_config(model_dir: Path, *, include_vision: bool) -> Qwen3VLConfig | Qwen3VLTextConfig:
    full_config = _read_config(model_dir)
    if full_config.get("model_type") != "qwen3_vl":
        raise ValueError(
            f"Unsupported text encoder model_type {full_config.get('model_type')!r}. Expected 'qwen3_vl'."
        )
    text_config = _text_config_kwargs(full_config)
    if not include_vision:
        return Qwen3VLTextConfig(**text_config)
    return Qwen3VLConfig(text_config=text_config, vision_config=dict(full_config["vision_config"]))


def _build_model(config: Qwen3VLConfig | Qwen3VLTextConfig, *, include_vision: bool) -> nn.Module:
    default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float16)
        with torch.device("meta"):
            model = Qwen3VLModel(config) if include_vision else Qwen3VLTextModel(config)
    finally:
        torch.set_default_dtype(default_dtype)

    text_config = config.text_config if include_vision else config
    text_rotary = model.language_model.rotary_emb if include_vision else model.rotary_emb
    replacement = Qwen3VLTextRotaryEmbedding(config=text_config)
    if include_vision:
        model.language_model.rotary_emb = replacement
        # The vision config has no `head_dim` field: the model derives it from
        # `hidden_size // num_heads`, so the rebuild must derive it the same way.
        vision_config = config.vision_config
        head_dim = vision_config.hidden_size // vision_config.num_heads
        model.visual.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)
    else:
        model.rotary_emb = replacement
    del text_rotary
    return model


def _checkpoint_keys(path: Path, key_transform) -> set[str]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return {key_transform(name) for name in handle.keys()}


def _cast_float_tensors_except_quantized(module: nn.Module, dtype: torch.dtype) -> None:
    if isinstance(
        module,
        (
            QuantizedLinear,
            QuantizedEmbedding,
            Qwen3VLTextRotaryEmbedding,
            Qwen3VLVisionRotaryEmbedding,
        ),
    ):
        return

    for parameter_name, parameter in module.named_parameters(recurse=False):
        if parameter is not None and parameter.is_floating_point() and parameter.dtype != dtype:
            module._parameters[parameter_name] = nn.Parameter(
                parameter.to(dtype=dtype),
                requires_grad=parameter.requires_grad,
            )

    for buffer_name, buffer in module.named_buffers(recurse=False):
        if buffer is not None and buffer.is_floating_point() and buffer.dtype != dtype:
            module._buffers[buffer_name] = buffer.to(dtype=dtype)

    for child in module.children():
        _cast_float_tensors_except_quantized(child, dtype)


def _load_weights(
    model: nn.Module,
    path: Path,
    *,
    key_transform,
    skip_prefixes: tuple[str, ...],
) -> dict[str, list[str]]:
    state_dict: dict[str, torch.Tensor] = {}
    handle = TensorFile(path)
    for name in handle.keys():
        if any(name.startswith(prefix) for prefix in skip_prefixes):
            continue
        state_dict[key_transform(name)] = handle.read(name)

    incompatible = model.load_state_dict(state_dict, strict=False, assign=True)
    del state_dict
    return {
        "unexpected_keys": list(incompatible.unexpected_keys),
        "missing_keys": list(incompatible.missing_keys),
    }


def _materialize_meta_tensors(model: nn.Module) -> None:
    unresolved = [
        name for name, tensor in model.named_parameters() if getattr(tensor, "is_meta", False)
    ]
    if unresolved:
        raise RuntimeError("Checkpoint load left parameters on meta: " + ", ".join(unresolved))


def _wrap_quantized_embedding(model: nn.Module, *, include_vision: bool, checkpoint_keys: set[str]) -> None:
    parent_path = "language_model" if include_vision else ""
    embedding_name = f"{parent_path}.embed_tokens" if parent_path else "embed_tokens"
    if f"{embedding_name}.sub_scales" not in checkpoint_keys:
        return
    parent = model.get_submodule(parent_path) if parent_path else model
    embedding = getattr(parent, "embed_tokens")
    if isinstance(embedding, QuantizedEmbedding):
        return
    parent.embed_tokens = QuantizedEmbedding.from_prequantized(embedding, method="q8")


def load_qwen2_text_encoder(
    path: str | Path,
    quant_method: str | None = None,
    *,
    include_vision: bool = False,
) -> nn.Module:
    """Load the Qwen Image 2.1 text encoder.

    Args:
        path: The `text_encoder` folder, or a file inside it.
        quant_method: A quantization method name, one of `fp16`, `bf16`, or
            `fp32` for the unquantized `model.safetensors`, or None for the
            same with fp16 weights.
        include_vision: Build the vision tower too. Text-to-image passes False;
            image editing passes True.

    Returns:
        The text encoder in eval mode with float weights in fp16.

    Raises:
        FileNotFoundError: If the config or the checkpoint is missing.
        ValueError: If `model_type` in `config.json` is not `qwen3_vl`.
        RuntimeError: If the checkpoint keys do not match the model.
    """
    model_path = Path(path).expanduser()
    model_dir = model_path.parent if model_path.is_file() else model_path
    quant_method, weight_dtype = parse_quant_method(quant_method)
    dtype = _TORCH_DTYPES[weight_dtype]

    config = _build_config(model_dir, include_vision=include_vision)
    key_transform = _strip_model_key if include_vision else _strip_text_key
    skip_prefixes = _SKIP_FULL if include_vision else _SKIP_TEXT

    if quant_method is not None:
        quant_method_key = replace_hyphens_with_underscores(quant_method)
        rank_1_method, rank_2_method, rank_3_method = convert_quant_ranks(quant_method, 3)
    else:
        quant_method_key = rank_1_method = rank_2_method = rank_3_method = None

    model = _build_model(config, include_vision=include_vision)

    if quant_method is None:
        plain_path = model_dir / "model.safetensors"
        if not plain_path.is_file():
            raise FileNotFoundError(f"Plain checkpoint not found: {plain_path}")
        incompatible = _load_weights(
            model,
            plain_path,
            key_transform=key_transform,
            skip_prefixes=skip_prefixes,
        )
        _check_incompatible(incompatible)
        _materialize_meta_tensors(model)
        _cast_float_tensors_except_quantized(model, dtype)
        model.eval()
        return model

    quantized_path = model_dir / f"{quant_method_key}_quant.safetensors"
    if not quantized_path.is_file():
        raise FileNotFoundError(
            f"Quantized checkpoint not found: {quantized_path}. Run the text encoder quantizer first."
        )
    print(f"    using saved quantized checkpoint: {quantized_path.name}")

    checkpoint_keys = _checkpoint_keys(quantized_path, key_transform)
    target_tensors = _build_qwen2_text_encoder_targets()
    for group, method in (
        ("rank_1", rank_1_method),
        ("rank_2", rank_2_method),
        ("rank_3", rank_3_method),
    ):
        if method in _PLAIN_METHODS:
            continue
        names = target_tensors_to_linear_names(target_tensors[group], key_transform=key_transform)
        replace_targeted_linear_modules(
            model,
            method=method,
            target_linear_names=names,
            quantize_weights=False,
            checkpoint_keys=checkpoint_keys,
        )

    _wrap_quantized_embedding(model, include_vision=include_vision, checkpoint_keys=checkpoint_keys)

    incompatible = _load_weights(
        model,
        quantized_path,
        key_transform=key_transform,
        skip_prefixes=skip_prefixes,
    )
    _check_incompatible(incompatible)
    _materialize_meta_tensors(model)
    _cast_float_tensors_except_quantized(model, dtype)
    model.eval()
    return model


def _check_incompatible(incompatible: dict[str, list[str]]) -> None:
    unexpected = incompatible.get("unexpected_keys") or []
    missing = incompatible.get("missing_keys") or []
    if unexpected or missing:
        raise RuntimeError(
            "Text encoder checkpoint does not match the model. "
            f"Unexpected keys: {', '.join(sorted(unexpected)) or 'none'}. "
            f"Missing keys: {', '.join(sorted(missing)) or 'none'}."
        )
