from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from flux2.models.text_encoder.model import Qwen3Model
from flux2.models.text_encoder.embedding import Qwen3RotaryEmbedding
from flux2.text_encoder.loader import _materialize_meta_tensors
from krea_2.config import TEXT_ENCODER_LAYERS
from krea_2.text_encoder.embedding import QuantizedEmbedding
from krea_2.text_encoder.target import _build_krea2_text_encoder_targets
from utils.loaders.single import load_local_single_checkpoint
from utils.quant.linear import QuantizedLinear
from utils.quant.name import convert_quant_ranks
from utils.quant.replace import replace_targeted_linear_modules, target_tensors_to_linear_names
from utils.text import replace_hyphens_with_underscores

__all__ = ["load_krea2_text_encoder"]

_LANGUAGE_PREFIX = "language_model."
_CHECKED_CONFIG_KEYS = (
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "rms_norm_eps",
)
_PLAIN_METHODS = ("fp16", "fp32", "bf16")


def _strip_language_prefix(name: str) -> str:
    if name.startswith(_LANGUAGE_PREFIX):
        return name[len(_LANGUAGE_PREFIX):]
    if name.startswith("model."):
        return name[len("model."):]
    return name


def _build_config(model_dir: Path) -> Qwen3Config:
    with (model_dir / "config.json").open("r", encoding="utf-8") as handle:
        full_config = json.load(handle)

    if full_config.get("model_type") != "qwen3_vl":
        raise ValueError(
            f"Unsupported text encoder model_type {full_config.get('model_type')!r}. Expected 'qwen3_vl'."
        )

    text_config = dict(full_config["text_config"])
    # transformers 5 moves rope_theta into rope_parameters.
    rope_parameters = text_config.get("rope_parameters") or {}
    rope_theta = rope_parameters.get("rope_theta", text_config.get("rope_theta"))
    if rope_theta is None:
        raise ValueError("text_config has no rope_theta, expected 'rope_parameters.rope_theta'.")
    init_parameters = inspect.signature(Qwen3Config.__init__).parameters
    init_kwargs = {key: value for key, value in text_config.items() if key in init_parameters and key != "self"}
    # A text-only input has three equal mRoPE axes, so plain 1D RoPE is exact.
    init_kwargs.pop("rope_scaling", None)
    init_kwargs["rope_parameters"] = {"rope_type": "default", "rope_theta": rope_theta}
    init_kwargs["num_hidden_layers"] = TEXT_ENCODER_LAYERS

    config = Qwen3Config(**init_kwargs)
    for key in _CHECKED_CONFIG_KEYS:
        if getattr(config, key) != text_config[key]:
            raise RuntimeError(
                f"Text encoder config key {key!r} is {getattr(config, key)!r}, expected {text_config[key]!r}."
            )
    config.dtype = torch.float16
    return config


def _list_checkpoint_keys(checkpoint_path: Path) -> set[str]:
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        return {_strip_language_prefix(name) for name in handle.keys()}


def _shard_paths(model_dir: Path) -> list[Path]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open("r", encoding="utf-8") as handle:
            weight_map = json.load(handle)["weight_map"]
        return [model_dir / name for name in sorted(set(weight_map.values()))]
    single_path = model_dir / "model.safetensors"
    if single_path.is_file():
        return [single_path]
    shards = sorted(model_dir.glob("model-*.safetensors"))
    if shards:
        return shards
    raise FileNotFoundError(
        f"Text encoder checkpoint not found in {model_dir}: expected model.safetensors, "
        "model.safetensors.index.json, or model-*.safetensors"
    )


def _load_plain_weights(model: nn.Module, model_dir: Path) -> dict[str, list[str]]:
    state_dict: dict[str, torch.Tensor] = {}
    for shard_path in _shard_paths(model_dir):
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name.startswith("model.visual.") or name.startswith("lm_head."):
                    continue
                mapped = _strip_language_prefix(name)
                tensor = handle.get_tensor(name)
                if tensor.is_floating_point():
                    tensor = tensor.to(dtype=torch.float16)
                state_dict[mapped] = tensor

    incompatible = model.load_state_dict(state_dict, strict=False, assign=True)
    return {
        "unexpected_keys": list(incompatible.unexpected_keys),
        "missing_keys": list(incompatible.missing_keys),
    }


def _cast_float_tensors_except_quantized(module: torch.nn.Module, dtype: torch.dtype) -> None:
    if isinstance(module, (QuantizedLinear, QuantizedEmbedding, Qwen3RotaryEmbedding)):
        return

    for parameter_name, parameter in module.named_parameters(recurse=False):
        if parameter is not None and parameter.is_floating_point() and parameter.dtype != dtype:
            module._parameters[parameter_name] = torch.nn.Parameter(
                parameter.to(dtype=dtype),
                requires_grad=parameter.requires_grad,
            )

    for buffer_name, buffer in module.named_buffers(recurse=False):
        if buffer is not None and buffer.is_floating_point() and buffer.dtype != dtype:
            module._buffers[buffer_name] = buffer.to(dtype=dtype)

    for child in module.children():
        _cast_float_tensors_except_quantized(child, dtype)


def load_krea2_text_encoder(path: str | Path, quant_method: str | None = None) -> Qwen3Model:
    """Load the Krea 2 text encoder.

    The model is the Qwen3-VL-4B language model, 36 layers. The vision tower is
    not in the quantized file and is not built.

    Args:
        path: The `text_encoder` folder, or a file inside it.
        quant_method: A quantization method name, or None for plain weights.

    Returns:
        The text model in eval mode with float weights in fp16.

    Raises:
        ValueError: If `model_type` in `config.json` is not `qwen3_vl`.
        FileNotFoundError: If the checkpoint or config is missing.
        RuntimeError: If the config or the checkpoint keys do not match.
    """
    model_path = Path(path).expanduser()
    model_dir = model_path.parent if model_path.is_file() else model_path

    if quant_method is not None:
        quant_method_key = replace_hyphens_with_underscores(quant_method)
        rank_1_method, rank_2_method = convert_quant_ranks(quant_method, 2)
    else:
        quant_method_key = rank_1_method = rank_2_method = None

    target_tensors = _build_krea2_text_encoder_targets()
    rank_1_linear = target_tensors_to_linear_names(target_tensors["rank_1"], key_transform=_strip_language_prefix)
    rank_2_linear = target_tensors_to_linear_names(target_tensors["rank_2"], key_transform=_strip_language_prefix)

    config = _build_config(model_dir)
    default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float16)
        with torch.device("meta"):
            model = Qwen3Model(config)
    finally:
        torch.set_default_dtype(default_dtype)

    if quant_method is None:
        incompatible = _load_plain_weights(model, model_dir)
        _materialize_meta_tensors(model)
        _cast_float_tensors_except_quantized(model, torch.float16)
        model.eval()
        return model

    quantized_path = model_dir / f"{quant_method_key}_quant.safetensors"
    if not quantized_path.is_file():
        raise FileNotFoundError(
            f"Quantized checkpoint not found: {quantized_path}. Run the text encoder quantizer first."
        )
    print(f"    using saved quantized checkpoint: {quantized_path.name}")

    checkpoint_keys = _list_checkpoint_keys(quantized_path)
    if rank_1_method not in _PLAIN_METHODS:
        replace_targeted_linear_modules(
            model,
            method=rank_1_method,
            target_linear_names=rank_1_linear,
            quantize_weights=False,
            checkpoint_keys=checkpoint_keys,
        )
    if rank_2_method not in _PLAIN_METHODS:
        replace_targeted_linear_modules(
            model,
            method=rank_2_method,
            target_linear_names=rank_2_linear,
            quantize_weights=False,
            checkpoint_keys=checkpoint_keys,
        )

    embedding_names = {
        _strip_language_prefix(name).removesuffix(".weight")
        for name in target_tensors.get("sym-high", [])
    }
    if "embed_tokens" in embedding_names and "embed_tokens.sub_scales" in checkpoint_keys:
        model.embed_tokens = QuantizedEmbedding.from_prequantized(model.embed_tokens, method="sym-high")

    incompatible = load_local_single_checkpoint(model, quantized_path, key_transform=_strip_language_prefix)
    if incompatible.get("unexpected_keys") or incompatible.get("missing_keys"):
        raise RuntimeError(
            "Text encoder checkpoint does not match the model. "
            f"Unexpected keys: {', '.join(sorted(incompatible['unexpected_keys'])) or 'none'}. "
            f"Missing keys: {', '.join(sorted(incompatible['missing_keys'])) or 'none'}."
        )

    _materialize_meta_tensors(model)
    _cast_float_tensors_except_quantized(model, torch.float16)
    model.eval()
    return model
