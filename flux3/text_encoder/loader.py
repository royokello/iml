from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from flux2.models.text_encoder.model import Qwen3Model
from flux2.text_encoder.loader import _materialize_meta_tensors
from flux3.config import TEXT_ENCODER_LAYERS
from flux3.text_encoder.target import _build_flux3_text_encoder_target_tensors
from utils.loaders.single import load_local_single_checkpoint
from utils.quant.name import convert_quant_name
from utils.quant.replace import replace_targeted_linear_modules, target_tensors_to_linear_names
from utils.text import replace_hyphens_with_underscores

_LANGUAGE_PREFIX = "model.language_model."
_VISUAL_PREFIX = "model.visual."
_CHECKED_CONFIG_KEYS = (
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "rms_norm_eps",
)


def _strip_language_prefix(name: str) -> str:
    if name.startswith(_LANGUAGE_PREFIX):
        return name[len(_LANGUAGE_PREFIX):]
    if name.startswith("model."):
        return name[len("model."):]
    return name


def _map_checkpoint_key(name: str) -> str | None:
    """Map an original checkpoint key to a model key, or None to drop it."""
    if name.startswith(_VISUAL_PREFIX) or name.startswith("lm_head."):
        return None
    mapped = _strip_language_prefix(name)
    if mapped == "norm.weight":
        return None
    if mapped.startswith("layers."):
        layer_index = int(mapped.split(".")[1])
        if layer_index >= TEXT_ENCODER_LAYERS:
            return None
    return mapped


def _list_checkpoint_keys(checkpoint_path: Path) -> set[str]:
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        return {_strip_language_prefix(name) for name in handle.keys()}


def _build_config(model_dir: Path) -> Qwen3Config:
    with (model_dir / "config.json").open("r", encoding="utf-8") as handle:
        full_config = json.load(handle)

    if full_config.get("model_type") != "qwen3_vl":
        raise ValueError(
            f"Unsupported text encoder model_type {full_config.get('model_type')!r}. Expected 'qwen3_vl'."
        )

    text_config = dict(full_config["text_config"])
    rope_theta = text_config["rope_theta"]
    init_parameters = inspect.signature(Qwen3Config.__init__).parameters
    init_kwargs = {key: value for key, value in text_config.items() if key in init_parameters and key != "self"}
    # The MRoPE fields are dropped. A text-only input uses plain 1D RoPE.
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


def _load_fp16_weights(model: nn.Module, model_dir: Path) -> None:
    state_dict: dict[str, torch.Tensor] = {}
    for shard_path in _shard_paths(model_dir):
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                mapped = _map_checkpoint_key(name)
                if mapped is None:
                    continue
                tensor = handle.get_tensor(name)
                if tensor.is_floating_point():
                    tensor = tensor.to(dtype=torch.float16)
                state_dict[mapped] = tensor

    incompatible = model.load_state_dict(state_dict, strict=False, assign=True)
    if incompatible.unexpected_keys or incompatible.missing_keys:
        raise RuntimeError(
            "Text encoder checkpoint does not match the model. "
            f"Unexpected keys: {', '.join(sorted(incompatible.unexpected_keys)) or 'none'}. "
            f"Missing keys: {', '.join(sorted(incompatible.missing_keys)) or 'none'}."
        )


def load_flux3_text_encoder(path: str | Path, quant_method: str | None = None) -> Qwen3Model:
    """Load the Flux 3 text encoder.

    The model has 33 layers. The visual tower, layers 33 to 35, and the final
    norm are not loaded. The final norm is an identity layer, so
    `hidden_states[33]` is the raw output of layer 32.

    Args:
        path: The `text_encoder` folder.
        quant_method: A quantization method name, or None for fp16 weights.

    Returns:
        The text model with the Qwen3 text layers.

    Raises:
        ValueError: If `model_type` in `config.json` is not `qwen3_vl`.
        FileNotFoundError: If the checkpoint files are missing.
        RuntimeError: If the checkpoint keys do not match the model, or if a
            tensor stays on the meta device.
    """
    model_path = Path(path).expanduser()
    model_dir = model_path.parent if model_path.is_file() else model_path

    if quant_method is not None:
        quant_method_key = replace_hyphens_with_underscores(quant_method)
        high_method, low_method = convert_quant_name(quant_method)
    else:
        quant_method_key = high_method = low_method = None

    target_tensors = _build_flux3_text_encoder_target_tensors()
    high_linear = target_tensors_to_linear_names(target_tensors["high"], key_transform=_strip_language_prefix)
    low_linear = target_tensors_to_linear_names(target_tensors["low"], key_transform=_strip_language_prefix)

    config = _build_config(model_dir)
    default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float16)
        with torch.device("meta"):
            model = Qwen3Model(config)
    finally:
        torch.set_default_dtype(default_dtype)
    # The final norm has no effect on the used hidden states. Its weight is not loaded.
    model.norm = nn.Identity()

    if quant_method is None:
        _load_fp16_weights(model, model_dir)
    else:
        quantized_state_path = model_dir / f"{quant_method_key}_quant.safetensors"
        if not quantized_state_path.is_file():
            raise FileNotFoundError(
                f"Quantized checkpoint not found: {quantized_state_path}. Run the text encoder quantizer first."
            )
        print(f"    using saved quantized checkpoint: {quantized_state_path.name}")
        checkpoint_keys = _list_checkpoint_keys(quantized_state_path)
        replace_targeted_linear_modules(
            model, method=high_method, target_linear_names=high_linear,
            quantize_weights=False, checkpoint_keys=checkpoint_keys,
        )
        replace_targeted_linear_modules(
            model, method=low_method, target_linear_names=low_linear,
            quantize_weights=False, checkpoint_keys=checkpoint_keys,
        )
        load_local_single_checkpoint(model, quantized_state_path, key_transform=_strip_language_prefix)

    _materialize_meta_tensors(model)
    return model
