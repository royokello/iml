from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from utils.loaders.sharded import load_local_sharded_checkpoint
from utils.loaders.single import load_local_single_checkpoint
from utils.quant.linear import QuantizedLinear

from flux2.models.text_encoder.embedding import Qwen3RotaryEmbedding
from flux2.models.text_encoder.model import Qwen3Model


def _resolve_model_dir(path: str | Path) -> Path:
    model_path = Path(path)
    if model_path.is_file():
        return model_path.parent
    return model_path


def _strip_model_prefix(name: str) -> str:
    if name.startswith("model."):
        return name[len("model.") :]
    return name


def _replace_linear_modules(
    module: nn.Module,
    *,
    method: str,
    quantize_weights: bool = True,
) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            quantized_linear = (
                QuantizedLinear(child, method=method)
                if quantize_weights
                else QuantizedLinear.from_prequantized(child, method=method)
            )
            setattr(module, name, quantized_linear)
            continue
        _replace_linear_modules(
            child,
            method=method,
            quantize_weights=quantize_weights,
        )


def _materialize_meta_tensors(model: nn.Module, *, validate: bool = True) -> None:
    for module in model.modules():
        if isinstance(module, Qwen3RotaryEmbedding) and getattr(module.inv_freq, "is_meta", False):
            inv_freq, attention_scaling = module.rope_init_fn(module.config, device="cpu")
            module.register_buffer("inv_freq", inv_freq, persistent=False)
            module.original_inv_freq = module.inv_freq
            module.attention_scaling = attention_scaling

    if not validate:
        return

    unresolved_parameters = [name for name, parameter in model.named_parameters() if getattr(parameter, "is_meta", False)]
    if unresolved_parameters:
        raise RuntimeError(
            "Checkpoint load left parameters on meta: " + ", ".join(unresolved_parameters)
        )

    unresolved_buffers = [name for name, buffer in model.named_buffers() if getattr(buffer, "is_meta", False)]
    if unresolved_buffers:
        raise RuntimeError(
            "Checkpoint load left buffers on meta: " + ", ".join(unresolved_buffers)
        )


def _load_flux2_text_encoder(
    path: str | Path,
    quant_method: str | None = None,
) -> Qwen3Model:
    model_dir = _resolve_model_dir(path)
    if quant_method is not None:
        quant_method = quant_method.lower()
        if quant_method not in {"single", "double"}:
            raise ValueError('quant_method must be "single", "double", or None.')

    config = Qwen3Config.from_pretrained(str(model_dir), local_files_only=True)
    default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float16)
        config.dtype = torch.float16
        with torch.device("meta"):
            model = Qwen3Model(config)
    finally:
        torch.set_default_dtype(default_dtype)

    quantized_state_path = None if quant_method is None else model_dir / f"{quant_method}_quant.safetensors"

    if quantized_state_path is None or not quantized_state_path.is_file():
        full_checkpoint_path = model_dir / "model.safetensors"
        shard_index_path = model_dir / "model.safetensors.index.json"
        shard_paths = sorted(model_dir.glob("model-*.safetensors"))
        if full_checkpoint_path.is_file():
            load_local_single_checkpoint(model, full_checkpoint_path, key_transform=_strip_model_prefix)
        elif shard_index_path.is_file():
            load_local_sharded_checkpoint(model, model_dir, key_transform=_strip_model_prefix)
        elif shard_paths:
            load_local_sharded_checkpoint(
                model,
                model_dir,
                index_filename=None,
                shard_pattern="model-*.safetensors",
                key_transform=_strip_model_prefix,
            )
        else:
            raise FileNotFoundError(
                f"Text encoder checkpoint not found in {model_dir}: expected model.safetensors, "
                "model.safetensors.index.json, or model-*.safetensors"
            )
        _materialize_meta_tensors(model)
        if quant_method is not None:
            _replace_linear_modules(model, method=quant_method)
    else:
        _replace_linear_modules(
            model,
            method=quant_method,
            quantize_weights=False,
        )
        load_local_single_checkpoint(model, quantized_state_path, key_transform=_strip_model_prefix)
        _materialize_meta_tensors(model)

    model.eval()
    return model
