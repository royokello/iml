from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from flux2.text_encoder.target import _build_target_tensors
from utils.loaders.sharded import load_local_sharded_checkpoint
from utils.loaders.single import load_local_single_checkpoint
from utils.quant.name import convert_quant_name
from utils.quant.replace import replace_targeted_linear_modules, target_tensors_to_linear_names

from flux2.models.text_encoder.embedding import Qwen3RotaryEmbedding
from flux2.models.text_encoder.model import Qwen3Model
from utils.text import replace_hyphens_with_underscores


def _strip_model_prefix(name: str) -> str:
    if name.startswith("model."):
        name = name[len("model."):]
    return name


def _list_checkpoint_keys(checkpoint_path: Path) -> set[str]:
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        return {_strip_model_prefix(name) for name in handle.keys()}


def _materialize_meta_tensors(model: nn.Module, *, validate: bool = True) -> None:
    for module in model.modules():
        if isinstance(module, Qwen3RotaryEmbedding) and getattr(module.inv_freq, "is_meta", False):
            inv_freq, attention_scaling = module.rope_init_fn(module.config, device="cpu")
            module.register_buffer("inv_freq", inv_freq, persistent=False)
            module.original_inv_freq = module.inv_freq
            module.attention_scaling = attention_scaling

    if not validate:
        return

    unresolved_parameters = [
        name for name, parameter in model.named_parameters()
        if getattr(parameter, "is_meta", False)
    ]
    if unresolved_parameters:
        raise RuntimeError(
            "Checkpoint load left parameters on meta: " + ", ".join(unresolved_parameters)
        )

    unresolved_buffers = [
        name for name, buffer in model.named_buffers()
        if getattr(buffer, "is_meta", False)
    ]
    if unresolved_buffers:
        raise RuntimeError(
            "Checkpoint load left buffers on meta: " + ", ".join(unresolved_buffers)
        )


def _load_flux2_text_encoder(
    path: str | Path,
    quant_method: str | None = None,
) -> Qwen3Model:
    # 1. ---- Resolve directory ----
    model_path = Path(path)
    model_dir = model_path.parent if model_path.is_file() else model_path

    # 2. ---- Parse mixed‑precision quantisation ----
    if quant_method is not None:
        quant_method_key = replace_hyphens_with_underscores(quant_method)   # e.g. "aff_med_max"
        high_method, low_method = convert_quant_name(quant_method)
    else:
        quant_method_key = None
        high_method = low_method = None

    # 3. ---- Build high/low target linear names ----
    target_tensors = _build_target_tensors()  # dict with "high" and "low"
    high_linear = target_tensors_to_linear_names(
        target_tensors["high"], key_transform=_strip_model_prefix
    )
    low_linear = target_tensors_to_linear_names(
        target_tensors["low"], key_transform=_strip_model_prefix
    )

    # 4. ---- Build model on meta device ----
    config = Qwen3Config.from_pretrained(str(model_dir), local_files_only=True)
    default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float16)
        config.dtype = torch.float16
        with torch.device("meta"):
            model = Qwen3Model(config)
    finally:
        torch.set_default_dtype(default_dtype)

    # 5. ---- Determine quantized checkpoint path ----
    quantized_state_path = None
    if quant_method is not None:
        quantized_state_path = model_dir / f"{quant_method_key}_quant.safetensors"
        if not quantized_state_path.is_file():
            print(f"    saved quantized checkpoint not found in {model_dir}; quantizing on the fly")
            quantized_state_path = None
        else:
            print(f"    using saved quantized checkpoint: {quantized_state_path.name}")

    # 6. ---- Load weights ----
    if quantized_state_path is None:
        # Load original FP16 checkpoint
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

        # Quantise on the fly if requested
        if quant_method is not None:
            replace_targeted_linear_modules(model, method=high_method, target_linear_names=high_linear)
            replace_targeted_linear_modules(model, method=low_method, target_linear_names=low_linear)
    else:
        # Load pre‑quantised checkpoint
        checkpoint_keys = _list_checkpoint_keys(quantized_state_path)
        replace_targeted_linear_modules(
            model, method=high_method, target_linear_names=high_linear,
            quantize_weights=False, checkpoint_keys=checkpoint_keys,
        )
        replace_targeted_linear_modules(
            model, method=low_method, target_linear_names=low_linear,
            quantize_weights=False, checkpoint_keys=checkpoint_keys,
        )
        load_local_single_checkpoint(model, quantized_state_path, key_transform=_strip_model_prefix)
        _materialize_meta_tensors(model)

    return model
