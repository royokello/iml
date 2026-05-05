from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Dict

import torch
from safetensors import safe_open

from utils.loaders.sharded import load_local_sharded_checkpoint
from utils.loaders.single import load_local_single_checkpoint
from utils.quant.linear import QuantizedLinear
from utils.quant.name import convert_quant_name
from utils.quant.replace import replace_targeted_linear_modules, target_tensors_to_linear_names
from utils.quant.validators import normalize_quant_method

from flux2.models.denoiser.transformer import Flux2Transformer2DModel
from flux2.denoiser.quant import _build_target_tensors


def _materialize_meta_tensors(model: torch.nn.Module) -> None:
    unresolved_parameters = [name for name, parameter in model.named_parameters()
                             if getattr(parameter, "is_meta", False)]
    if unresolved_parameters:
        raise RuntimeError("Checkpoint load left parameters on meta: " + ", ".join(unresolved_parameters))

    unresolved_buffers = [name for name, buffer in model.named_buffers()
                          if getattr(buffer, "is_meta", False)]
    if unresolved_buffers:
        raise RuntimeError("Checkpoint load left buffers on meta: " + ", ".join(unresolved_buffers))


def _cast_float_tensors_except_quantized_linear(module: torch.nn.Module, dtype: torch.dtype) -> None:
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
        if isinstance(child, QuantizedLinear):
            continue
        _cast_float_tensors_except_quantized_linear(child, dtype)


def _build_model_from_config(model_dir: Path) -> Flux2Transformer2DModel:
    config_path = model_dir / "config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    init_parameters = inspect.signature(Flux2Transformer2DModel.__init__).parameters
    init_kwargs = {
        key: value
        for key, value in config.items()
        if key in init_parameters and key != "self"
    }

    default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float16)
        with torch.device("meta"):
            model = Flux2Transformer2DModel(**init_kwargs)
    finally:
        torch.set_default_dtype(default_dtype)
    return model


def _list_checkpoint_keys(checkpoint_path: Path) -> set[str]:
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        return set(handle.keys())


def _load_flux2_denoiser(
    path: str | Path,
    quant_method: str | None = None,
    variant: str = "distill",
    version: str = "4b",
) -> Flux2Transformer2DModel:
    variant = variant.strip().lower()
    version = version.strip().lower()

    # ---- Parse mixed-precision quantisation ----
    if quant_method is not None:
        # Normalise the raw name (e.g. "aff-med-max" → "aff_med_max")
        quant_method_key = quant_method.replace("-", "_")
        # Split into high / low methods (e.g. "aff_med_max" → ("sym_med", "aff_med"))
        high_raw, low_raw = convert_quant_name(quant_method)
        high_method = normalize_quant_method(high_raw)
        low_method = normalize_quant_method(low_raw)
    else:
        high_method = low_method = None

    # ---- Build target tensor groups ----
    target_tensors: Dict[str, list[str]] = _build_target_tensors(version)
    high_linear = target_tensors_to_linear_names(target_tensors["high"])
    low_linear = target_tensors_to_linear_names(target_tensors["low"])

    model_dir = Path(path).expanduser().resolve()
    if model_dir.is_file():
        model_dir = model_dir.parent

    if model_dir.name in {"base", "distill"} and model_dir.parent.name == "transformer":
        if model_dir.name != variant:
            raise ValueError(
                f"Transformer variant mismatch: path points to {model_dir.name!r}, variant={variant!r}."
            )
        checkpoint_dir = model_dir
        model_dir = model_dir.parent
    else:
        checkpoint_dir = model_dir / variant

    if not model_dir.is_dir():
        raise FileNotFoundError(f"Denoiser directory not found: {model_dir}")
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Denoiser checkpoint directory not found: {checkpoint_dir}")

    model = _build_model_from_config(model_dir)

    # ---- Quantised checkpoint path ----
    quantized_checkpoint_path = None
    if quant_method is not None:
        quantized_filename = f"{quant_method_key}_quant.safetensors"  # e.g. "aff_med_max_quant.safetensors"
        quantized_checkpoint_path = checkpoint_dir / quantized_filename
        if not quantized_checkpoint_path.is_file():
            print(f"    saved quantized checkpoint not found in {checkpoint_dir}; quantizing on the fly")
            quantized_checkpoint_path = None
        else:
            print(f"    using saved quantized checkpoint: {quantized_checkpoint_path.name}")

    # ---- Load weights ----
    if quantized_checkpoint_path is None:
        # Load original FP16 checkpoint
        checkpoint_path = checkpoint_dir / "diffusion_pytorch_model.safetensors"
        checkpoint_shards = sorted(
            path.name for path in checkpoint_dir.glob("diffusion_pytorch_model-*.safetensors")
        )
        if checkpoint_path.is_file():
            incompatible = load_local_single_checkpoint(model, checkpoint_path)
            if incompatible.unexpected_keys:
                raise RuntimeError(
                    "Unexpected keys in denoiser checkpoint: " + ", ".join(sorted(incompatible.unexpected_keys))
                )
        elif checkpoint_shards:
            print(f"    loading sharded checkpoint from {checkpoint_dir}")
            load_local_sharded_checkpoint(
                model,
                checkpoint_dir,
                index_filename=None,
                shard_pattern="diffusion_pytorch_model-*.safetensors",
            )
        else:
            raise FileNotFoundError(f"Denoiser checkpoint not found: {checkpoint_path}")
        _materialize_meta_tensors(model)
    else:
        # Load a pre‑quantised checkpoint
        checkpoint_keys = _list_checkpoint_keys(quantized_checkpoint_path)

        # Replace high‑priority modules (they expect weights in high_method quant format)
        replace_targeted_linear_modules(
            model,
            method=high_method,
            target_linear_names=high_linear,
            quantize_weights=False,
            checkpoint_keys=checkpoint_keys,
        )
        # Replace low‑priority modules (low_method quant format)
        replace_targeted_linear_modules(
            model,
            method=low_method,
            target_linear_names=low_linear,
            quantize_weights=False,
            checkpoint_keys=checkpoint_keys,
        )

        incompatible = load_local_single_checkpoint(model, quantized_checkpoint_path)
        if incompatible.unexpected_keys:
            raise RuntimeError(
                "Unexpected keys in denoiser checkpoint: " + ", ".join(sorted(incompatible.unexpected_keys))
            )
        _materialize_meta_tensors(model)

    # ---- Quantise on the fly if no pre‑quantised file was found ----
    if quant_method is not None and quantized_checkpoint_path is None:
        replace_targeted_linear_modules(
            model,
            method=high_method,
            target_linear_names=high_linear,
        )
        replace_targeted_linear_modules(
            model,
            method=low_method,
            target_linear_names=low_linear,
        )

    # ---- Finalise dtype ----
    if quant_method is None:
        model = model.to(dtype=torch.float16)
    else:
        _cast_float_tensors_except_quantized_linear(model, torch.float16)

    model.eval()
    return model