from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Dict

import torch
from safetensors import safe_open

from utils.loaders.sharded import (
    load_local_sharded_checkpoint,
    stream_local_sharded_checkpoint,
)
from utils.loaders.single import load_local_single_checkpoint, stream_local_single_checkpoint
from torch.nn import RMSNorm
from utils.quant.linear import QuantizedLinear
from utils.quant.name import convert_quant_ranks
from utils.quant.replace import replace_targeted_linear_modules, target_tensors_to_linear_names
from utils.quant.streaming import StreamedQuantizedLinear, pin_quantized_buffers

from flux2.models.denoiser.transformer import Flux2Transformer2DModel
from flux2.denoiser.targets import _build_flux2_denoiser_target_tensors


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
        if isinstance(child, (QuantizedLinear, RMSNorm)):
            continue
        _cast_float_tensors_except_quantized_linear(child, dtype)


def _build_model_from_config(model_dir: Path, *, device: str = "meta") -> Flux2Transformer2DModel:
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
        with torch.device(device):
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
    *,
    offloading: bool = False,
    stream_load: bool | None = None,
    stream_denoiser: bool = False,
) -> Flux2Transformer2DModel:
    variant = variant.strip().lower()
    version = version.strip().lower()

    if stream_denoiser and quant_method is None:
        raise ValueError(
            "stream_denoiser needs a quantized checkpoint. "
            "Pass a --denoiser-quant method, or drop --stream-denoiser."
        )
    linear_cls = StreamedQuantizedLinear if stream_denoiser else QuantizedLinear

    # ---- Parse mixed-precision quantisation ----
    if quant_method is not None:
        quant_method_key = quant_method.replace("-", "_")
        rank_1_method, rank_2_method = convert_quant_ranks(quant_method, 2)
    else:
        rank_1_method = rank_2_method = None

    # ---- Build target tensor groups ----
    target_tensors: Dict[str, list[str]] = _build_flux2_denoiser_target_tensors(version)
    rank_1_linear = target_tensors_to_linear_names(target_tensors["rank_1"])
    rank_2_linear = target_tensors_to_linear_names(target_tensors["rank_2"])

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

    build_device = "cpu" if offloading else "meta"
    model = _build_model_from_config(model_dir, device=build_device)
    if stream_load is None:
        stream_load = offloading

    # ---- Quantised checkpoint path ----
    quantized_checkpoint_path = None
    if quant_method is not None:
        quantized_filename = f"{quant_method_key}_quant.safetensors"  # e.g. "q4_max_quant.safetensors"
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
            if stream_load:
                incompatible = stream_local_single_checkpoint(model, checkpoint_path, offloading=offloading)
            else:
                incompatible = load_local_single_checkpoint(model, checkpoint_path)
            if incompatible.get("unexpected_keys"):
                raise RuntimeError(
                    "Unexpected keys in denoiser checkpoint: " + ", ".join(sorted(incompatible["unexpected_keys"]))
                )
        elif checkpoint_shards:
            print(f"    loading sharded checkpoint from {checkpoint_dir}")
            if stream_load:
                stream_local_sharded_checkpoint(
                    model,
                    checkpoint_dir,
                    index_filename=None,
                    shard_pattern="diffusion_pytorch_model-*.safetensors",
                    offloading=offloading,
                )
            else:
                load_local_sharded_checkpoint(
                    model,
                    checkpoint_dir,
                    index_filename=None,
                    shard_pattern="diffusion_pytorch_model-*.safetensors",
                )
        else:
            raise FileNotFoundError(f"Denoiser checkpoint not found: {checkpoint_path}")
        if not offloading:
            _materialize_meta_tensors(model)
    else:
        # Load a pre‑quantised checkpoint
        checkpoint_keys = _list_checkpoint_keys(quantized_checkpoint_path)

        # Replace high‑priority modules (they expect weights in rank_1_method quant format)
        replace_targeted_linear_modules(
            model,
            method=rank_1_method,
            target_linear_names=rank_1_linear,
            quantize_weights=False,
            linear_cls=linear_cls,
            checkpoint_keys=checkpoint_keys,
        )
        # Replace low‑priority modules (rank_2_method quant format)
        replace_targeted_linear_modules(
            model,
            method=rank_2_method,
            target_linear_names=rank_2_linear,
            quantize_weights=False,
            linear_cls=linear_cls,
            checkpoint_keys=checkpoint_keys,
        )

        if stream_load:
            incompatible = stream_local_single_checkpoint(model, quantized_checkpoint_path, offloading=offloading)
        else:
            incompatible = load_local_single_checkpoint(model, quantized_checkpoint_path)
        if incompatible.get("unexpected_keys"):
            raise RuntimeError(
                "Unexpected keys in denoiser checkpoint: " + ", ".join(sorted(incompatible["unexpected_keys"]))
            )
        if not offloading:
            _materialize_meta_tensors(model)

    # ---- Quantise on the fly if no pre‑quantised file was found ----
    if quant_method is not None and quantized_checkpoint_path is None:
        replace_targeted_linear_modules(
            model,
            method=rank_1_method,
            target_linear_names=rank_1_linear,
            linear_cls=linear_cls,
        )
        replace_targeted_linear_modules(
            model,
            method=rank_2_method,
            target_linear_names=rank_2_linear,
            linear_cls=linear_cls,
        )

    # ---- Finalise dtype ----
    if quant_method is None:
        if offloading:
            _cast_float_tensors_except_quantized_linear(model, torch.float16)
        else:
            model = model.to(dtype=torch.float16)
    else:
        _cast_float_tensors_except_quantized_linear(model, torch.float16)

    if stream_denoiser:
        pin_quantized_buffers(model)

    model.eval()
    return model
