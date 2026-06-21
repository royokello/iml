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
from utils.quant.name import convert_quant_name
from utils.quant.replace import replace_targeted_linear_modules, target_tensors_to_linear_names

from flux2.models.denoiser.transformer import Flux2Transformer2DModel
from flux2.denoiser.targets import _build_flux2_denoiser_target_tensors


_RESIDENT_MODULE_NAMES = (
    "pos_embed",
    "time_guidance_embed",
    "double_stream_modulation_img",
    "double_stream_modulation_txt",
    "single_stream_modulation",
    "x_embedder",
    "context_embedder",
    "norm_out",
    "proj_out",
)


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


def move_resident_layers_to(model: Flux2Transformer2DModel, device: torch.device) -> None:
    """Move always-resident layers to `device`.

    The two block lists (`transformer_blocks`, `single_transformer_blocks`)
    are intentionally left in place; they are streamed to `device` per-block
    during `forward` when `enable_block_offload` is set.
    """
    for name in _RESIDENT_MODULE_NAMES:
        if not hasattr(model, name):
            continue
        module = getattr(model, name)
        if isinstance(module, torch.nn.Module):
            module.to(device)


def pin_module_parameters(model: torch.nn.Module) -> int:
    """Pin all float parameters/buffers of `model` to host-pinned memory.

    Returns the number of tensors pinned. Each pinned tensor can be transferred
    to CUDA asynchronously with `non_blocking=True`. Use this for the CPU
    master copy under `--pin-memory`.
    """
    pinned = 0
    for parameter_name, parameter in list(model.named_parameters(recurse=True)):
        if parameter is None or not parameter.is_floating_point():
            continue
        if parameter.device.type != "cpu":
            continue
        if not parameter.is_contiguous():
            parameter.data = parameter.data.contiguous()
        pinned_tensor = torch.empty_like(parameter.data, device="cpu", pin_memory=True)
        pinned_tensor.copy_(parameter.data)
        module_path, _, param_name = parameter_name.rpartition(".")
        parent = model.get_submodule(module_path) if module_path else model
        parent._parameters[param_name] = torch.nn.Parameter(pinned_tensor, requires_grad=parameter.requires_grad)
        pinned += 1
    for buffer_name, buffer in list(model.named_buffers(recurse=True)):
        if buffer is None or not buffer.is_floating_point():
            continue
        if buffer.device.type != "cpu":
            continue
        if not buffer.is_contiguous():
            buffer = buffer.contiguous()
        pinned_buffer = torch.empty_like(buffer, device="cpu", pin_memory=True)
        pinned_buffer.copy_(buffer)
        module_path, _, buf_name = buffer_name.rpartition(".")
        parent = model.get_submodule(module_path) if module_path else model
        parent._buffers[buf_name] = pinned_buffer
        pinned += 1
    return pinned


def _load_flux2_denoiser(
    path: str | Path,
    quant_method: str | None = None,
    variant: str = "distill",
    version: str = "4b",
    *,
    cpu_residency: bool = False,
    pin_memory: bool = False,
    stream_load: bool | None = None,
) -> Flux2Transformer2DModel:
    variant = variant.strip().lower()
    version = version.strip().lower()

    # ---- Parse mixed-precision quantisation ----
    if quant_method is not None:
        quant_method_key = quant_method.replace("-", "_")
        high_method, low_method = convert_quant_name(quant_method)
    else:
        high_method = low_method = None

    # ---- Build target tensor groups ----
    target_tensors: Dict[str, list[str]] = _build_flux2_denoiser_target_tensors(version)
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

    build_device = "cpu" if cpu_residency else "meta"
    model = _build_model_from_config(model_dir, device=build_device)
    if stream_load is None:
        stream_load = cpu_residency

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
            if stream_load:
                incompatible = stream_local_single_checkpoint(model, checkpoint_path, pin_memory=pin_memory)
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
                    pin_memory=pin_memory,
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
        if not cpu_residency:
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

        if stream_load:
            incompatible = stream_local_single_checkpoint(model, quantized_checkpoint_path, pin_memory=pin_memory)
        else:
            incompatible = load_local_single_checkpoint(model, quantized_checkpoint_path)
        if incompatible.get("unexpected_keys"):
            raise RuntimeError(
                "Unexpected keys in denoiser checkpoint: " + ", ".join(sorted(incompatible["unexpected_keys"]))
            )
        if not cpu_residency:
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
        if cpu_residency:
            _cast_float_tensors_except_quantized_linear(model, torch.float16)
        else:
            model = model.to(dtype=torch.float16)
    else:
        _cast_float_tensors_except_quantized_linear(model, torch.float16)

    model.eval()
    return model
