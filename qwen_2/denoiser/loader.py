from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch
from diffusers import QwenImage21Transformer2DModel
from diffusers.models.transformers.transformer_qwenimage21 import (
    QwenImage21Rope,
    QwenImage21TemporalTimesteps,
    QwenImage21TimestepProjEmbeddings,
)
from safetensors import safe_open

from qwen_2.config import VARIANTS
from qwen_2.denoiser.targets import _build_qwen2_denoiser_target_tensors
from utils.loaders.sharded import load_local_sharded_checkpoint, stream_local_sharded_checkpoint
from utils.loaders.single import load_local_single_checkpoint, stream_local_single_checkpoint
from utils.quant.linear import QuantizedLinear
from utils.quant.replace import replace_targeted_linear_modules, target_tensors_to_linear_names
from utils.quant.targets import build_mixed_target_config
from utils.quant.streaming import StreamedQuantizedLinear, pin_quantized_buffers
from utils.text import replace_hyphens_with_underscores

__all__ = ["load_qwen2_denoiser"]

_SHARD_PATTERN = "diffusion_pytorch_model-*.safetensors"
_PLAIN_METHODS = ("fp16", "fp32", "bf16")


def _build_model_from_config(model_dir: Path, *, device: str = "meta") -> QwenImage21Transformer2DModel:
    with (model_dir / "config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    init_parameters = inspect.signature(QwenImage21Transformer2DModel.__init__).parameters
    init_kwargs = {key: value for key, value in config.items() if key in init_parameters and key != "self"}

    default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float16)
        with torch.device(device):
            model = QwenImage21Transformer2DModel(**init_kwargs)
    finally:
        torch.set_default_dtype(default_dtype)

    # The RoPE table and the sinusoidal timestep table are non-persistent
    # buffers built in `__init__`, so a meta build leaves them on meta and
    # `.to(device)` fails. Rebuild both on CPU; the persistent weights arrive
    # from the checkpoint.
    axes_dims_rope = tuple(config.get("axes_dims_rope", (16, 56, 56)))
    model.pos_embed = QwenImage21Rope(theta=10000, axes_dim=list(axes_dims_rope))
    model.time_text_embed = QwenImage21TimestepProjEmbeddings(embedding_dim=model.inner_dim)
    return model


def _cast_float_tensors_except_quantized(module: torch.nn.Module, dtype: torch.dtype) -> None:
    # The derived tables stay float32: their values lose precision in float16.
    if isinstance(
        module,
        (QuantizedLinear, QwenImage21Rope, QwenImage21TemporalTimesteps),
    ):
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


def _materialize_meta_parameters(model: torch.nn.Module) -> None:
    for module in model.modules():
        for name, parameter in list(module._parameters.items()):
            if parameter is not None and parameter.is_meta:
                module._parameters[name] = torch.nn.Parameter(
                    torch.empty(parameter.shape, dtype=parameter.dtype),
                    requires_grad=parameter.requires_grad,
                )


def _check_meta_tensors(model: torch.nn.Module) -> None:
    unresolved = [name for name, parameter in model.named_parameters() if parameter.is_meta]
    unresolved += [name for name, buffer in model.named_buffers() if buffer.is_meta]
    if unresolved:
        raise RuntimeError("Denoiser load left tensors on meta: " + ", ".join(unresolved))


def _checkpoint_keys(path: Path) -> set[str]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return set(handle.keys())


def _shard_paths(checkpoint_dir: Path) -> list[Path]:
    index_path = checkpoint_dir / "diffusion_pytorch_model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open("r", encoding="utf-8") as handle:
            weight_map = json.load(handle)["weight_map"]
        return [checkpoint_dir / name for name in sorted(set(weight_map.values()))]
    return sorted(checkpoint_dir.glob(_SHARD_PATTERN))


def load_qwen2_denoiser(
    path: str | Path,
    quant_method: str | None = None,
    *,
    variant: str = "turbo",
    offloading: bool = False,
    stream_load: bool | None = None,
    stream_denoiser: bool = False,
) -> QwenImage21Transformer2DModel:
    """Load the Qwen Image 2.1 denoiser.

    Args:
        path: The `denoiser` folder, or a `denoiser/<variant>` folder.
        quant_method: A quantization method name, or None for plain weights.
        variant: `turbo` or `base`.
        offloading: Build and keep the weights in host memory.
        stream_load: Copy tensors into the model one at a time. Defaults to
            `offloading`.
        stream_denoiser: Keep the quantized weights in pinned host memory and
            copy each layer onto the GPU for its own multiply.

    Returns:
        The denoiser in eval mode with float weights in fp16.

    Raises:
        ValueError: If the variant is unknown, if `path` points to a different
            variant, or if `stream_denoiser` has no quantized checkpoint.
        FileNotFoundError: If a folder or checkpoint is missing. The loader
            never quantizes on the fly.
        RuntimeError: If the checkpoint keys do not match the model, or if a
            tensor stays on the meta device.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown denoiser variant {variant!r}. Expected one of {VARIANTS}.")
    if stream_denoiser and quant_method is None:
        raise ValueError(
            "stream_denoiser needs a quantized checkpoint. "
            "Pass a quant method, or drop --stream-denoiser."
        )
    linear_cls = StreamedQuantizedLinear if stream_denoiser else QuantizedLinear

    if quant_method is not None:
        quant_method_key = replace_hyphens_with_underscores(quant_method)
    else:
        quant_method_key = None

    target_tensors = _build_qwen2_denoiser_target_tensors()
    targets = (
        build_mixed_target_config(quant_method, target_tensors)
        if quant_method is not None
        else {}
    )

    model_dir = Path(path).expanduser().resolve()
    if model_dir.is_file():
        model_dir = model_dir.parent

    if model_dir.name in VARIANTS and model_dir.parent.name == "denoiser":
        if model_dir.name != variant:
            raise ValueError(
                f"Denoiser variant mismatch: path points to {model_dir.name!r}, variant={variant!r}."
            )
        checkpoint_dir = model_dir
        model_dir = model_dir.parent
    else:
        checkpoint_dir = model_dir / variant

    if not model_dir.is_dir():
        raise FileNotFoundError(f"Denoiser directory not found: {model_dir}")
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Denoiser checkpoint directory not found: {checkpoint_dir}")

    build_device = "cpu" if (offloading and not stream_denoiser) else "meta"
    model = _build_model_from_config(model_dir, device=build_device)
    if stream_load is None:
        stream_load = offloading

    quantized_checkpoint_path = None
    if quant_method is not None:
        quantized_checkpoint_path = checkpoint_dir / f"{quant_method_key}_quant.safetensors"
        if not quantized_checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Quantized checkpoint not found: {quantized_checkpoint_path}. Run the denoiser quantizer first."
            )
        print(f"    using saved quantized checkpoint: {quantized_checkpoint_path.name}")

    if quantized_checkpoint_path is None:
        shard_paths = _shard_paths(checkpoint_dir)
        if not shard_paths:
            raise FileNotFoundError(f"Denoiser checkpoint not found in {checkpoint_dir}")
        if stream_load and len(shard_paths) == 1:
            incompatible = stream_local_single_checkpoint(
                model, shard_paths[0], offloading=offloading
            )
        elif stream_load:
            stream_local_sharded_checkpoint(
                model,
                checkpoint_dir,
                index_filename=None,
                shard_pattern=_SHARD_PATTERN,
                offloading=offloading,
            )
            incompatible = {"unexpected_keys": [], "missing_keys": []}
        elif len(shard_paths) == 1:
            incompatible = load_local_single_checkpoint(model, shard_paths[0])
        else:
            load_local_sharded_checkpoint(
                model,
                checkpoint_dir,
                index_filename=None,
                shard_pattern=_SHARD_PATTERN,
            )
            incompatible = {"unexpected_keys": [], "missing_keys": []}
    else:
        checkpoint_keys = _checkpoint_keys(quantized_checkpoint_path)
        for target_method, target_names in targets.items():
            if target_method in _PLAIN_METHODS:
                continue
            replace_targeted_linear_modules(
                model,
                method=target_method,
                target_linear_names=target_tensors_to_linear_names(target_names),
                quantize_weights=False,
                linear_cls=linear_cls,
                checkpoint_keys=checkpoint_keys,
            )

        if stream_denoiser:
            _materialize_meta_parameters(model)

        if stream_load:
            incompatible = stream_local_single_checkpoint(
                model, quantized_checkpoint_path, offloading=offloading
            )
        else:
            incompatible = load_local_single_checkpoint(model, quantized_checkpoint_path)

    if not offloading:
        _check_meta_tensors(model)

    unexpected = incompatible.get("unexpected_keys") or []
    missing = incompatible.get("missing_keys") or []
    if unexpected or missing:
        raise RuntimeError(
            "Denoiser checkpoint does not match the model. "
            f"Unexpected keys: {', '.join(sorted(unexpected)) or 'none'}. "
            f"Missing keys: {', '.join(sorted(missing)) or 'none'}."
        )

    _cast_float_tensors_except_quantized(model, torch.float16)

    if stream_denoiser:
        pin_quantized_buffers(model)

    model.eval()
    return model
