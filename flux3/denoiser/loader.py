from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch
from safetensors import safe_open
from torch.nn import RMSNorm

from flux2.denoiser.loader import (
    _cast_float_tensors_except_quantized_linear,
    _list_checkpoint_keys,
    _materialize_meta_tensors,
    pin_module_parameters,
)
from flux3.denoiser.targets import _build_flux3_denoiser_target_tensors
from flux3.models.denoiser.transformer import Flux3Transformer2DModel
from utils.loaders.sharded import load_local_sharded_checkpoint, stream_local_sharded_checkpoint
from utils.loaders.single import load_local_single_checkpoint, stream_local_single_checkpoint
from utils.quant.name import convert_quant_name
from utils.quant.replace import replace_targeted_linear_modules, target_tensors_to_linear_names

__all__ = ["load_flux3_denoiser", "move_resident_layers_to", "pin_module_parameters"]

_VARIANTS = ("base", "distill")
_SHARD_PATTERN = "diffusion_pytorch_model-*.safetensors"

_RESIDENT_MODULE_NAMES = (
    "pos_embed",
    "time_proj",
    "timestep_embedder",
    "target_timestep_embedder",
    "early_stream_modulations",
    "single_stream_modulations",
    "x_embedders",
    "context_embedder",
    "norm_out",
    "proj_out",
)


def move_resident_layers_to(model: Flux3Transformer2DModel, device: torch.device) -> None:
    """Move the always-resident layers to `device`.

    The block lists (`mode_blocks`, `txt_mode_blocks`, `single_transformer_blocks`)
    stay where they are. `forward` streams them when `enable_block_offload` is set.

    Args:
        model: The denoiser.
        device: The target device.
    """
    for name in _RESIDENT_MODULE_NAMES:
        module = getattr(model, name, None)
        if isinstance(module, torch.nn.Module):
            module.to(device)


def _build_model_from_config(model_dir: Path, variant: str, *, device: str = "meta") -> Flux3Transformer2DModel:
    with (model_dir / "config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    init_parameters = inspect.signature(Flux3Transformer2DModel.__init__).parameters
    init_kwargs = {key: value for key, value in config.items() if key in init_parameters and key != "self"}
    # config.json has no `flow_map` key. The variant sets it.
    init_kwargs["flow_map"] = variant == "distill"

    default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float16)
        with torch.device(device):
            model = Flux3Transformer2DModel(**init_kwargs)
    finally:
        torch.set_default_dtype(default_dtype)
    return model


def _check_shard_keys(model: torch.nn.Module, checkpoint_dir: Path) -> None:
    shard_paths = sorted(checkpoint_dir.glob(_SHARD_PATTERN))
    checkpoint_keys: set[str] = set()
    for shard_path in shard_paths:
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            checkpoint_keys.update(handle.keys())

    model_keys = set(model.state_dict().keys())
    unexpected = sorted(checkpoint_keys - model_keys)
    missing = sorted(model_keys - checkpoint_keys)
    if unexpected or missing:
        raise RuntimeError(
            f"Denoiser checkpoint in {checkpoint_dir} does not match the model. "
            f"Unexpected keys: {', '.join(unexpected) or 'none'}. "
            f"Missing keys: {', '.join(missing) or 'none'}."
        )


def _cast_qk_norms_to_fp32(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, RMSNorm):
            module.to(dtype=torch.float32)


def load_flux3_denoiser(
    path: str | Path,
    quant_method: str | None = None,
    variant: str = "distill",
    *,
    offloading: bool = False,
    stream_load: bool | None = None,
) -> Flux3Transformer2DModel:
    """Load the Flux 3 denoiser.

    Args:
        path: The `transformer` folder, or the `transformer/<variant>` folder.
        quant_method: A quantization method name, or None for fp16 weights.
        variant: `base` or `distill`. `distill` builds the flow-map model.
        offloading: If true, build the model on the CPU and keep the weights there.
        stream_load: If true, copy tensors into the model one at a time. Defaults
            to the value of `offloading`.

    Returns:
        The denoiser in eval mode. Float weights are fp16. The QK norm weights
        are fp32. `QuantizedLinear` layers keep their own dtypes.

    Raises:
        ValueError: If `variant` is not `base` or `distill`, or if `path` points
            to the other variant.
        FileNotFoundError: If the folders, the checkpoint files, or the quantized
            checkpoint for `quant_method` are missing. The loader never quantizes
            on the fly.
        RuntimeError: If the checkpoint keys do not match the model, or if a
            tensor stays on the meta device.
    """
    variant = variant.strip().lower()
    if variant not in _VARIANTS:
        raise ValueError(f"Unsupported variant {variant!r}. Expected one of: {', '.join(_VARIANTS)}.")

    # ---- Quantization setup ----
    if quant_method is not None:
        quant_method_key = quant_method.replace("-", "_")
        high_method, low_method = convert_quant_name(quant_method)
    else:
        quant_method_key = high_method = low_method = None

    target_tensors = _build_flux3_denoiser_target_tensors()
    high_linear = target_tensors_to_linear_names(target_tensors["high"])
    low_linear = target_tensors_to_linear_names(target_tensors["low"])

    # ---- Resolve folders ----
    model_dir = Path(path).expanduser().resolve()
    if model_dir.is_file():
        model_dir = model_dir.parent

    if model_dir.name in _VARIANTS and model_dir.parent.name == "transformer":
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
    model = _build_model_from_config(model_dir, variant, device=build_device)
    if stream_load is None:
        stream_load = offloading

    # ---- Quantized checkpoint path ----
    quantized_checkpoint_path = None
    if quant_method is not None:
        quantized_checkpoint_path = checkpoint_dir / f"{quant_method_key}_quant.safetensors"
        if not quantized_checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Quantized checkpoint not found: {quantized_checkpoint_path}. Run the denoiser quantizer first."
            )
        print(f"    using saved quantized checkpoint: {quantized_checkpoint_path.name}")

    # ---- Load weights ----
    if quantized_checkpoint_path is None:
        single_path = checkpoint_dir / "diffusion_pytorch_model.safetensors"
        if single_path.is_file():
            if stream_load:
                incompatible = stream_local_single_checkpoint(model, single_path, offloading=offloading)
            else:
                incompatible = load_local_single_checkpoint(model, single_path)
            problems = incompatible.get("unexpected_keys") or incompatible.get("missing_keys")
            if problems:
                raise RuntimeError(
                    "Denoiser checkpoint does not match the model. "
                    f"Unexpected keys: {', '.join(sorted(incompatible['unexpected_keys'])) or 'none'}. "
                    f"Missing keys: {', '.join(sorted(incompatible['missing_keys'])) or 'none'}."
                )
        elif any(checkpoint_dir.glob(_SHARD_PATTERN)):
            _check_shard_keys(model, checkpoint_dir)
            print(f"    loading sharded checkpoint from {checkpoint_dir}")
            if stream_load:
                stream_local_sharded_checkpoint(
                    model,
                    checkpoint_dir,
                    index_filename=None,
                    shard_pattern=_SHARD_PATTERN,
                    offloading=offloading,
                )
            else:
                load_local_sharded_checkpoint(
                    model,
                    checkpoint_dir,
                    index_filename=None,
                    shard_pattern=_SHARD_PATTERN,
                )
        else:
            raise FileNotFoundError(f"Denoiser checkpoint not found in {checkpoint_dir}")
        if not offloading:
            _materialize_meta_tensors(model)
    else:
        checkpoint_keys = _list_checkpoint_keys(quantized_checkpoint_path)
        replace_targeted_linear_modules(
            model,
            method=high_method,
            target_linear_names=high_linear,
            quantize_weights=False,
            checkpoint_keys=checkpoint_keys,
        )
        replace_targeted_linear_modules(
            model,
            method=low_method,
            target_linear_names=low_linear,
            quantize_weights=False,
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

    # ---- Final dtypes ----
    if quant_method is None and not offloading:
        model = model.to(dtype=torch.float16)
    else:
        _cast_float_tensors_except_quantized_linear(model, torch.float16)
    _cast_qk_norms_to_fp32(model)

    model.eval()
    return model
