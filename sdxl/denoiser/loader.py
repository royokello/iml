from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open

from sdxl.denoiser.targets import classify_linear_module
from sdxl.models._unet import SDXLUNet
from utils.loaders.single import load_local_single_checkpoint
from utils.quant.linear import QuantizedLinear
from utils.quant.name import convert_quant_name
from utils.quant.replace import replace_targeted_linear_modules

SINGLE_CHECKPOINT = "diffusion_pytorch_model.safetensors"


def _list_checkpoint_keys(checkpoint_path: Path) -> set[str]:
    """Return the tensor names in a safetensors file.

    Args:
        checkpoint_path: The checkpoint file.

    Returns:
        The tensor names.
    """
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        return set(handle.keys())


def _materialize_meta_tensors(model: torch.nn.Module) -> None:
    """Fail when the checkpoint load left a tensor on the meta device.

    Args:
        model: The loaded model.

    Raises:
        RuntimeError: If a parameter or buffer is still on the meta device.
    """
    unresolved_parameters = [
        name for name, parameter in model.named_parameters() if getattr(parameter, "is_meta", False)
    ]
    unresolved_buffers = [
        name for name, buffer in model.named_buffers() if getattr(buffer, "is_meta", False)
    ]
    if unresolved_parameters or unresolved_buffers:
        raise RuntimeError(
            "Checkpoint load left tensors on meta. "
            f"Parameters: {', '.join(unresolved_parameters) or 'none'}. "
            f"Buffers: {', '.join(unresolved_buffers) or 'none'}."
        )


def _cast_float_tensors_to_fp16(module: torch.nn.Module) -> None:
    """Cast every floating parameter and buffer to fp16.

    Quantized linears keep their own dtypes and are skipped.

    Args:
        module: The module to cast, including its children.
    """
    if isinstance(module, QuantizedLinear):
        return

    for name, parameter in module.named_parameters(recurse=False):
        if parameter is not None and parameter.is_floating_point() and parameter.dtype != torch.float16:
            module._parameters[name] = torch.nn.Parameter(
                parameter.to(dtype=torch.float16),
                requires_grad=parameter.requires_grad,
            )

    for name, buffer in module.named_buffers(recurse=False):
        if buffer is not None and buffer.is_floating_point() and buffer.dtype != torch.float16:
            module._buffers[name] = buffer.to(dtype=torch.float16)

    for child in module.children():
        _cast_float_tensors_to_fp16(child)


def _linear_targets(model: torch.nn.Module) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split the model linear modules into the high and low target groups.

    Args:
        model: The UNet.

    Returns:
        The high and low module names. Modules in the `fp16` group are omitted.
    """
    high: list[str] = []
    low: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        group = classify_linear_module(name)
        if group == "high":
            high.append(name)
        elif group == "low":
            low.append(name)
    return tuple(high), tuple(low)


def _build_model() -> SDXLUNet:
    """Build an empty UNet on the meta device.

    Returns:
        The UNet, with every tensor on the meta device.
    """
    default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float16)
        with torch.device("meta"):
            model = SDXLUNet()
    finally:
        torch.set_default_dtype(default_dtype)
    return model


def load_sdxl_denoiser(
    model_root: str | Path,
    quant_method: str | None = None,
    *,
    device: torch.device | str | None = None,
) -> SDXLUNet:
    """Load the SDXL UNet from a model folder.

    Args:
        model_root: The model folder, or its `unet` folder.
        quant_method: A quantization method name, or None for the fp16 weights.
        device: Where to put the model. The default leaves it on the CPU.

    Returns:
        The UNet in eval mode. Float weights are fp16. `QuantizedLinear` layers
        keep their own dtypes.

    Raises:
        FileNotFoundError: If the folder, the fp16 checkpoint, or the quantized
            checkpoint is missing. The loader never quantizes on the fly.
        RuntimeError: If the checkpoint keys do not match the model, or if a
            tensor stays on the meta device.
    """
    root = Path(model_root).expanduser().resolve()
    unet_dir = root if root.name == "unet" else root / "unet"
    if not unet_dir.is_dir():
        raise FileNotFoundError(f"UNet folder not found: {unet_dir}")

    model = _build_model()

    if quant_method is None:
        checkpoint_path = unet_dir / SINGLE_CHECKPOINT
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"UNet checkpoint not found: {checkpoint_path}")
    else:
        checkpoint_path = unet_dir / f"{quant_method.replace('-', '_')}_quant.safetensors"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Quantized checkpoint not found: {checkpoint_path}. Run sdxl.denoiser.quant first."
            )
        print(f"    using saved quantized checkpoint: {checkpoint_path.name}", flush=True)

        high_method, low_method = convert_quant_name(quant_method)
        checkpoint_keys = _list_checkpoint_keys(checkpoint_path)
        high_targets, low_targets = _linear_targets(model)
        replace_targeted_linear_modules(
            model,
            method=high_method,
            target_linear_names=high_targets,
            quantize_weights=False,
            checkpoint_keys=checkpoint_keys,
        )
        replace_targeted_linear_modules(
            model,
            method=low_method,
            target_linear_names=low_targets,
            quantize_weights=False,
            checkpoint_keys=checkpoint_keys,
        )

    incompatible = load_local_single_checkpoint(model, checkpoint_path)
    if incompatible["unexpected_keys"]:
        raise RuntimeError(
            "Unexpected keys in UNet checkpoint: "
            + ", ".join(sorted(incompatible["unexpected_keys"]))
        )
    if incompatible["missing_keys"]:
        raise RuntimeError(
            "Missing keys in UNet checkpoint: " + ", ".join(sorted(incompatible["missing_keys"]))
        )

    _materialize_meta_tensors(model)
    _cast_float_tensors_to_fp16(model)
    model.eval()

    if device is not None:
        model.to(device)
    return model


__all__ = ["load_sdxl_denoiser"]
