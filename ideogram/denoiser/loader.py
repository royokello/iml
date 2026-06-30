from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import load_file as safe_load_file

from utils.loaders.single import load_local_single_checkpoint
from utils.quant.linear import QuantizedLinear
from utils.quant.replace import replace_targeted_linear_modules, target_tensors_to_linear_names
from utils.quant.targets import build_mixed_target_config

from ideogram.denoiser.targets import _build_ideogram_denoiser_target_tensors
from ideogram.models import Ideogram4Config, Ideogram4MRoPE, Ideogram4Transformer


def _list_checkpoint_keys(checkpoint_path: Path) -> set[str]:
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        return set(handle.keys())


def _materialize_meta_buffers(model: nn.Module) -> None:
    rope_theta = model.config.rope_theta
    for module in model.modules():
        if isinstance(module, Ideogram4MRoPE):
            inv = getattr(module, "inv_freq", None)
            if inv is not None and getattr(inv, "is_meta", False):
                head_dim = module.head_dim
                inv_freq = 1.0 / (
                    rope_theta
                    ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
                )
                module.register_buffer("inv_freq", inv_freq, persistent=False)

    unresolved = [
        name
        for name, parameter in model.named_parameters()
        if getattr(parameter, "is_meta", False)
    ]
    if unresolved:
        raise RuntimeError(
            "Checkpoint load left parameters on meta: " + ", ".join(unresolved)
        )

    unresolved = [
        name
        for name, buffer in model.named_buffers()
        if getattr(buffer, "is_meta", False)
    ]
    if unresolved:
        raise RuntimeError(
            "Checkpoint load left buffers on meta: " + ", ".join(unresolved)
        )

def pin_module_parameters(model: nn.Module) -> int:
  pinned = 0
  for param_name, param in list(model.named_parameters(recurse=True)):
    if param is None or not param.is_floating_point() or param.device.type != "cpu":
      continue
    if not param.is_contiguous():
      param.data = param.data.contiguous()
    pinned_tensor = torch.empty_like(param.data, device="cpu", pin_memory=True)
    pinned_tensor.copy_(param.data)
    module_path, _, local_name = param_name.rpartition(".")
    parent = model.get_submodule(module_path) if module_path else model
    parent._parameters[local_name] = nn.Parameter(pinned_tensor, requires_grad=param.requires_grad)
    pinned += 1
  for buf_name, buf in list(model.named_buffers(recurse=True)):
    if buf is None or not buf.is_floating_point() or buf.device.type != "cpu":
      continue
    if not buf.is_contiguous():
      buf = buf.contiguous()
    pinned_buf = torch.empty_like(buf, device="cpu", pin_memory=True)
    pinned_buf.copy_(buf)
    module_path, _, local_name = buf_name.rpartition(".")
    parent = model.get_submodule(module_path) if module_path else model
    parent._buffers[local_name] = pinned_buf
    pinned += 1
  return pinned


def load_ideogram_transformer(
    path: str | Path,
    quant_method: str,
    *,
    offloading: bool = False,
) -> Ideogram4Transformer:
    path = Path(path)
    model_dir = path.parent

    quant_name = quant_method.replace("-", "_")
    quant_path = model_dir / f"{quant_name}_quant.safetensors"
    if not quant_path.is_file():
        raise FileNotFoundError(
            f"Quantized checkpoint not found: {quant_path}. "
            "Run the denoiser quantizer first."
        )

    checkpoint_keys = _list_checkpoint_keys(quant_path)

    target_tensors = _build_ideogram_denoiser_target_tensors()
    config = build_mixed_target_config(quant_method, target_tensors)

    with torch.device("meta"):
        model = Ideogram4Transformer(Ideogram4Config())

    for method, tensor_names in config.items():
        if method in ("fp32", "fp16"):
            continue
        linear_names = target_tensors_to_linear_names(tensor_names)
        replace_targeted_linear_modules(
            model,
            method=method,
            target_linear_names=linear_names,
            quantize_weights=False,
            checkpoint_keys=checkpoint_keys,
        )

    incompatible = load_local_single_checkpoint(model, quant_path)
    _materialize_meta_buffers(model)
    if incompatible.get("unexpected_keys"):
        raise RuntimeError(
            "Unexpected keys in denoiser checkpoint: "
            + ", ".join(sorted(incompatible["unexpected_keys"]))
        )

    model.eval()
    return model


def load_ideogram_transformer_pair(
    root: str | Path,
    quant_method: str,
    *,
    offloading: bool = False,
) -> tuple[Ideogram4Transformer, Ideogram4Transformer]:
    root = Path(root)
    cond = load_ideogram_transformer(
        root / "transformer" / "cond" / "diffusion_pytorch_model.safetensors",
        quant_method,
        offloading=offloading,
    )
    uncond = load_ideogram_transformer(
        root / "transformer" / "uncond" / "diffusion_pytorch_model.safetensors",
        quant_method,
        offloading=offloading,
    )
    return cond, uncond
