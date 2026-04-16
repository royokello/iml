from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch
from safetensors import safe_open

from utils.loaders.single import load_local_single_checkpoint
from utils.quant.linear import QuantizedLinear

from .transformer import Flux2Transformer2DModel

DEFAULT_TARGET_LINEAR_NAMES = (
    "to_q",
    "to_k",
    "to_v",
    "to_out",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",
    "to_qkv_mlp_proj",
    "linear_in",
    "linear_out",
    "x_embedder",
    "context_embedder",
)


def _resolve_model_dir(path: str | Path) -> Path:
    model_path = Path(path)
    if model_path.is_file():
        return model_path.parent
    return model_path


def _resolve_transformer_dirs(path: str | Path, variant: str) -> tuple[Path, Path, str]:
    resolved_variant = variant.strip().lower()
    model_path = _resolve_model_dir(path).expanduser().resolve()

    if model_path.name in {"base", "distill"} and model_path.parent.name == "transformer":
        checkpoint_dir = model_path
        model_dir = model_path.parent
        path_variant = model_path.name
        if path_variant != resolved_variant:
            raise ValueError(
                f"Transformer variant mismatch: path points to {path_variant!r}, variant={resolved_variant!r}."
            )
        return model_dir, checkpoint_dir, resolved_variant

    return model_path, model_path / resolved_variant, resolved_variant


def _parse_target_linear_names(value: str | None) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_TARGET_LINEAR_NAMES
    items = tuple(part.strip() for part in value.split(",") if part.strip())
    return items or DEFAULT_TARGET_LINEAR_NAMES

def _materialize_meta_tensors(model: torch.nn.Module) -> None:
    unresolved_parameters = [name for name, parameter in model.named_parameters() if getattr(parameter, "is_meta", False)]
    if unresolved_parameters:
        raise RuntimeError("Checkpoint load left parameters on meta: " + ", ".join(unresolved_parameters))

    unresolved_buffers = [name for name, buffer in model.named_buffers() if getattr(buffer, "is_meta", False)]
    if unresolved_buffers:
        raise RuntimeError("Checkpoint load left buffers on meta: " + ", ".join(unresolved_buffers))


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


def _replace_targeted_linear_modules(
    module: torch.nn.Module,
    *,
    method: str,
    target_linear_names: tuple[str, ...],
    quantize_weights: bool = True,
    path: tuple[str, ...] = (),
    checkpoint_keys: set[str] | None = None,
) -> None:
    for name, child in list(module.named_children()):
        child_path = (*path, name)
        if isinstance(child, torch.nn.Linear) and _is_targeted_linear(
            child_path,
            target_linear_names,
            checkpoint_keys=checkpoint_keys,
        ):
            setattr(
                module,
                name,
                _build_quantized_linear(
                    child,
                    method=method,
                    quantize_weights=quantize_weights,
                ),
            )
            continue
        _replace_targeted_linear_modules(
            child,
            method=method,
            target_linear_names=target_linear_names,
            quantize_weights=quantize_weights,
            path=child_path,
            checkpoint_keys=checkpoint_keys,
        )


def _is_targeted_linear(
    module_path: tuple[str, ...],
    target_linear_names: tuple[str, ...],
    *,
    checkpoint_keys: set[str] | None = None,
) -> bool:
    if not module_path:
        return False

    full_name = ".".join(module_path)
    is_target = (
        full_name in target_linear_names
        or module_path[-1] in target_linear_names
        or (module_path[-1].isdigit() and len(module_path) > 1 and module_path[-2] in target_linear_names)
    )
    if not is_target:
        return False

    if checkpoint_keys is None:
        return True

    return f"{full_name}.scales" in checkpoint_keys


def _build_quantized_linear(
    linear: torch.nn.Linear,
    *,
    method: str,
    quantize_weights: bool,
) -> torch.nn.Module:
    if quantize_weights:
        return QuantizedLinear(linear, method=method)
    return QuantizedLinear.from_prequantized(linear, method=method)


def load_flux2_denoiser(
    path: str | Path,
    quant_method: str | None = None,
    target_linear_names: tuple[str, ...] = DEFAULT_TARGET_LINEAR_NAMES,
    variant: str = "distill",
) -> Flux2Transformer2DModel:
    model_dir, checkpoint_dir, variant = _resolve_transformer_dirs(path, variant)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Denoiser directory not found: {model_dir}")
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Denoiser checkpoint directory not found: {checkpoint_dir}")

    if quant_method is not None:
        quant_method = quant_method.lower()
        if quant_method not in {"single", "double"}:
            raise ValueError('quant_method must be "single", "double", or None.')

    model = _build_model_from_config(model_dir)
    quantized_checkpoint_path = None
    if quant_method is not None:
        quantized_filename = f"{quant_method}_quant.safetensors"
        quantized_checkpoint_path = checkpoint_dir / quantized_filename
        if not quantized_checkpoint_path.is_file():
            print(f"    saved quantized checkpoint not found in {checkpoint_dir}; quantizing on the fly")
            quantized_checkpoint_path = None
        else:
            print(f"    using saved quantized checkpoint: {quantized_checkpoint_path.name}")

    if quantized_checkpoint_path is None:
        checkpoint_path = checkpoint_dir / "diffusion_pytorch_model.safetensors"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Denoiser checkpoint not found: {checkpoint_path}")
        incompatible = load_local_single_checkpoint(model, checkpoint_path)
        if incompatible.unexpected_keys:
            raise RuntimeError("Unexpected keys in denoiser checkpoint: " + ", ".join(sorted(incompatible.unexpected_keys)))
        _materialize_meta_tensors(model)
    else:
        checkpoint_keys = _list_checkpoint_keys(quantized_checkpoint_path)
        _replace_targeted_linear_modules(
            model,
            method=quant_method,
            target_linear_names=target_linear_names,
            quantize_weights=False,
            checkpoint_keys=checkpoint_keys,
        )
        incompatible = load_local_single_checkpoint(model, quantized_checkpoint_path)
        if incompatible.unexpected_keys:
            raise RuntimeError("Unexpected keys in denoiser checkpoint: " + ", ".join(sorted(incompatible.unexpected_keys)))
        _materialize_meta_tensors(model)
    if quant_method is not None and quantized_checkpoint_path is None:
        _replace_targeted_linear_modules(
            model,
            method=quant_method,
            target_linear_names=target_linear_names,
        )
    model = model.to(dtype=torch.float16)
    model.eval()
    return model
