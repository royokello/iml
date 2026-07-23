from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open

from utils.loaders.single import load_local_single_checkpoint, stream_local_single_checkpoint
from utils.quant.name import convert_quant_name
from utils.quant.replace import replace_targeted_linear_modules, target_tensors_to_linear_names
from utils.text import replace_hyphens_with_underscores

from anima.denoiser.models.denoiser import AnimaModel
from anima.denoiser.targets import _build_anima_denoiser_target_tensors


_VARIANT_MAP = {
    "base": "anima-base-v1.0.safetensors",
    "aesthetic": "anima-aesthetic-v1.1.safetensors",
    "turbo": "anima-turbo-v1.0.safetensors",
    "preview": "anima-preview2.safetensors",
}


def _strip_net_prefix(name: str) -> str:
    return name.removeprefix("net.")


def _list_keys(path: Path) -> set[str]:
    with safe_open(str(path), framework="pt", device="cpu") as f:
        return set(f.keys())


def _load_anima_denoiser(
    path: str | Path,
    variant: str = "base",
    quant_method: str | None = None,
) -> AnimaModel:
    variant = variant.strip().lower()
    if variant not in _VARIANT_MAP:
        raise ValueError(f"Unknown variant {variant!r}. Options: {', '.join(_VARIANT_MAP)}")

    model_path = Path(path)
    model_dir = model_path.parent if model_path.is_file() else model_path

    if quant_method is not None:
        quant_method_key = replace_hyphens_with_underscores(quant_method)
        high_method, low_method = convert_quant_name(quant_method)
    else:
        high_method = low_method = None

    target_tensors = _build_anima_denoiser_target_tensors()
    high_linear = target_tensors_to_linear_names(target_tensors["high"], key_transform=_strip_net_prefix)
    low_linear = target_tensors_to_linear_names(target_tensors["low"], key_transform=_strip_net_prefix)

    model = AnimaModel()

    variant_dir = model_dir / variant

    quantized_path = None
    if quant_method is not None:
        quantized_path = variant_dir / f"{quant_method_key}_quant.safetensors"
        if quantized_path.is_file():
            print(f"    using saved quantized checkpoint: {quantized_path.name}")
        else:
            print(f"    quantized checkpoint not found; quantizing on the fly")
            quantized_path = None

    if quantized_path is None:
        ckpt_path = variant_dir / _VARIANT_MAP[variant]
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Denoiser checkpoint not found: {ckpt_path}")
        load_local_single_checkpoint(model, ckpt_path, key_transform=_strip_net_prefix)
        if quant_method is not None and quant_method != "fp16":
            replace_targeted_linear_modules(
                model, method=high_method, target_linear_names=high_linear,
            )
            replace_targeted_linear_modules(
                model, method=low_method, target_linear_names=low_linear,
            )
    else:
        if quant_method == "fp16":
            stream_local_single_checkpoint(model, quantized_path, key_transform=_strip_net_prefix)
        else:
            checkpoint_keys = {_strip_net_prefix(k) for k in _list_keys(quantized_path)}
            replace_targeted_linear_modules(
                model, method=high_method, target_linear_names=high_linear,
                quantize_weights=False, checkpoint_keys=checkpoint_keys,
            )
            replace_targeted_linear_modules(
                model, method=low_method, target_linear_names=low_linear,
                quantize_weights=False, checkpoint_keys=checkpoint_keys,
            )
            load_local_single_checkpoint(model, quantized_path, key_transform=_strip_net_prefix)

    model.eval()
    return model
