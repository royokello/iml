from __future__ import annotations

from pathlib import Path

import torch

from utils.loaders.single import load_local_single_checkpoint
from utils.quant.name import convert_quant_name
from utils.quant.replace import replace_targeted_linear_modules, target_tensors_to_linear_names
from utils.text import replace_hyphens_with_underscores

from anima.text_encoder.models.model import Qwen3Model
from anima.text_encoder.targets import _build_anima_text_encoder_targets


def _strip_prefix(name: str) -> str:
    return name.removeprefix("model.")


def _load_anima_text_encoder(
    path: str | Path,
    quant_method: str | None = None,
) -> Qwen3Model:
    model_path = Path(path)
    model_dir = model_path.parent if model_path.is_file() else model_path

    if quant_method is not None:
        quant_method_key = replace_hyphens_with_underscores(quant_method)
        high_method, low_method = convert_quant_name(quant_method)
    else:
        high_method = low_method = None

    target_tensors = _build_anima_text_encoder_targets()
    high_linear = target_tensors_to_linear_names(target_tensors["high"], key_transform=_strip_prefix)
    low_linear = target_tensors_to_linear_names(target_tensors["low"], key_transform=_strip_prefix)

    model = Qwen3Model()

    quantized_path = None
    if quant_method is not None:
        quantized_path = model_dir / f"{quant_method_key}_quant.safetensors"
        if quantized_path.is_file():
            print(f"    using saved quantized checkpoint: {quantized_path.name}")
        else:
            print(f"    quantized checkpoint not found at {quantized_path}; quantizing on the fly")
            quantized_path = None

    if quantized_path is None:
        ckpt_path = model_dir / "qwen_3_06b_base.safetensors"
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Text encoder checkpoint not found: {ckpt_path}")
        load_local_single_checkpoint(model, ckpt_path, key_transform=_strip_prefix)
        if quant_method is not None and quant_method != "fp16":
            replace_targeted_linear_modules(
                model, method=high_method, target_linear_names=high_linear,
            )
            replace_targeted_linear_modules(
                model, method=low_method, target_linear_names=low_linear,
            )
    else:
        if quant_method == "fp16":
            load_local_single_checkpoint(model, quantized_path, key_transform=_strip_prefix)
        else:
            checkpoint_keys = {_strip_prefix(k) for k in _list_keys(quantized_path)}
            replace_targeted_linear_modules(
                model, method=high_method, target_linear_names=high_linear,
                quantize_weights=False, checkpoint_keys=checkpoint_keys,
            )
            replace_targeted_linear_modules(
                model, method=low_method, target_linear_names=low_linear,
                quantize_weights=False, checkpoint_keys=checkpoint_keys,
            )
            load_local_single_checkpoint(model, quantized_path, key_transform=_strip_prefix)

    model.eval()
    return model


def _list_keys(path: Path) -> set[str]:
    from safetensors import safe_open
    with safe_open(str(path), framework="pt", device="cpu") as f:
        return set(f.keys())
