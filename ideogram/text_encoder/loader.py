from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open

from transformers import AutoConfig

from utils.loaders.single import load_local_single_checkpoint
from utils.quant.replace import replace_targeted_linear_modules, target_tensors_to_linear_names
from utils.quant.targets import build_mixed_target_config

from ideogram.text_encoder.model import Qwen3VLTextModel
from ideogram.text_encoder.target import _build_ideogram_text_encoder_targets


def _list_checkpoint_keys(checkpoint_path: Path) -> set[str]:
    lm_prefix = "language_model."
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        return {
            (name.removeprefix(lm_prefix) if name.startswith(lm_prefix) else name)
            for name in handle.keys()
        }


def load_ideogram_text_encoder(
    model_path: str | Path,
    quant_method: str,
    config_path: str | Path,
    *,
    offloading: bool = False,
) -> nn.Module:
    model_path = Path(model_path)
    config_path = Path(config_path)

    quant_name = quant_method.replace("-", "_")
    quant_path = model_path.parent / f"{quant_name}_quant.safetensors"
    if not quant_path.is_file():
        raise FileNotFoundError(
            f"Quantized checkpoint not found: {quant_path}. "
            "Run the text encoder quantizer first."
        )

    checkpoint_keys = _list_checkpoint_keys(quant_path)

    target_tensors = _build_ideogram_text_encoder_targets()
    config = build_mixed_target_config(quant_method, target_tensors)

    model_config = AutoConfig.from_pretrained(str(config_path), trust_remote_code=True)
    text_config = model_config.text_config

    default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float16)
        with torch.device("meta"):
            model = Qwen3VLTextModel(text_config)
    finally:
        torch.set_default_dtype(default_dtype)

    lm_prefix = "language_model."

    def _strip_lm_prefix(name: str) -> str:
        if name.startswith(lm_prefix):
            return name[len(lm_prefix):]
        return name

    print(f"    quantized checkpoint: {quant_path.name}")
    for method, tensor_names in config.items():
        if method in ("fp32", "fp16"):
            continue
        linear_names = target_tensors_to_linear_names(tensor_names, key_transform=_strip_lm_prefix)
        replace_targeted_linear_modules(
            model,
            method=method,
            target_linear_names=linear_names,
            quantize_weights=False,
            checkpoint_keys=checkpoint_keys,
        )

    incompatible = load_local_single_checkpoint(
        model,
        quant_path,
        key_transform=_strip_lm_prefix,
    )

    missing = incompatible.get("missing_keys", [])
    unexpected = incompatible.get("unexpected_keys", [])
    print(f"    load: missing={len(missing)} unexpected={len(unexpected)}")

    if missing:
        raise RuntimeError(
            "Text encoder checkpoint missing keys: "
            + ", ".join(sorted(missing))
        )

    model.eval()
    return model

