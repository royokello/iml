#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch.nn as nn
from safetensors.torch import save_file

from flux_2_klein_4b.denoiser.loader import (
    DEFAULT_TARGET_LINEAR_NAMES,
    _build_model_from_config,
    _load_local_single_checkpoint,
    _materialize_meta_tensors,
)
from utils.quant.double.to import quantize_to_double_block
from utils.quant.single.to import quantize_to_single_block


def _prepare_tensor(tensor):
    return tensor.detach().cpu().contiguous().clone()


def _quantize_linear(module: nn.Linear, *, method: str) -> dict[str, object]:
    if method == "single":
        weight, scales = quantize_to_single_block(module.weight.detach())
        return {
            "weight": _prepare_tensor(weight),
            "scales": _prepare_tensor(scales),
        }

    weight, scales, super_scales = quantize_to_double_block(module.weight.detach())
    return {
        "weight": _prepare_tensor(weight),
        "scales": _prepare_tensor(scales),
        "super_scales": _prepare_tensor(super_scales),
    }


def _is_targeted_linear(module_path: tuple[str, ...], target_linear_names: tuple[str, ...]) -> bool:
    if not module_path:
        return False

    full_name = ".".join(module_path)
    return (
        full_name in target_linear_names
        or module_path[-1] in target_linear_names
        or (module_path[-1].isdigit() and len(module_path) > 1 and module_path[-2] in target_linear_names)
    )


def _build_quantized_state_dict(model, *, method: str) -> dict[str, object]:
    prepared: dict[str, object] = {}
    replaced_keys: set[str] = set()

    for module_name, module in model.named_modules():
        module_path = tuple(part for part in module_name.split(".") if part)
        if not isinstance(module, nn.Linear) or not _is_targeted_linear(module_path, DEFAULT_TARGET_LINEAR_NAMES):
            continue

        prefix = f"{module_name}." if module_name else ""
        for suffix, tensor in _quantize_linear(module, method=method).items():
            prepared[f"{prefix}{suffix}"] = tensor

        replaced_keys.add(f"{prefix}weight")
        if module.bias is not None:
            bias_key = f"{prefix}bias"
            prepared[bias_key] = _prepare_tensor(module.bias)
            replaced_keys.add(bias_key)

    for name, tensor in model.state_dict().items():
        if name in replaced_keys:
            continue
        prepared[name] = _prepare_tensor(tensor)

    return prepared


def quantize_denoiser(root: str | Path, *, method: str, variant: str) -> Path:
    method = method.strip().lower()
    variant = variant.strip().lower()
    model_dir = Path(root).expanduser().resolve() / "flux_2_klein_4b" / "model" / "transformer"
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Denoiser directory not found: {model_dir}")

    input_checkpoint_path = model_dir / variant / "diffusion_pytorch_model.safetensors"
    if not input_checkpoint_path.is_file():
        raise FileNotFoundError(f"Denoiser checkpoint not found: {input_checkpoint_path}")

    output_path = model_dir / variant / f"{method}_quant.safetensors"

    print("Loading denoiser ...")
    load_start = time.perf_counter()
    model = _build_model_from_config(model_dir)
    _load_local_single_checkpoint(model, input_checkpoint_path)
    _materialize_meta_tensors(model)
    load_seconds = time.perf_counter() - load_start
    print(f"Loaded in {load_seconds:.3f}s")

    print(f"Applying {method} quantization ...")
    quantize_start = time.perf_counter()
    state_dict = _build_quantized_state_dict(model, method=method)
    quantize_seconds = time.perf_counter() - quantize_start
    print(f"Quantized in {quantize_seconds:.3f}s")

    print(f"Saving {output_path} ...")
    save_start = time.perf_counter()
    save_file(
        state_dict,
        str(output_path),
        metadata={
            "component": "denoiser",
            "variant": variant,
            "method": method,
        },
    )
    save_seconds = time.perf_counter() - save_start
    print(f"Saved in {save_seconds:.3f}s")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize and save the Flux 2 Klein 4B denoiser.")
    parser.add_argument(
        "--root",
        required=True,
        help="Root folder that contains flux_2_klein_4b/model/transformer.",
    )
    parser.add_argument(
        "--method",
        choices=("single", "double"),
        required=True,
        help="Quantization method to apply.",
    )
    parser.add_argument(
        "--variant",
        choices=("base", "distill"),
        required=True,
        help="Transformer checkpoint variant to read from and save into.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = quantize_denoiser(
        args.root,
        method=args.method,
        variant=args.variant,
    )
    print(f"Saved quantized denoiser to {output_path}")


if __name__ == "__main__":
    main()
