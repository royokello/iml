#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from pathlib import Path

from safetensors.torch import save_file

from utils.quant.targets import build_mixed_target_config
from utils.text import replace_hyphens_with_underscores

from ideogram.denoiser.targets import _build_ideogram_denoiser_target_tensors
from ideogram.quant_utils import quantize_f8_model_tensors

_VARIANT_DIRS = {
    "cond": "transformer",
    "uncond": "unconditional_transformer",
}


def _quantize_denoiser(
    root: str | Path,
    input_root: str | Path,
    *,
    method: str,
    variant: str,
) -> Path:
    input_root = Path(input_root)
    model_dir = input_root / _VARIANT_DIRS[variant]
    model_file = model_dir / "diffusion_pytorch_model.safetensors"
    if not model_file.is_file():
        raise FileNotFoundError(f"F8 model file not found: {model_file}")

    quant_name = replace_hyphens_with_underscores(method)
    output_dir = Path(root) / "ideogram" / "transformer" / variant
    output_path = output_dir / f"{quant_name}_quant.safetensors"

    target_tensors = _build_ideogram_denoiser_target_tensors()
    targets = build_mixed_target_config(method, target_tensors)

    print(
        "Quantize denoiser args:\n"
        f"  root={root}\n"
        f"  input={input_root}\n"
        f"  method={method}\n"
        f"  variant={variant}"
    )
    print("Targets")
    for target_method, target_names in targets.items():
        print(f"  {target_method}: {len(target_names)} tensors")

    print(f"Quantizing {model_file} with {method} ...")
    quant_start = time.perf_counter()
    state_dict = quantize_f8_model_tensors(files=model_file, targets=targets)
    quant_seconds = time.perf_counter() - quant_start
    print(f"Quantized in {quant_seconds:.3f}s")

    print(f"Saving {output_path} ...")
    save_start = time.perf_counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        state_dict,
        str(output_path),
        metadata={
            "component": "denoiser",
            "model": "ideogram",
            "variant": variant,
            "method": quant_name,
        },
    )
    save_seconds = time.perf_counter() - save_start
    print(f"Saved in {save_seconds:.3f}s")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dequantize F8 and quantize the Ideogram 4 denoiser."
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Root folder to receive the quantized output.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Source root directory containing transformer/ and unconditional_transformer/ subdirs.",
    )
    parser.add_argument(
        "--method",
        required=True,
        help="Quantization method to apply (e.g. sym-high-nano, aff-med-mini).",
    )
    parser.add_argument(
        "--variant",
        choices=("cond", "uncond"),
        required=True,
        help="Denoiser variant to quantize.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = _quantize_denoiser(
        args.root,
        args.input,
        method=args.method,
        variant=args.variant,
    )
    print(f"Saved quantized denoiser to {output_path}")


if __name__ == "__main__":
    main()
