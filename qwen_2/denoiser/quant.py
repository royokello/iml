#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from pathlib import Path

from safetensors.torch import save_file

from utils.quant.model import quantize_model_tensors
from utils.quant.targets import build_mixed_target_config
from utils.text import replace_hyphens_with_underscores

from qwen_2.config import VARIANTS, resolve_model_root
from qwen_2.denoiser.targets import _build_qwen2_denoiser_target_tensors


def _quantize_denoiser(
    root: str | Path,
    input_dirpath: str | Path,
    method: str,
    variant: str,
) -> Path:
    """Quantize the Qwen Image 2.1 denoiser.

    Args:
        root: The root folder. The output goes to
            `<root>/qwen_2/model/denoiser/<variant>`.
        input_dirpath: The folder that holds the shard files.
        method: The quantization method.
        variant: The checkpoint variant, `turbo` or `base`.

    Returns:
        The path of the quantized file.

    Raises:
        FileNotFoundError: If the input folder has no safetensors shard.
    """
    input_dir = Path(input_dirpath)
    model_files = sorted(input_dir.glob("*.safetensors"))
    if not model_files:
        raise FileNotFoundError(f"No .safetensors files found in {input_dir}")

    quant_name = replace_hyphens_with_underscores(method)
    output_dir = resolve_model_root(root) / "denoiser" / variant
    output_path = output_dir / f"{quant_name}_quant.safetensors"

    target_tensors = _build_qwen2_denoiser_target_tensors()
    targets = build_mixed_target_config(method, target_tensors)

    print(
        "Quantize denoiser args:\n"
        f"  root={root}\n"
        f"  input={input_dirpath}\n"
        f"  method={method}\n"
        f"  variant={variant}"
    )
    print("Targets")
    for target_method, target_names in targets.items():
        print(f"  {target_method}: {len(target_names)} tensors")

    print(f"Quantizing {len(model_files)} shard(s) with {method} ...")
    quant_start = time.perf_counter()
    state_dict = quantize_model_tensors(files=model_files, targets=targets)
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
            "model": "qwen_2",
            "variant": variant,
            "method": quant_name,
        },
    )
    save_seconds = time.perf_counter() - save_start
    print(f"Saved in {save_seconds:.3f}s")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize the Qwen Image 2.1 denoiser.")
    parser.add_argument("--root", required=True, help="Root folder to receive the quantized output.")
    parser.add_argument("--input", required=True, help="Source directory containing .safetensors shards.")
    parser.add_argument(
        "--variant",
        required=True,
        choices=VARIANTS,
        help="Checkpoint variant. The output goes to <root>/qwen_2/model/denoiser/<variant>/.",
    )
    parser.add_argument("--method", default="q6-mini", help="Quantization method (default: q6-mini).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = _quantize_denoiser(args.root, args.input, method=args.method, variant=args.variant)
    print(f"Saved quantized denoiser to {output_path}")


if __name__ == "__main__":
    main()
