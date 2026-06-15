#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from safetensors.torch import save_file

from flux2.denoiser.targets import _build_flux2_denoiser_target_tensors
from utils.quant.model import quantize_model_tensors
from utils.quant.targets import build_mixed_target_config

def _build_checkpoint_files(checkpoint_dir: Path, version: str) -> list[Path]:
    if version == "4b":
        return [checkpoint_dir / "diffusion_pytorch_model.safetensors"]
    return [
        checkpoint_dir / "diffusion_pytorch_model-00001-of-00002.safetensors",
        checkpoint_dir / "diffusion_pytorch_model-00002-of-00002.safetensors",
    ]

def _quantize_denoiser(
    root: str | Path,
    input_root: str | Path,
    *,
    version: str,
    method: str,
    variant: str,
) -> Path:
    output_model_dir = Path(root) / "flux2" / version / "model" / "transformer"
    checkpoint_dir = Path(input_root)
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Transformer checkpoint directory not found: {checkpoint_dir}")

    checkpoint_files = _build_checkpoint_files(checkpoint_dir, version)
    print(
        "Quantize denoiser args:\n"
        f"  root={root}\n"
        f"  input={input_root}\n"
        f"  version={version}\n"
        f"  method={method}\n"
        f"  variant={variant}"
    )
    print("Checkpoint")
    for checkpoint in checkpoint_files:
        print(f" * {checkpoint}")

    target_tensors = _build_flux2_denoiser_target_tensors(version)
    targets = build_mixed_target_config(method, target_tensors)

    print("Targets")
    for target_method, target_names in targets.items():
        print(f" * {target_method}: {len(target_names)} tensors")

    quant_name = method.replace("-", "_")
    output_path = output_model_dir / variant / f"{quant_name}_quant.safetensors"

    print(f"Applying {method} mixed‑precision quantization for Flux2 {version} denoiser ({variant}) ...")
    quantize_start = time.perf_counter()
    state_dict = quantize_model_tensors(
        files=checkpoint_files,
        targets=build_mixed_target_config(method, target_tensors),
    )
    quantize_seconds = time.perf_counter() - quantize_start
    print(f"Quantized in {quantize_seconds:.3f}s")

    print(f"Saving {output_path} ...")
    save_start = time.perf_counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        state_dict,
        str(output_path),
        metadata={
            "model": "flux2",
            "component": "denoiser",
            "version": version,
            "variant": variant,
            "method": method,
            "application": "IML",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    save_seconds = time.perf_counter() - save_start
    print(f"Saved in {save_seconds:.3f}s")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize and save the Flux2 denoiser.")
    parser.add_argument(
        "--root",
        required=True,
        help="Root folder where flux2_4b or flux2_9b should receive the quantized output.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Full source transformer checkpoint directory. Bypasses source variant selection.",
    )
    parser.add_argument(
        "--version",
        choices=("4b", "9b"),
        required=True,
        help="Flux2 denoiser version to quantize.",
    )
    parser.add_argument(
        "--method",
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
    output_path = _quantize_denoiser(
        args.root,
        args.input,
        version=args.version,
        method=args.method,
        variant=args.variant,
    )
    print(f"Saved quantized denoiser to {output_path}")


if __name__ == "__main__":
    main()
