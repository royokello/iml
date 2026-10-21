#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

from flux3.config import resolve_model_root
from flux3.denoiser.targets import _build_flux3_denoiser_target_tensors
from utils.quant.model import quantize_model_tensors
from utils.quant.targets import build_mixed_target_config

_SHARD_PATTERN = "diffusion_pytorch_model-*.safetensors"


def _quantize_denoiser(
    root: str | Path,
    input_dir: str | Path | None,
    *,
    method: str,
    variant: str,
) -> Path:
    output_dir = resolve_model_root(root) / "transformer" / variant
    checkpoint_dir = Path(input_dir).expanduser() if input_dir is not None else output_dir
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Transformer checkpoint directory not found: {checkpoint_dir}")

    checkpoint_files = sorted(checkpoint_dir.glob(_SHARD_PATTERN))
    if not checkpoint_files:
        raise FileNotFoundError(f"No files match {_SHARD_PATTERN} in {checkpoint_dir}")

    print(
        "Quantize denoiser args:\n"
        f"  root={root}\n"
        f"  input={checkpoint_dir}\n"
        f"  method={method}\n"
        f"  variant={variant}"
    )
    print("Checkpoint")
    for checkpoint in checkpoint_files:
        print(f" * {checkpoint}")

    # The base checkpoint has no target_timestep_embedder. Keep only the names that exist.
    source_keys: set[str] = set()
    for checkpoint in checkpoint_files:
        with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
            source_keys.update(handle.keys())
    target_tensors = {
        group: [name for name in names if name in source_keys]
        for group, names in _build_flux3_denoiser_target_tensors().items()
    }
    targets = build_mixed_target_config(method, target_tensors)

    print("Targets")
    for target_method, target_names in targets.items():
        print(f" * {target_method}: {len(target_names)} tensors")

    quant_name = method.replace("-", "_")
    output_path = output_dir / f"{quant_name}_quant.safetensors"

    print(f"Applying {method} mixed-precision quantization for Flux3 denoiser ({variant}) ...")
    quantize_start = time.perf_counter()
    state_dict = quantize_model_tensors(files=checkpoint_files, targets=targets)
    print(f"Quantized in {time.perf_counter() - quantize_start:.3f}s")

    print(f"Saving {output_path} ...")
    save_start = time.perf_counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        state_dict,
        str(output_path),
        metadata={
            "model": "flux3",
            "component": "denoiser",
            "variant": variant,
            "method": method,
            "application": "IML",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(f"Saved in {time.perf_counter() - save_start:.3f}s")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize and save the Flux3 denoiser.")
    parser.add_argument(
        "--root",
        required=True,
        help="Root folder. The output goes to <root>/flux3/model/transformer/<variant>/. The folder is created if missing.",
    )
    parser.add_argument(
        "--variant",
        choices=("base", "distill"),
        required=True,
        help="Transformer checkpoint variant to read from and save into.",
    )
    parser.add_argument(
        "--method",
        required=True,
        help="Quantization method to apply (for example sym-med-nano).",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Source shard folder. Default: <root>/flux3/model/transformer/<variant>.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = _quantize_denoiser(args.root, args.input, method=args.method, variant=args.variant)
    print(f"Saved quantized denoiser to {output_path}")


if __name__ == "__main__":
    main()
