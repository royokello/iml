#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from pathlib import Path

from safetensors.torch import save_file

from utils.quant.model import quantize_model_tensors
from utils.quant.targets import build_mixed_target_config

from flux2.text_encoder.target import _build_target_tensors
from utils.text import replace_hyphens_with_underscores


def _quantize_text_encoder(
    root: str | Path,
    input_dirpath: str | Path,
    version: str,
    method: str,
) -> Path:
    print(
        "Quantize text encoder args:\n"
        f"  root={root}\n"
        f"  input_dirpath={input_dirpath}\n"
        f"  version={version}\n"
        f"  method={method}"
    )

    # Build output filename (use underscored version of original input)
    quant_name = replace_hyphens_with_underscores(method)
    output_dir = Path(root) / f"flux2_{version}" / "model" / "text_encoder"
    model_dir = Path(input_dirpath)

    if not model_dir.is_dir():
        raise FileNotFoundError(f"Text encoder directory not found: {model_dir}")

    if version == "4b":
        checkpoint_files = [
            model_dir / "model-00001-of-00002.safetensors",
            model_dir / "model-00002-of-00002.safetensors",
        ]
    else:
        checkpoint_files = [
            model_dir / "model-00001-of-00004.safetensors",
            model_dir / "model-00002-of-00004.safetensors",
            model_dir / "model-00003-of-00004.safetensors",
            model_dir / "model-00004-of-00004.safetensors",
        ]

    print("Checkpoint")
    for checkpoint in checkpoint_files:
        print(f" * {checkpoint}")

    # Get high/low target tensors from the shared builder
    target_tensors = _build_target_tensors()
    targets = build_mixed_target_config(method, target_tensors)

    print("Targets")
    for target_method, target_names in targets.items():
        print(f" * {target_method}: {len(target_names)} tensors")

    output_path = output_dir / f"{quant_name}_quant.safetensors"

    print(f"Applying {method} mixed‑precision quantization for Flux2 {version} text encoder ...")
    quantize_start = time.perf_counter()

    state_dict = quantize_model_tensors(
        files=checkpoint_files,
        targets=targets,
    )

    # Exclude lm_head weights if present
    state_dict = {
        name: tensor
        for name, tensor in state_dict.items()
        if not name.startswith("lm_head.")
    }

    quantize_seconds = time.perf_counter() - quantize_start
    print(f"Quantized in {quantize_seconds:.3f}s")

    print(f"Saving {output_path} ...")
    save_start = time.perf_counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        state_dict,
        str(output_path),
        metadata={
            "component": "text_encoder",
            "version": version,
            "method": quant_name,   # store the mixed method name
        },
    )
    save_seconds = time.perf_counter() - save_start
    print(f"Saved in {save_seconds:.3f}s")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize and save the Flux2 text encoder.")
    parser.add_argument(
        "--root",
        required=True,
        help="Root folder where flux2_4b or flux2_9b is stored or should receive the quantized output.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Full source text_encoder model directory.",
    )
    parser.add_argument(
        "--version",
        choices=("4b", "9b"),
        required=True,
        help="Flux2 text encoder version to quantize.",
    )
    parser.add_argument(
        "--method",
        required=True,
        help="Quantization method to apply (e.g., sym-high-nano, aff-med-mini).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = _quantize_text_encoder(
        args.root,
        args.input,
        version=args.version,
        method=args.method,
    )
    print(f"Saved quantized text encoder to {output_path}")


if __name__ == "__main__":
    main()
