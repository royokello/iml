#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from pathlib import Path

from safetensors.torch import save_file

from utils.quant.model import quantize_model_tensors
from utils.quant.name import convert_quant_name
from utils.quant.validators import normalize_quant_method

from flux2.text_encoder.quant import _build_target_tensors
from utils.text import replace_hyphens_with_underscores

_MODEL_SHARDS = {
    "4b": 2,
    "9b": 4,
}

def _build_checkpoint_files(model_dir: Path, version: str) -> list[Path]:
    shard_count = _MODEL_SHARDS[version]
    return [
        model_dir / f"model-{index:05d}-of-{shard_count:05d}.safetensors"
        for index in range(1, shard_count + 1)
    ]


def _quantize_text_encoder(
    root: str | Path,
    input_root: str | Path,
    *,
    version: str,
    method: str,
) -> Path:
    # Normalise the method name (may convert hyphens to underscores)
    base_method = normalize_quant_method(method)

    # Split into high/low methods for mixed precision
    high_raw, low_raw = convert_quant_name(method)   # e.g. ("sym_med", "aff_med")
    high_method = normalize_quant_method(high_raw)
    low_method = normalize_quant_method(low_raw)

    # Build output filename (use underscored version of original input)
    quant_name = replace_hyphens_with_underscores(method)
    output_dir = Path(root) / f"flux2_{version}" / "model" / "text_encoder"
    model_dir = Path(input_root).expanduser().resolve()

    if not model_dir.is_dir():
        raise FileNotFoundError(f"Text encoder directory not found: {model_dir}")

    checkpoint_files = _build_checkpoint_files(model_dir, version)

    # Get high/low target tensors from the shared builder
    target_tensors = _build_target_tensors()
    targets = {
        high_method: target_tensors["high"],
        low_method: target_tensors["low"],
    }

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
        help="Root folder where flux_2_klein_4b or flux2_9b is stored or should receive the quantized output.",
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
        help="Quantization method to apply (e.g., aff-med-max, aff-med-mini).",
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
