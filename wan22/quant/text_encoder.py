#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from pathlib import Path

from safetensors.torch import save_file

from utils.quant.model import quantize_model_tensors

_MODEL_DIR = "wan22"
_CHECKPOINT_NAME = "t5_umt5-xxl-enc-bf16.pth"
_NUM_BLOCKS = 24
_T5_LINEAR_WEIGHT_SUFFIXES = (
    "attn.q.weight",
    "attn.k.weight",
    "attn.v.weight",
    "attn.o.weight",
    "ffn.gate.0.weight",
    "ffn.fc1.weight",
    "ffn.fc2.weight",
)


def _build_target_tensors() -> list[str]:
    tensors: list[str] = []
    for block_idx in range(_NUM_BLOCKS):
        block_prefix = f"blocks.{block_idx}."
        for suffix in _T5_LINEAR_WEIGHT_SUFFIXES:
            tensors.append(block_prefix + suffix)
    return tensors


def quantize_text_encoder(
    root: str | Path,
    input_root: str | Path | None = None,
    *,
    method: str,
) -> Path:
    output_dir = Path(root).expanduser().resolve() / _MODEL_DIR / "model" / "text_encoder"
    if input_root is not None:
        model_dir = Path(input_root).expanduser().resolve()
    else:
        model_dir = output_dir

    if not model_dir.is_dir():
        raise FileNotFoundError(f"Text encoder directory not found: {model_dir}")

    checkpoint_path = model_dir / _CHECKPOINT_NAME
    output_path = output_dir / f"{method}_quant.safetensors"

    print(f"Applying {method} quantization for WAN22 T5 text encoder ...")
    quantize_start = time.perf_counter()
    state_dict = quantize_model_tensors(
        files=checkpoint_path,
        targets={method: _build_target_tensors()},
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
            "component": "text_encoder",
            "method": method,
        },
    )
    save_seconds = time.perf_counter() - save_start
    print(f"Saved in {save_seconds:.3f}s")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize and save the WAN22 T5 text encoder.")
    parser.add_argument(
        "--root",
        required=True,
        help="Root folder where wan22/model/text_encoder is stored or should receive the quantized output.",
    )
    parser.add_argument(
        "--input",
        help="Full text_encoder model directory. Bypasses the --root default path.",
    )
    parser.add_argument(
        "--method",
        required=True,
        help="Quantization method to apply.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = quantize_text_encoder(
        args.root,
        args.input,
        method=args.method,
    )
    print(f"Saved quantized text encoder to {output_path}")


if __name__ == "__main__":
    main()
