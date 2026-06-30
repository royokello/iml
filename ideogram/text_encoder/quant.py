#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from pathlib import Path

from safetensors.torch import save_file

from utils.quant.targets import build_mixed_target_config
from utils.text import replace_hyphens_with_underscores

from ideogram.quant_utils import quantize_f8_model_tensors
from ideogram.text_encoder.target import _build_ideogram_text_encoder_targets


def _quantize_text_encoder(
    root: str | Path,
    input_dirpath: str | Path,
    method: str,
) -> Path:
    input_dir = Path(input_dirpath)
    model_file = input_dir / "model.safetensors"
    if not model_file.is_file():
        raise FileNotFoundError(f"F8 model file not found: {model_file}")

    quant_name = replace_hyphens_with_underscores(method)
    output_dir = Path(root) / "ideogram" / "text_encoder"
    output_path = output_dir / f"{quant_name}_quant.safetensors"

    target_tensors = _build_ideogram_text_encoder_targets()
    targets = build_mixed_target_config(method, target_tensors)
    targets.pop("fp16", None)

    print(
        "Quantize text encoder args:\n"
        f"  root={root}\n"
        f"  input={input_dirpath}\n"
        f"  method={method}"
    )
    print("Targets")
    for target_method, target_names in targets.items():
        print(f"  {target_method}: {len(target_names)} tensors")

    print(f"Quantizing {model_file} with {method} ...")
    quant_start = time.perf_counter()
    state_dict = quantize_f8_model_tensors(files=model_file, targets=targets, exclude_prefix="visual.")
    quant_seconds = time.perf_counter() - quant_start
    print(f"Quantized in {quant_seconds:.3f}s")

    print(f"Saving {output_path} ...")
    save_start = time.perf_counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lm_prefix = "language_model."
    renamed = {
        (key.removeprefix(lm_prefix) if key.startswith(lm_prefix) else key): tensor
        for key, tensor in state_dict.items()
    }

    save_file(
        renamed,
        str(output_path),
        metadata={
            "component": "text_encoder",
            "model": "ideogram",
            "method": quant_name,
        },
    )
    save_seconds = time.perf_counter() - save_start
    print(f"Saved in {save_seconds:.3f}s")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dequantize F8 and quantize the Ideogram 4 text encoder."
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Root folder to receive the quantized output.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Source F8 text encoder directory (containing model.safetensors).",
    )
    parser.add_argument(
        "--method",
        required=True,
        help="Quantization method to apply (e.g. sym-high-nano, aff-med-mini).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = _quantize_text_encoder(
        args.root,
        args.input,
        method=args.method,
    )
    print(f"Saved quantized text encoder to {output_path}")


if __name__ == "__main__":
    main()
