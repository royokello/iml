#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from pathlib import Path

from safetensors.torch import save_file

from utils.quant.model import quantize_model_tensors
from utils.quant.targets import build_mixed_target_config
from utils.text import replace_hyphens_with_underscores

from krea_2.text_encoder.target import _build_krea2_text_encoder_targets


def _quantize_text_encoder(
    root: str | Path,
    input_dirpath: str | Path,
    method: str,
) -> Path:
    input_dir = Path(input_dirpath)
    model_file = input_dir / "model.safetensors"
    if not model_file.is_file():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    quant_name = replace_hyphens_with_underscores(method)
    output_dir = Path(root) / "krea_2" / "model" / "text_encoder"
    output_path = output_dir / f"{quant_name}_quant.safetensors"

    target_tensors = _build_krea2_text_encoder_targets()
    targets = build_mixed_target_config(method, target_tensors)

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
    state_dict = quantize_model_tensors(files=model_file, targets=targets)
    quant_seconds = time.perf_counter() - quant_start
    print(f"Quantized in {quant_seconds:.3f}s")

    print(f"Saving {output_path} ...")
    save_start = time.perf_counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        state_dict,
        str(output_path),
        metadata={
            "component": "text_encoder",
            "model": "krea_2",
            "method": quant_name,
        },
    )
    save_seconds = time.perf_counter() - save_start
    print(f"Saved in {save_seconds:.3f}s")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize the Krea-2 text encoder.")
    parser.add_argument("--root", required=True, help="Root folder to receive the quantized output.")
    parser.add_argument("--input", required=True, help="Source directory containing model.safetensors.")
    parser.add_argument("--method", default="sym-med-nano", help="Quantization method (default: sym-med-nano).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = _quantize_text_encoder(args.root, args.input, method=args.method)
    print(f"Saved quantized text encoder to {output_path}")


if __name__ == "__main__":
    main()
