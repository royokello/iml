#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from pathlib import Path

from safetensors.torch import save_file

from utils.quant.model import quantize_model_tensors
from utils.quant.targets import build_mixed_target_config
from utils.text import replace_hyphens_with_underscores

from qwen_2.config import resolve_model_root
from qwen_2.text_encoder.target import (
    TEXT_ENCODER_EXCLUSION_PREFIX,
    TEXT_ENCODER_INCLUSION_PREFIX,
    _build_qwen2_text_encoder_targets,
)


def _quantize_text_encoder(
    root: str | Path,
    input_dirpath: str | Path,
    method: str,
) -> Path:
    """Quantize the Qwen3-VL text encoder.

    The vision tower stays BF16 and the LM head is dropped.

    Args:
        root: The root folder. The output goes to `<root>/qwen_2/model/text_encoder`.
        input_dirpath: The source folder that holds `model.safetensors`.
        method: The quantization method.

    Returns:
        The path of the quantized file.

    Raises:
        FileNotFoundError: If the source model file is missing.
    """
    input_dir = Path(input_dirpath)
    model_file = input_dir / "model.safetensors"
    if not model_file.is_file():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    quant_name = replace_hyphens_with_underscores(method)
    output_dir = resolve_model_root(root) / "text_encoder"
    output_path = output_dir / f"{quant_name}_quant.safetensors"

    target_tensors = _build_qwen2_text_encoder_targets()
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
    state_dict = quantize_model_tensors(
        files=model_file,
        targets=targets,
        inclusion_prefix=TEXT_ENCODER_INCLUSION_PREFIX,
        exclusion_prefix=TEXT_ENCODER_EXCLUSION_PREFIX,
    )
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
            "model": "qwen_2",
            "method": quant_name,
            "vision": "bf16",
        },
    )
    save_seconds = time.perf_counter() - save_start
    print(f"Saved in {save_seconds:.3f}s")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize the Qwen Image 2.1 text encoder.")
    parser.add_argument("--root", required=True, help="Root folder to receive the quantized output.")
    parser.add_argument("--input", required=True, help="Source directory containing model.safetensors.")
    parser.add_argument("--method", default="q8-mini", help="Quantization method (default: q8-mini).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = _quantize_text_encoder(args.root, args.input, method=args.method)
    print(f"Saved quantized text encoder to {output_path}")


if __name__ == "__main__":
    main()
