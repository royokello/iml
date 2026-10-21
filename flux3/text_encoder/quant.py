#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

from safetensors.torch import save_file

from flux3.config import resolve_model_root
from flux3.text_encoder.target import (
    _FLUX3_TEXT_ENCODER_EXCLUSION_PREFIXES,
    _FLUX3_TEXT_ENCODER_INCLUSION_PREFIX,
    _build_flux3_text_encoder_target_tensors,
)
from utils.quant.model import quantize_model_tensors
from utils.quant.targets import build_mixed_target_config
from utils.text import replace_hyphens_with_underscores

_SHARD_PATTERN = "model-*.safetensors"


def _quantize_text_encoder(
    root: str | Path,
    input_dir: str | Path | None,
    *,
    method: str,
) -> Path:
    output_dir = resolve_model_root(root) / "text_encoder"
    model_dir = Path(input_dir).expanduser() if input_dir is not None else output_dir
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Text encoder directory not found: {model_dir}")

    checkpoint_files = sorted(model_dir.glob(_SHARD_PATTERN))
    if not checkpoint_files:
        raise FileNotFoundError(f"No files match {_SHARD_PATTERN} in {model_dir}")

    print(
        "Quantize text encoder args:\n"
        f"  root={root}\n"
        f"  input={model_dir}\n"
        f"  method={method}"
    )
    print("Checkpoint")
    for checkpoint in checkpoint_files:
        print(f" * {checkpoint}")

    targets = build_mixed_target_config(method, _build_flux3_text_encoder_target_tensors())

    print("Targets")
    for target_method, target_names in targets.items():
        print(f" * {target_method}: {len(target_names)} tensors")

    quant_name = replace_hyphens_with_underscores(method)
    output_path = output_dir / f"{quant_name}_quant.safetensors"

    print(f"Applying {method} mixed-precision quantization for Flux3 text encoder ...")
    quantize_start = time.perf_counter()
    # The inclusion prefix drops the visual tower. The exclusion prefixes drop
    # layers 33 to 35 and the final norm, which no output uses.
    state_dict = quantize_model_tensors(
        files=checkpoint_files,
        targets=targets,
        inclusion_prefix=_FLUX3_TEXT_ENCODER_INCLUSION_PREFIX,
        exclusion_prefix=_FLUX3_TEXT_ENCODER_EXCLUSION_PREFIXES,
    )
    print(f"Quantized in {time.perf_counter() - quantize_start:.3f}s")

    print(f"Saving {output_path} ...")
    save_start = time.perf_counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        state_dict,
        str(output_path),
        metadata={
            "model": "flux3",
            "component": "text_encoder",
            "method": method,
            "application": "IML",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(f"Saved in {time.perf_counter() - save_start:.3f}s")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize and save the Flux3 text encoder.")
    parser.add_argument(
        "--root",
        required=True,
        help="Root folder. The output goes to <root>/flux3/model/text_encoder/. The folder is created if missing.",
    )
    parser.add_argument(
        "--method",
        required=True,
        help="Quantization method to apply (for example sym-high-mini).",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Source text_encoder folder. Default: <root>/flux3/model/text_encoder.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = _quantize_text_encoder(args.root, args.input, method=args.method)
    print(f"Saved quantized text encoder to {output_path}")


if __name__ == "__main__":
    main()
