#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from pathlib import Path

from safetensors.torch import save_file

from utils.quant.model import quantize_model_tensors
from utils.quant.targets import build_mixed_target_config
from wan22.quant.targets import _build_wan22_denoiser_mixed_target_tensors

_MODEL_DIR = "wan22"
_MODEL_SHARDS = 3


def _build_checkpoint_files(model_dir: Path) -> list[Path]:
    return [
        model_dir / f"diffusion_pytorch_model-{index:05d}-of-{_MODEL_SHARDS:05d}.safetensors"
        for index in range(1, _MODEL_SHARDS + 1)
    ]


def _build_targets_for_method(method: str) -> dict[str, list[str]]:
    return build_mixed_target_config(
        method,
        _build_wan22_denoiser_mixed_target_tensors(),
    )


def quantize_denoiser(
    root: str | Path,
    input_root: str | Path | None = None,
    *,
    method: str,
) -> Path:
    output_dir = Path(root).expanduser().resolve() / _MODEL_DIR / "model" / "denoiser"
    if input_root is not None:
        model_dir = Path(input_root).expanduser().resolve()
    else:
        model_dir = output_dir

    if not model_dir.is_dir():
        raise FileNotFoundError(f"Denoiser directory not found: {model_dir}")

    checkpoint_files = _build_checkpoint_files(model_dir)
    missing_files = [path.name for path in checkpoint_files if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(
            "WAN22 denoiser input directory is missing expected shard files: "
            f"{', '.join(missing_files)}. "
            f"Directory checked: {model_dir}. "
            "If you meant to quantize the text encoder, use `python -m wan22.quant.text_encoder` instead."
        )
    output_path = output_dir / f"{method}_quant.safetensors"

    print(f"Applying {method} quantization for WAN22 denoiser ...")
    quantize_start = time.perf_counter()
    state_dict = quantize_model_tensors(
        files=checkpoint_files,
        targets=_build_targets_for_method(method),
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
            "component": "denoiser",
            "method": method,
        },
    )
    save_seconds = time.perf_counter() - save_start
    print(f"Saved in {save_seconds:.3f}s")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize and save the WAN22 denoiser.")
    parser.add_argument(
        "--root",
        required=True,
        help="Root folder where wan22/model/denoiser is stored or should receive the quantized output.",
    )
    parser.add_argument(
        "--input",
        help="Full denoiser model directory. Bypasses the --root default path.",
    )
    parser.add_argument(
        "--method",
        required=True,
        help="Quantization method to apply.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = quantize_denoiser(
        args.root,
        args.input,
        method=args.method,
    )
    print(f"Saved quantized denoiser to {output_path}")


if __name__ == "__main__":
    main()
