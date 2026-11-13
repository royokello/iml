#!/usr/bin/env python
"""Quantize the SDXL UNet and save the result inside the model folder.

The default method is `sym-high`. Any other name from `utils.quant` is passed
through as-is.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from sdxl.config import DEFAULT_MODEL_NAME, resolve_model_root
from sdxl.denoiser.targets import build_sdxl_denoiser_target_tensors
from utils.quant.model import quantize_model_tensors
from utils.quant.targets import build_mixed_target_config

DEFAULT_METHOD = "sym-high"
SINGLE_CHECKPOINT = "diffusion_pytorch_model.safetensors"


def resolve_unet_dir(root: str | Path, name: str = DEFAULT_MODEL_NAME) -> Path:
    """Return `<root>/sdxl/<name>/unet` and check that it exists.

    Args:
        root: The data root.
        name: The model folder name. Defaults to `base`.

    Returns:
        The `unet` folder.

    Raises:
        FileNotFoundError: If the model folder or its `unet` folder is missing.
    """
    model_root = resolve_model_root(root, name)
    if not model_root.is_dir():
        raise FileNotFoundError(f"Model folder not found: {model_root}")
    unet_dir = model_root / "unet"
    if not unet_dir.is_dir():
        raise FileNotFoundError(f"UNet folder not found: {unet_dir}")
    return unet_dir


def resolve_source_dir(input_dir: str | Path) -> Path:
    """Return the `unet` folder of the converted model and check its checkpoint.

    Args:
        input_dir: The `unet` folder of the converted model, for example
            `/models/illustrious-xl/unet`.

    Returns:
        The `unet` folder.

    Raises:
        FileNotFoundError: If the folder or the checkpoint is missing.
    """
    unet_dir = Path(input_dir).expanduser().resolve()
    if not unet_dir.is_dir():
        raise FileNotFoundError(f"UNet folder not found: {unet_dir}")

    checkpoint = unet_dir / SINGLE_CHECKPOINT
    if not checkpoint.is_file():
        raise FileNotFoundError(f"UNet checkpoint not found: {checkpoint}")
    return unet_dir


def linear_weight_names(checkpoint: Path) -> list[str]:
    """Collect the rank-2 `.weight` names from the checkpoint header.

    Args:
        checkpoint: The checkpoint file.

    Returns:
        The weight names, in file order.
    """
    names: list[str] = []
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if not key.endswith(".weight"):
                continue
            if len(handle.get_slice(key).get_shape()) == 2:
                names.append(key)
    return names


def quantize_denoiser(
    root: str | Path,
    input_dir: str | Path,
    *,
    name: str = DEFAULT_MODEL_NAME,
    method: str = DEFAULT_METHOD,
) -> Path:
    """Quantize the UNet and write `<root>/sdxl/<name>/unet/<method>_quant.safetensors`.

    Args:
        root: The data root.
        input_dir: The `unet` folder of the converted model. The fp16
            checkpoint is read from `<input_dir>/diffusion_pytorch_model.safetensors`.
        name: The model folder name. Defaults to `base`.
        method: A quantization method name. The default is `sym-high`.

    Returns:
        The path of the quantized checkpoint.

    Raises:
        FileNotFoundError: If the model folder, the converted folder, or the
            checkpoint is missing.
        RuntimeError: If CUDA is not available, because the quantizers run on
            the GPU.
    """
    unet_dir = resolve_unet_dir(root, name)
    model_root = unet_dir.parent
    source_dir = resolve_source_dir(input_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("Quantization needs a CUDA device.")

    checkpoint = source_dir / SINGLE_CHECKPOINT
    names = linear_weight_names(checkpoint)
    target_tensors = build_sdxl_denoiser_target_tensors(names)
    targets = build_mixed_target_config(method, target_tensors)

    print(
        "Quantize SDXL denoiser:\n"
        f"  root={model_root.parent.parent}\n"
        f"  model={model_root}\n"
        f"  input={source_dir}\n"
        f"  method={method}",
        flush=True,
    )

    print(f"Targets ({len(names)} linear weights)", flush=True)
    for target_method, target_names in targets.items():
        print(f" * {target_method}: {len(target_names)} tensors", flush=True)

    print(f"Applying {method} quantization ...", flush=True)
    start = time.perf_counter()
    state_dict = quantize_model_tensors(files=checkpoint, targets=targets)
    print(f"Quantized in {time.perf_counter() - start:.3f}s", flush=True)

    output_path = unet_dir / f"{method.replace('-', '_')}_quant.safetensors"
    print(f"Saving {output_path} ...", flush=True)
    start = time.perf_counter()
    save_file(
        state_dict,
        str(output_path),
        metadata={
            "model": "sdxl",
            "component": "denoiser",
            "method": method,
            "application": "IML",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(f"Saved in {time.perf_counter() - start:.3f}s", flush=True)
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Quantize and save the SDXL denoiser.")
    parser.add_argument(
        "--root",
        required=True,
        help="Data root. The result goes to <root>/sdxl/<name>/unet/<method>_quant.safetensors.",
    )
    parser.add_argument(
        "--name",
        default=DEFAULT_MODEL_NAME,
        help=f"Model folder name. Default: {DEFAULT_MODEL_NAME}.",
    )
    parser.add_argument(
        "--method",
        default=DEFAULT_METHOD,
        help=f"Quantization method. Default: {DEFAULT_METHOD}.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="unet folder of the converted model, holding diffusion_pytorch_model.safetensors.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the quantizer from the command line."""
    args = parse_args()
    output_path = quantize_denoiser(
        args.root,
        args.input,
        name=args.name,
        method=args.method,
    )
    print(f"Saved quantized denoiser to {output_path}")


if __name__ == "__main__":
    main()
