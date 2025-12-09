#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an SDXL .safetensors checkpoint to a Diffusers UNet-only repo."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the original SDXL checkpoint (.safetensors).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Models repo directory; a new *_unet folder will be created inside.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    in_path = Path(args.input).resolve()
    if not in_path.is_file():
        raise SystemExit(f"Input checkpoint not found: {in_path}")

    out_root = Path(args.output).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    out_unet_dir = out_root / f"{in_path.stem}_unet"
    out_unet_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load SDXL pipeline in fp32 on CPU (Diffusers does the name mapping)
    pipe = StableDiffusionXLPipeline.from_single_file(
        str(in_path),
        torch_dtype=torch.float16,
        use_safetensors=True,
    )

    # 2. Grab UNet and free the rest ASAP
    unet = pipe.unet
    del pipe
    torch.cuda.empty_cache()

    # 3. Optionally cast UNet to fp16 before saving to cut size
    unet.to(torch.float16)

    # 4. Save only UNet as a Diffusers repo
    unet.save_pretrained(str(out_unet_dir), safe_serialization=True)
    print(f"[convert] saved UNet to {out_unet_dir}")


if __name__ == "__main__":
    main()
