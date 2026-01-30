#!/usr/bin/env python
"""
Quantized SDXL inference helper.

Uses `--base` for tokenizer/config folders while loading all weights from a
single combined `.safetensors` checkpoint, then denoises with the INT8
`SDXLUNet`.
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from typing import Iterable

import torch
from PIL import Image
from diffusers import AutoencoderKL
from diffusers.schedulers import EulerAncestralDiscreteScheduler
from safetensors.torch import load_file as load_safetensors
from transformers import (
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)
from .pipeline import StableDiffusionXLPipeline
from .models._unet import SDXLUNet

DEFAULT_PROMPT = "A propaganda poster depicting a cat dressed as french emperor napoleon holding a piece of cheese."
DEFAULT_NEGATIVE = ""
DEFAULT_SEED = 19930625
DEFAULT_HEIGHT = 512
DEFAULT_WIDTH = 512
DEFAULT_STEPS = 20
DEFAULT_CFG = 5.0

# DEFAULT_PROMPT = "a close-up of a fire spitting dragon, cinematic shot."


def _auto_name(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    for idx in range(1, 100):
        candidate = os.path.join(out_dir, f"{ts}-{idx:02}.png")
        if not os.path.exists(candidate):
            return candidate
    raise RuntimeError("Unable to derive unique filename for output image.")


def _coerce_dimension(value: int) -> int:
    return max(64, (value // 8) * 8)


def _ensure_unet_config(unet_dir: str, base_dir: str) -> None:
    base_unet_dir = os.path.join(base_dir, "unet")
    for filename in ("config.json",):
        target = os.path.join(unet_dir, filename)
        if os.path.exists(target):
            continue
        source = os.path.join(base_unet_dir, filename)
        if os.path.isfile(source):
            os.makedirs(unet_dir, exist_ok=True)
            shutil.copyfile(source, target)
        else:
            raise SystemExit(f"Missing UNet config file: {source}")


def _resolve_unet_checkpoint(unet_dir: str) -> str:
    path = os.path.abspath(unet_dir)
    if os.path.isfile(path):
        if path.endswith(".safetensors"):
            return path
        raise SystemExit(f"Expected a .safetensors file for UNet weights: {path}")

    if not os.path.isdir(path):
        raise SystemExit(f"UNet directory not found: {unet_dir}")

    preferred = os.path.join(path, "diffusion_pytorch_model.safetensors")
    if os.path.isfile(preferred):
        return preferred

    safetensors = sorted(
        filename for filename in os.listdir(path) if filename.endswith(".safetensors")
    )
    if safetensors:
        return os.path.join(path, safetensors[0])

    raise SystemExit(
        f"No .safetensors file found in UNet directory: {path}. "
        "Provide --unet pointing to a folder that contains quantized UNet weights."
    )


def _load_quantized_unet(
    base_dir: str,
    unet_dir: str,
    device: torch.device,
) -> SDXLUNet:
    checkpoint_path = _resolve_unet_checkpoint(unet_dir)
    print(f"Loading quantized UNet checkpoint '{checkpoint_path}' ...")
    model = load_safetensors(checkpoint_path, device="cpu")
    unet = SDXLUNet(model=model)
    return unet.to(device=device, dtype=torch.float16).eval()


def _build_scheduler(base_dir: str):
    return EulerAncestralDiscreteScheduler.from_pretrained(base_dir, subfolder="scheduler")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SDXL inference.")
    parser.add_argument("--output", required=True, help="Directory where the PNG result is written.")
    parser.add_argument(
        "--base",
        required=True,
        help="Directory containing tokenizer/, text_encoder/, vae/, scheduler/ configs.",
    )
    parser.add_argument(
        "--unet",
        help="Optional path to an alternate UNet folder (defaults to --base/unet).",
    )
    parser.add_argument(
        "--prompt",
        "--pos-prompt",
        dest="prompt",
        default=DEFAULT_PROMPT,
        help="Positive prompt text.",
    )
    parser.add_argument(
        "--negative",
        "--neg-prompt",
        dest="negative",
        default=DEFAULT_NEGATIVE,
        help="Negative (unconditional) prompt.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="Image height (multiple of 8).")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Image width (multiple of 8).")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Number of sampling steps.")
    parser.add_argument("--cfg", type=float, default=DEFAULT_CFG, help="Classifier-free guidance scale.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging for tensor stats.",
    )
    return parser.parse_args()


@torch.inference_mode()
def generate_images(
    *,
    output_dir: str,
    base_dir: str,
    unet_dir: str | None = None,
    prompt: str = DEFAULT_PROMPT,
    negative: str = DEFAULT_NEGATIVE,
    seed: int = DEFAULT_SEED,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    steps: int = DEFAULT_STEPS,
    cfg: float = DEFAULT_CFG,
    enable_cpu_offload: bool = False,
) -> list[str]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for SDXL inference.")

    base_dir = os.path.abspath(base_dir)
    unet_dir = os.path.abspath(unet_dir) if unet_dir else os.path.join(base_dir, "unet")

    width = _coerce_dimension(width)
    height = _coerce_dimension(height)

    gpu_device = torch.device("cuda")
    cpu_device = torch.device("cpu")

    torch.manual_seed(seed)
    generator = torch.Generator(device=gpu_device).manual_seed(seed)

    print("Loading tokenizer 1 ...", flush=True)
    tokenizer_path = os.path.join(base_dir, "tokenizer")
    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)

    print("Loading tokenizer 2 ...", flush=True)
    tokenizer_2_path = os.path.join(base_dir, "tokenizer_2")
    tokenizer_2 = CLIPTokenizer.from_pretrained(tokenizer_2_path)

    print("Loading text encoder 1 ...", flush=True)
    text_encoder_path = os.path.join(base_dir, "text_encoder")
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path=text_encoder_path,
        dtype=torch.float16,
        local_files_only=True,
    ).to(device=gpu_device, dtype=torch.float16).eval()

    print("Loading text encoder 2 ...", flush=True)
    text_encoder_2_path = os.path.join(base_dir, "text_encoder_2")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        pretrained_model_name_or_path=text_encoder_2_path,
        dtype=torch.float16,
        local_files_only=True,
    ).to(device=gpu_device, dtype=torch.float16).eval()

    print("Loading scheduler ...", flush=True)
    scheduler_obj = _build_scheduler(base_dir)
    scheduler_obj.set_timesteps(steps, device=gpu_device)

    print("Loading unet ...", flush=True)
    _ensure_unet_config(unet_dir, base_dir)
    unet = _load_quantized_unet(
        base_dir=base_dir,
        unet_dir=unet_dir,
        device=cpu_device,
    )

    print("Loading vae ...", flush=True)
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path=base_dir,
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    vae.to(device=cpu_device, dtype=torch.float32).eval()

    print("Loading pipeline ...", flush=True)
    sdxl_pipe = StableDiffusionXLPipeline(
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        unet=unet,
        scheduler=scheduler_obj,
    )

    if enable_cpu_offload:
        print("Enabling CPU offload to reduce VRAM requirements ...", flush=True)
        sdxl_pipe.enable_model_cpu_offload()

    print("generating image ...", flush=True)
    output = sdxl_pipe.generate(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=cfg,
        negative_prompt=negative,
        generator=generator,
        gpu_device=gpu_device,
    )

    if isinstance(output, Image.Image):
        images = (output,)
    elif isinstance(output, Iterable):
        images = tuple(output)
        if not all(isinstance(img, Image.Image) for img in images):
            raise TypeError("Pipeline returned iterable with non-image entries.")
    else:
        raise TypeError(f"Unexpected output from pipeline: {type(output).__name__}")

    saved_paths = []
    for image in images:
        out_path = _auto_name(output_dir)
        image.save(out_path)
        print(f"[done] Saved '{out_path}'", flush=True)
        saved_paths.append(out_path)
    return saved_paths


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    generate_images(
        output_dir=args.output,
        base_dir=args.base,
        unet_dir=args.unet,
        prompt=args.prompt,
        negative=args.negative,
        seed=args.seed,
        height=args.height,
        width=args.width,
        steps=args.steps,
        cfg=args.cfg,
    )


if __name__ == "__main__":
    main()
