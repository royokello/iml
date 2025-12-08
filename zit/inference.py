#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from diffusers import AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from zit.models import ZImageTransformer2DModel
from zit.pipeline import ZImagePipeline

DEFAULT_PROMPT = (
    "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, "
    "red floral forehead pattern. Elaborate high bun, golden phoenix headdress, "
    "red flowers, beads. Holds round folding fan. Soft-lit outdoor night background."
)


def _auto_name(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    for idx in range(1, 100):
        candidate = os.path.join(out_dir, f"{ts}-{idx:02}.png")
        if not os.path.exists(candidate):
            return candidate
    raise RuntimeError("Unable to derive unique filename for output image.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Z-Image Turbo inference.")
    parser.add_argument(
        "--model",
        required=True,
        help="Base directory with tokenizer/, text_encoder/, transformer/, vae/, scheduler/.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory where the PNG result is written.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Positive prompt text.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=19930625,
        help="Random seed.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Image height (multiple of 16 recommended).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Image width (multiple of 16 recommended).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=9,
        help="Number of sampling steps.",
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=0.0,
        help="Classifier-free guidance scale (Turbo models expect 0.0).",
    )
    parser.add_argument(
        "--text-model",
        default=None,
        help="Optional override directory for tokenizer/ and text_encoder/ (e.g. Qwen3-4B).",
    )
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()

    base_dir = args.model
    gpu_device = torch.device("cuda")

    torch.manual_seed(args.seed)
    generator = torch.Generator(device=gpu_device).manual_seed(args.seed)

    text_model_dir = args.text_model or base_dir

    if args.text_model:
        print(f"Using text model directory: {text_model_dir}")

    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(text_model_dir, "tokenizer"),
        trust_remote_code=True,
    )

    print("Loading text encoder (Qwen3) ...")
    text_encoder = AutoModelForCausalLM.from_pretrained(
        os.path.join(text_model_dir, "text_encoder"),
        dtype=torch.float16,
        trust_remote_code=True,
    )

    print("Loading scheduler ...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        base_dir,
        subfolder="scheduler",
    )

    print("Loading transformer (DiT) ...")
    transformer = ZImageTransformer2DModel.from_pretrained(
        base_dir,
        subfolder="transformer",
        dtype=torch.float16,
    )

    print("Loading VAE ...")
    vae = AutoencoderKL.from_pretrained(
        base_dir,
        subfolder="vae",
        dtype=torch.float16,
    )

    print("Building pipeline ...")
    pipe = ZImagePipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
    )
    pipe.to(gpu_device, dtype=torch.float16)

    # Optional optimizations (same as your previous script)
    # pipe.transformer.compile()
    pipe.enable_model_cpu_offload()

    print("Generating image ...")
    result = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        generator=generator,
    )
    image = result.images[0]

    out_path = _auto_name(args.output)
    image.save(out_path)
    print(f"[done] Saved {out_path}")


if __name__ == "__main__":
    main()
