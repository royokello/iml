#!/usr/bin/env python
"""
Quantized SDXL inference helper.

Loads CLIP text encoders and the VAE from an SDXL base repo (HuggingFace ID or
local path) but runs denoising with the custom INT8 `SDXLUNet`.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Tuple

import torch
from PIL import Image
from diffusers import AutoencoderKL
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
)
from safetensors.torch import load_file as load_safetensors
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from sdxl.unet import SDXLUNet


DEFAULT_PROMPT = (
    "A propaganda poster depicting a cat dressed as french emperor napoleon holding a piece of cheese."
)
DEFAULT_BASE = "stabilityai/stable-diffusion-xl-base-1.0"
SD_SCALE = 0.18215


def _auto_name(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    for idx in range(1, 100):
        candidate = os.path.join(out_dir, f"{ts}-{idx:02}.png")
        if not os.path.exists(candidate):
            return candidate
    raise RuntimeError("Unable to derive unique filename for output image.")


def _normalize_sampler(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def _build_scheduler(base_repo: str, sampler: str, schedule: str):
    sampler_name = _normalize_sampler(sampler)
    schedule_name = schedule.strip().lower()

    if sampler_name in {"euler_a", "euler_ancestral"}:
        cls = EulerAncestralDiscreteScheduler
    elif sampler_name in {"euler", "euler_discrete"}:
        cls = EulerDiscreteScheduler
    elif sampler_name == "heun":
        cls = HeunDiscreteScheduler
    elif sampler_name == "ddim":
        cls = DDIMScheduler
    elif sampler_name in {"dpmpp_2m", "dpmpp2m"}:
        cls = DPMSolverMultistepScheduler
    else:
        raise SystemExit(f"Unsupported sampler '{sampler}'.")

    scheduler = cls.from_pretrained(base_repo, subfolder="scheduler")
    use_karras = schedule_name.startswith("karras")
    if hasattr(scheduler, "use_karras_sigmas"):
        scheduler.use_karras_sigmas = use_karras
    return scheduler


def _tokenize_pair(tokenizer: CLIPTokenizer, prompt: str, negative: str) -> dict:
    texts = [negative, prompt]
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )


def _encode_with_model(
    model_cls,
    base_repo: str,
    subfolder: str,
    tokens: dict,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    model = model_cls.from_pretrained(base_repo, subfolder=subfolder, torch_dtype=dtype)
    model = model.to(device=device)
    model.eval()

    inputs = {k: v.to(device=device) for k, v in tokens.items()}
    outputs = model(**inputs)

    hidden = outputs.last_hidden_state.detach().to("cpu", dtype=dtype)
    pooled = None
    if hasattr(outputs, "text_embeds") and getattr(outputs, "text_embeds") is not None:
        pooled = outputs.text_embeds.detach().to("cpu", dtype=dtype)
    elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        pooled = outputs.pooler_output.detach().to("cpu", dtype=dtype)

    del model
    torch.cuda.empty_cache()
    return hidden, pooled


def _encode_prompts(
    base_repo: str,
    prompt: str,
    negative: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    tokenizer = CLIPTokenizer.from_pretrained(base_repo, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(base_repo, subfolder="tokenizer_2")

    tokens = _tokenize_pair(tokenizer, prompt, negative)
    tokens_2 = _tokenize_pair(tokenizer_2, prompt, negative)

    hidden_1, _ = _encode_with_model(
        CLIPTextModel,
        base_repo,
        "text_encoder",
        tokens,
        device,
        dtype,
    )
    hidden_2, _ = _encode_with_model(
        CLIPTextModelWithProjection,
        base_repo,
        "text_encoder_2",
        tokens_2,
        device,
        dtype,
    )

    prompt_embeds = torch.cat([hidden_2, hidden_1], dim=-1)
    return prompt_embeds


def _prepare_latents(
    height: int,
    width: int,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
    scheduler,
) -> torch.Tensor:
    latent_h = height // 8
    latent_w = width // 8
    latents = torch.randn((1, 4, latent_h, latent_w), generator=generator, dtype=dtype, device=device)
    init_sigma = getattr(scheduler, "init_noise_sigma", 1.0)
    latents = latents * init_sigma
    return latents


def _timestep_batch(timestep, batch_size: int, device: torch.device) -> torch.Tensor:
    if isinstance(timestep, torch.Tensor):
        t = timestep.to(device=device, dtype=torch.float32)
        if t.dim() == 0:
            t = t[None]
    else:
        t = torch.tensor([t], device=device, dtype=torch.float32)
    if t.numel() == 1:
        t = t.expand(batch_size)
    return t


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="INT8 SDXL inference using custom UNet.")
    parser.add_argument("--model", required=True, help="Path to quantized SDXL UNet .safetensors checkpoint.")
    parser.add_argument("--output", required=True, help="Directory where the PNG result is written.")
    parser.add_argument(
        "--base",
        default=DEFAULT_BASE,
        help="Diffusers SDXL base repo or local path for tokenizer/text encoders/VAE/scheduler.",
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
        default="",
        help="Negative (unconditional) prompt.",
    )
    parser.add_argument("--seed", type=int, default=19930625, help="Random seed.")
    parser.add_argument("--height", type=int, default=1024, help="Image height (multiple of 8).")
    parser.add_argument("--width", type=int, default=1024, help="Image width (multiple of 8).")
    parser.add_argument("--steps", type=int, default=20, help="Number of sampling steps.")
    parser.add_argument("--sampler", default="euler_a", help="Sampler name (euler_a, euler, heun, ddim, dpmpp_2m).")
    parser.add_argument("--scheduler", default="karras", help="Scheduler noise spacing (karras, linear).")
    parser.add_argument("--cfg", type=float, default=5.0, help="Classifier-free guidance scale.")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.model):
        raise SystemExit(f"Checkpoint not found: {args.model}")
    if args.steps < 1:
        raise SystemExit("--steps must be >= 1")

    width = max(64, (args.width // 8) * 8)
    height = max(64, (args.height // 8) * 8)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA device is required for INT8 SDXL inference.")
    device = torch.device("cuda")
    dtype = torch.float16

    torch.manual_seed(args.seed)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    print("[text] Encoding prompts on GPU …")
    prompt_embeds = _encode_prompts(args.base, args.prompt, args.negative, device, dtype)
    torch.cuda.empty_cache()

    scheduler = _build_scheduler(args.base, args.sampler, args.scheduler)
    scheduler.set_timesteps(args.steps, device=device)

    print("[unet] Loading quantized UNet …")
    state_dict = load_safetensors(args.model, device="cpu")
    unet = SDXLUNet(model=state_dict)
    unet = unet.to(device=device).eval()

    latents = _prepare_latents(height, width, generator, device, dtype, scheduler)
    prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)

    print(f"[sample] Running {len(scheduler.timesteps)} denoising steps @ CFG {args.cfg}")
    for step_idx, t in enumerate(scheduler.timesteps, 1):
        latent_model_input = latents.repeat(2, 1, 1, 1)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        timestep = _timestep_batch(t, latent_model_input.shape[0], device)
        noise = unet(
            sample=latent_model_input,
            timesteps=timestep,
            encoder_hidden_states=prompt_embeds,
        )
        noise_uncond, noise_text = noise.chunk(2)
        guided = noise_uncond + args.cfg * (noise_text - noise_uncond)

        latents = scheduler.step(guided, t, latents).prev_sample
        latents = latents.to(dtype=dtype)
        print(f"  step {step_idx}/{len(scheduler.timesteps)} done")

    del unet, state_dict
    torch.cuda.empty_cache()

    print("[vae] Loading decoder …")
    vae = AutoencoderKL.from_pretrained(args.base, subfolder="vae", torch_dtype=dtype)
    vae = vae.to(device=device).eval()

    print("[decode] Converting latents to image …")
    latents = latents / SD_SCALE
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image[0].permute(1, 2, 0).detach().cpu().numpy()
    image = (image * 255).round().astype("uint8")
    pil_image = Image.fromarray(image)

    out_path = _auto_name(args.output)
    pil_image.save(out_path)
    print(f"[done] Saved → {out_path}")


if __name__ == "__main__":
    main()
