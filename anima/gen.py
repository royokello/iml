#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from anima.tokenizer import AnimaTokenizer
from anima.text_encoder.loader import _load_anima_text_encoder
from anima.denoiser.loader import _load_anima_denoiser
from anima.sampler import CosmosRFlowScheduler, get_sigmas, euler_ancestral_step


def _load_vae(root_dir, device):
    from anima.vae import QwenImageVAE, _load_vae_checkpoint
    vae_path = Path(root_dir) / "anima" / "model" / "vae" / "qwen_image_vae.safetensors"
    vae = QwenImageVAE()
    _load_vae_checkpoint(vae, vae_path)
    return vae.to(device=device, dtype=torch.float32)


def generate_image(
    root: str,
    prompt: str,
    negative_prompt: str | None = None,
    variant: str = "base",
    steps: int = 30,
    cfg: float = 4.0,
    seed: int | None = None,
    width: int = 1024,
    height: int = 1024,
    text_quant_method: str | None = None,
    denoiser_quant_method: str | None = None,
) -> list:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required")

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)

    root_dir = Path(root).expanduser().resolve()

    print("1. Tokenizer")
    tokenizer = AnimaTokenizer(root_dir)

    print("2. Text Encoder")
    print(f"    quantization: {text_quant_method or 'none'}")
    te_path = root_dir / "anima" / "model" / "text_encoder"
    text_encoder = _load_anima_text_encoder(str(te_path), quant_method=text_quant_method)
    text_encoder = text_encoder.to(device=device)

    print("  * encoding prompt ...")
    encoded = tokenizer.tokenize(prompt)
    qwen_ids = encoded["qwen_input_ids"].to(device)
    qwen_mask = encoded["qwen_attention_mask"].to(device)
    t5_ids = encoded["t5_input_ids"].to(device)
    t5_mask = encoded["t5_attention_mask"].to(device)

    hidden_states = text_encoder(qwen_ids, attention_mask=qwen_mask)

    context = hidden_states.to(dtype=torch.float32)
    t5xxl_ids = t5_ids
    t5xxl_weights = t5_mask.to(dtype=torch.float32)

    if negative_prompt:
        neg_encoded = tokenizer.tokenize(negative_prompt)
        neg_qwen_ids = neg_encoded["qwen_input_ids"].to(device)
        neg_qwen_mask = neg_encoded["qwen_attention_mask"].to(device)
        neg_t5_ids = neg_encoded["t5_input_ids"].to(device)
        neg_t5_mask = neg_encoded["t5_attention_mask"].to(device)
        neg_hidden = text_encoder(neg_qwen_ids, attention_mask=neg_qwen_mask)
        neg_context = neg_hidden.to(dtype=torch.float32)
        neg_t5xxl_ids = neg_t5_ids
        neg_t5xxl_weights = neg_t5_mask.to(dtype=neg_context.dtype)
    else:
        neg_context = None

    del text_encoder
    torch.cuda.empty_cache()

    print("3. Scheduler")
    scheduler = CosmosRFlowScheduler(sigma_max=80.0)
    sigmas = get_sigmas(scheduler, steps).to(device)

    print("4. Latents")
    H, W = height, width
    latent_h, latent_w = H // 8, W // 8
    latent_shape = (1, 16, 1, latent_h, latent_w)
    noise = torch.randn(latent_shape, device=device, generator=generator)
    latents = sigmas[0] * noise

    print("5. Denoiser")
    print(f"    variant: {variant}")
    print(f"    quantization: {denoiser_quant_method or 'none'}")
    denoiser_path = root_dir / "anima" / "model" / "denoiser"
    denoiser = _load_anima_denoiser(str(denoiser_path), variant=variant, quant_method=denoiser_quant_method)
    denoiser = denoiser.to(device=device)
    denoiser.eval()

    context = context.to(device=device)
    t5xxl_ids = t5xxl_ids.to(device)
    if neg_context is not None:
        neg_context = neg_context.to(device=device)

    print("6. Denoising")
    with torch.inference_mode():
        for i in range(steps):
            step_start = time.perf_counter()
            sigma_curr = sigmas[i]
            sigma_next = sigmas[i + 1]

            timesteps = sigma_curr.view(1).expand(latents.shape[0])

            denoised = denoiser(
                latents, timesteps, context,
                t5xxl_ids=t5xxl_ids, t5xxl_weights=t5xxl_weights,
            )
            if neg_context is not None:
                neg_denoised = denoiser(
                    latents, timesteps, neg_context,
                    t5xxl_ids=neg_t5xxl_ids, t5xxl_weights=neg_t5xxl_weights,
                )
                denoised = neg_denoised + cfg * (denoised - neg_denoised)

            if i == 0:
                print(f"    sigma={sigma_curr.item():.3f}")
                print(f"    input : {latents.min():.3f} {latents.mean():.3f} {latents.max():.3f} ±{latents.std():.3f}")
                print(f"    denoised: {denoised.min():.3f} {denoised.mean():.3f} {denoised.max():.3f} ±{denoised.std():.3f}")

            rf_ratio = sigma_next / sigma_curr
            latents = rf_ratio * latents + (1.0 - rf_ratio) * denoised

            step_seconds = time.perf_counter() - step_start
            print(f"    step {i + 1}/{steps} in {step_seconds:.3f}s", flush=True)

    del denoiser
    torch.cuda.empty_cache()

    print("7. VAE Decode")
    vae = _load_vae(root_dir, device)
    with torch.inference_mode():
        decoded = vae.decode(latents.to(dtype=torch.float32))
        decoded = (decoded / 2 + 0.5).clamp(0, 1)
        decoded = decoded.cpu().float()
        decoded = decoded[:, :, 0]  # T=1, drop temporal dim
        decoded = decoded.permute(0, 2, 3, 1).numpy()
        from PIL import Image
        images = [Image.fromarray((img * 255).astype("uint8")) for img in decoded]

    del vae
    torch.cuda.empty_cache()
    return images


def parse_args():
    parser = argparse.ArgumentParser(description="Anima text-to-image generation")
    parser.add_argument("--root", required=True)
    parser.add_argument("--prompt", default="anime girl, detailed, masterpiece")
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--variant", choices=("base", "aesthetic", "turbo", "preview"), default="base")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--cfg", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--text-quant", default="fp16")
    parser.add_argument("--denoiser-quant", default="sym-high")
    return parser.parse_args()


def main():
    args = parse_args()
    images = generate_image(
        root=args.root,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        variant=args.variant,
        steps=args.steps,
        cfg=args.cfg,
        seed=args.seed,
        width=args.width,
        height=args.height,
        text_quant_method=args.text_quant,
        denoiser_quant_method=args.denoiser_quant,
    )
    output_dir = Path(args.root) / "anima" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        path = output_dir / f"{args.prompt[:50].replace(' ', '_')}_{i}.png"
        img.save(str(path))
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
