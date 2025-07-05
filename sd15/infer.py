#!/usr/bin/env python
"""
Stable Diffusion v1.5 Inference

```
py -m sd15.infer --model "/models/sd15" --output "/pictures" --prompt "analogue photo, ultrawide shot of a waterfall" [--lora PATH] [--lora-strength N]
```
"""

from __future__ import annotations

import argparse
import os
import random
import time
from typing import Tuple

import torch
from PIL import Image
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel
from peft import LoraConfig, get_peft_model, TaskType
from safetensors.torch import load_file


def _auto_name(out_dir: str) -> str:
    """Return a non-clashing PNG path in *out_dir* using a timestamp."""
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    for idx in range(1, 100):
        fn = f"{ts}-{idx:02}.png"
        fp = os.path.join(out_dir, fn)
        if not os.path.exists(fp):
            return fp
    raise RuntimeError("Too many files created in one second; try again.")


def apply_lora(unet, lora_path: str, scale: float = 1.0):
    """
    Wrap the UNet with the exact LoRA adapter you trained,
    load its weights, and apply strength scaling.
    If scale ≤ 0, returns the base UNet unchanged.
    """
    # 1) Skip adapter entirely if non-positive scale
    if scale <= 0:
        return unet

    # 2) Reconstruct the same LoRA config used at train time,
    #    matching FEATURE_EXTRACTION (how you trained)
    cfg = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    unet_lora = get_peft_model(unet, cfg)

    # 3) Load your .safetensors file
    sd = load_file(lora_path, device="cpu")
    result = unet_lora.load_state_dict(sd, strict=False)

    # 4) Fail if any saved LoRA weights didn't match an injected adapter
    if result.unexpected_keys:
        raise RuntimeError(
            "LoRA checkpoint didn’t load cleanly:\n"
            f"  unexpected adapter keys: {len(result.unexpected_keys)}"
        )

    # 5) Apply the official PEFT scaling
    unet_lora.set_adapter_scale(scale)
    return unet_lora

@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser(description="Stable Diffusion v1.5 off-load inference")
    ap.add_argument("--model", required=True, help="Folder containing tokenizer/, text_encoder/, unet/, vae/, scheduler/")
    ap.add_argument("--prompt", required=True, help="Text prompt")
    ap.add_argument("--output", required=True, help="Folder for PNGs")
    ap.add_argument("--steps", type=int, default=24, help="DDIM steps ≥1")
    ap.add_argument("--size", default="512,512", help="WIDTH,HEIGHT (multiples of 8)")
    ap.add_argument("--scale", type=float, default=7.5, help="Classifier-free guidance scale")
    ap.add_argument("--seed", type=int, default=93, help="Seed for RNG")
    ap.add_argument("--lora", type=str, default=None, help="Path to a LoRA weight file (*.safetensors)")
    ap.add_argument("--lora-strength", type=float, default=1.0, help="LoRA global strength (scale)")
    ap.add_argument("--negative", type=str, default=(
        "(worst quality:2), low quality, lowres, blurry, JPEG artifacts, "
        "watermark, text, logo, duplicate, cropped, out of frame, bad anatomy, "
        "deformed, mutated, extra limbs, missing limbs, extra fingers, bad hands, "
        "bad feet, long neck, wrong proportions, ugly, grainy, oversaturated, "
        "underexposed, overexposed, nsfw, cartoon, sketch, 3d render, monochrome"
    ), help="Negative prompt (unconditional text)")

    args = ap.parse_args()

    # parse size
    if "," in args.size:
        W, H = map(int, args.size.split(","))
    else:
        W = H = int(args.size)
    W -= W % 8
    H -= H % 8

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        print(f"Seeded RNG with {args.seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Off-load script requires CUDA GPU")

    # ──────────────────────────────────────────
    # 1) Tokenizer (CPU) + text-encoder (GPU)
    # ──────────────────────────────────────────
    print("[1/3] Loading tokenizer + CLIP text-encoder …")
    tok = CLIPTokenizer.from_pretrained(os.path.join(args.model, "tokenizer"))
    txtE = (
        CLIPTextModel
        .from_pretrained(os.path.join(args.model, "text_encoder"), torch_dtype=torch.float16)
        .to(device)
        .eval()
    )

    print("Encoding prompt …")
    neg = args.negative or ""
    ids = tok([neg, args.prompt], return_tensors="pt", padding=True, truncation=True, max_length=tok.model_max_length).input_ids.to(device)
    text_emb = txtE(ids).last_hidden_state.half()

    del txtE
    torch.cuda.empty_cache()
    print("Text-encoder cleared from VRAM. Current mem:",
          f"{torch.cuda.memory_allocated()/1e6:.0f} MB")

    sched = DDIMScheduler.from_pretrained(args.model, subfolder="scheduler")
    sched.set_timesteps(args.steps, device=device)

    channels = 4
    latents = torch.randn(1, channels, H // 8, W // 8,
                          device=device, dtype=torch.float16)

    # ──────────────────────────────────────────
    # 2) UNet on GPU for denoising loop
    # ──────────────────────────────────────────
    print("[2/3] Loading UNet …")
    unet = (
        UNet2DConditionModel
        .from_pretrained(os.path.join(args.model, "unet"), torch_dtype=torch.float16)
        .to(device)
        .eval()
    )
    if args.lora:
        print(f"Applying LoRA weights from {args.lora} @ strength {args.lora_strength} …")
        unet = apply_lora(unet, args.lora, args.lora_strength)

    total_steps = len(sched.timesteps)
    print(f"Denoising for {total_steps} steps @ CFG {args.scale} …")

    t0 = time.time()
    for i, t in enumerate(sched.timesteps, 1):
        print(f"   step {i:02}/{total_steps}  t={int(t):>3}")
        lat_in = torch.cat([latents, latents], dim=0)
        if hasattr(unet, "model"):              # LoRA / PEFT wrapper
            noise = unet.model(
                sample=lat_in,
                timestep=t,
                encoder_hidden_states=text_emb,
            ).sample
        else:                                # vanilla UNet
            noise = unet(
                sample=lat_in,
                timestep=t,
                encoder_hidden_states=text_emb,
            ).sample
        n_uncond, n_text = noise.chunk(2)
        guided = n_uncond + args.scale * (n_text - n_uncond)
        latents = sched.step(guided, t, latents).prev_sample.half()
    print(f"UNet loop done in {time.time() - t0:.1f}s")

    del unet
    torch.cuda.empty_cache()
    print("UNet cleared from VRAM. Current mem:",
          f"{torch.cuda.memory_allocated()/1e6:.0f} MB")

    # ──────────────────────────────────────────
    # 3) VAE decode on GPU
    # ──────────────────────────────────────────
    print("[3/3] Loading VAE decoder …")
    vae = (
        AutoencoderKL
        .from_pretrained(os.path.join(args.model, "vae"), torch_dtype=torch.float16)
        .to(device)
        .eval()
    )

    print("Decoding latents …")
    img_latents = latents / 0.18215
    rgb = vae.decode(img_latents).sample[0].clamp(-1, 1)
    rgb = ((rgb + 1) / 2).cpu().permute(1, 2, 0).numpy()
    pil = Image.fromarray((rgb * 255).round().astype("uint8"))

    fn = _auto_name(args.output)
    pil.save(fn)
    print("Saved →", fn)


if __name__ == "__main__":
    main()