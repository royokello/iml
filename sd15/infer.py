#!/usr/bin/env python
"""
Stable Diffusion v1.5 Inference

```
py -m sd15.infer --model "/models/sd15" --output "/pictures" --prompt "analogue photo, ultrawide shot of a waterfall"
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

def _auto_name(out_dir: str) -> str:
    """Return a non‑clashing PNG path in *out_dir* using a timestamp."""
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    for idx in range(1, 100):
        fn = f"{ts}-{idx:02}.png"
        fp = os.path.join(out_dir, fn)
        if not os.path.exists(fp):
            return fp
    raise RuntimeError("Too many files created in one second; try again.")

@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser(description="Stable Diffusion v1.5 off‑load inference")
    ap.add_argument("--model", required=True, help="Folder containing tokenizer/, text_encoder/, unet/, vae/, scheduler/")
    ap.add_argument("--prompt", required=True, help="Text prompt")
    ap.add_argument("--output", required=True, help="Folder for PNGs")
    ap.add_argument("--steps", type=int, default=24, help="DDIM steps ≥1")
    ap.add_argument("--size", default="512,512", help="WIDTH,HEIGHT (multiples of 8)")
    ap.add_argument("--scale", type=float, default=7.5, help="Classifier‑free guidance scale")
    ap.add_argument("--seed", type=int, default=93, help="Seed for RNG")
    args = ap.parse_args()

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
        raise RuntimeError("Off‑load script requires CUDA GPU")

    # ──────────────────────────────────────────
    # 1) Tokenizer (CPU) + text‑encoder (GPU)
    # ──────────────────────────────────────────
    print("[1/3] Loading tokenizer + CLIP text‑encoder …")
    tok = CLIPTokenizer.from_pretrained(os.path.join(args.model, "tokenizer"))
    txtE = (CLIPTextModel
            .from_pretrained(os.path.join(args.model, "text_encoder"), torch_dtype=torch.float16)
            .to(device)
            .eval())

    print("Encoding prompt …")
    ids = tok(["", args.prompt], return_tensors="pt", padding=True).input_ids.to(device)
    text_emb = txtE(ids).last_hidden_state.half()

    del txtE
    torch.cuda.empty_cache()
    print("Text‑encoder cleared from VRAM. Current mem:", f"{torch.cuda.memory_allocated()/1e6:.0f} MB")

    sched = DDIMScheduler.from_pretrained(args.model, subfolder="scheduler")
    sched.set_timesteps(args.steps, device=device)

    channels = 4
    latents = torch.randn(1, channels, H // 8, W // 8, device=device, dtype=torch.float16)

    # ──────────────────────────────────────────
    # 2) UNet on GPU for denoising loop
    # ──────────────────────────────────────────
    print("[2/3] Loading UNet …")
    unet = (UNet2DConditionModel
            .from_pretrained(os.path.join(args.model, "unet"), torch_dtype=torch.float16)
            .to(device)
            .eval())

    total_steps = len(sched.timesteps)
    print(f"Denoising for {total_steps} steps @ CFG {args.scale} …")

    t0 = time.time()
    for i, t in enumerate(sched.timesteps, 1):
        print(f"   step {i:02}/{total_steps}  t={int(t):>3}")
        lat_in = torch.cat([latents, latents], dim=0)
        noise = unet(lat_in, t, encoder_hidden_states=text_emb).sample
        n_uncond, n_text = noise.chunk(2)
        guided = n_uncond + args.scale * (n_text - n_uncond)
        latents = sched.step(guided, t, latents).prev_sample.half()
    print(f"UNet loop done in {time.time() - t0:.1f}s")

    del unet
    torch.cuda.empty_cache()
    print("UNet cleared from VRAM. Current mem:", f"{torch.cuda.memory_allocated()/1e6:.0f} MB")

    # ──────────────────────────────────────────
    # 3) VAE decode on GPU
    # ──────────────────────────────────────────
    print("[3/3] Loading VAE decoder …")
    vae = (AutoencoderKL
           .from_pretrained(os.path.join(args.model, "vae"), torch_dtype=torch.float16)
           .to(device)
           .eval())

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
