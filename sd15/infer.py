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

import torch
from PIL import Image
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTokenizer, CLIPTextModel


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


def _encode_text_cpu(model_root: str, prompt: str, negative: str) -> torch.Tensor:
    tok = CLIPTokenizer.from_pretrained(os.path.join(model_root, "tokenizer"))
    txtE = CLIPTextModel.from_pretrained(
        os.path.join(model_root, "text_encoder"),
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False,   # force real weights, not meta
        device_map=None            # no lazy device map
    )  # stays on CPU

    batch = tok([negative or "", prompt],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=tok.model_max_length)
    with torch.inference_mode():
        out = txtE(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
        text_emb = out.last_hidden_state

    # Make 100% sure this is a real, materialized CPU tensor (not meta / view)
    text_emb = text_emb.detach().contiguous().to("cpu", copy=True)

    # free the text models ASAP
    del tok, txtE
    return text_emb


def _load_unet_gpu(model_root: str, device: torch.device) -> UNet2DConditionModel:
    kwargs = dict(torch_dtype=torch.float16, use_safetensors=True)
    try:
        # diffusers >= 0.24 supports this; fastest if flash-attn is available
        kwargs["attn_implementation"] = "flash_attention_2"
    except TypeError:
        pass

    unet = UNet2DConditionModel.from_pretrained(os.path.join(model_root, "unet"), **kwargs)
    unet = unet.to(device=device, memory_format=torch.channels_last).eval()

    # Fallbacks if flash-attn isn’t available
    try:
        unet.enable_xformers_memory_efficient_attention()
    except Exception:
        try:
            unet.set_attn_processor(AttnProcessor2_0())
        except Exception:
            pass

    # Optional: compile for extra speed on GPUs that support Triton (SM ≥ 7.0)
    try:
        major, _ = torch.cuda.get_device_capability(device)
    except Exception:
        major = 0

    if major >= 7:
        try:
            unet = torch.compile(unet, mode="max-autotune", fullgraph=False)
        except Exception:
            pass

    return unet


def _load_vae_gpu(model_root: str, device: torch.device) -> AutoencoderKL:
    vae = AutoencoderKL.from_pretrained(
        os.path.join(model_root, "vae"),
        torch_dtype=torch.float16,
    )
    vae = vae.to(device=device, dtype=torch.float16, memory_format=torch.channels_last).eval()
    return vae

@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser(description="Stable Diffusion v1.5 off-load inference")
    ap.add_argument("--model", required=True, help="Folder containing tokenizer/, text_encoder/, unet/, vae/, scheduler/")
    ap.add_argument("--prompt", required=True, help="Text prompt")
    ap.add_argument("--output", required=True, help="Folder for PNGs")
    ap.add_argument("--steps", type=int, default=20, help="DDIM steps ≥1")
    ap.add_argument("--size", default="512,512")
    ap.add_argument("--scale", type=float, default=7.5, help="Classifier-free guidance scale")
    ap.add_argument("--seed", type=int, default=None, help="Seed for RNG (random if omitted)")
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

    if args.seed is None:
        seed = random.SystemRandom().randint(0, 2**32 - 1)
        print(f"No seed provided; using random seed {seed}")
    else:
        seed = args.seed
        print(f"Seeded RNG with {seed}")
    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Off-load script requires CUDA GPU")

    print("[prep] Building scheduler/latents …")
    sched = DDIMScheduler.from_pretrained(args.model, subfolder="scheduler")
    sched.set_timesteps(args.steps, device=device)
    latents = torch.randn(1, 4, H // 8, W // 8, device=device, dtype=torch.float16)
    latents = latents.to(memory_format=torch.channels_last)

    print("[text] Encoding prompt on CPU …")
    t0 = time.time()
    text_emb_cpu = _encode_text_cpu(args.model, args.prompt, args.negative)
    print(f"[text] Encoded on CPU in {time.time()-t0:.1f}s")

    print("[unet] Loading UNet on GPU …")
    unet = _load_unet_gpu(args.model, device)
    print("[unet] Loaded; starting diffusion …")

    # move text embeddings to GPU fp16 for denoising
    text_emb = text_emb_cpu.to(device=device, dtype=torch.float16)

    total_steps = len(sched.timesteps)
    print(f"Denoising for {total_steps} steps @ CFG {args.scale} …")
    t1 = time.time()

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        for i, t in enumerate(sched.timesteps, 1):
            # avoid extra allocs: repeat instead of cat each step
            lat_in = latents.repeat(2, 1, 1, 1)  # [2,4,h,w], channels_last inherited
            noise = unet(sample=lat_in, timestep=t, encoder_hidden_states=text_emb).sample
            n_uncond, n_text = noise.chunk(2)
            guided = n_uncond + args.scale * (n_text - n_uncond)
            latents = sched.step(guided, t, latents).prev_sample
            latents = latents.to(dtype=torch.float16, memory_format=torch.channels_last, non_blocking=True)

    print(f"[unet] Loop done in {time.time()-t1:.1f}s")

    # cleanup UNet VRAM right before decode
    del unet
    torch.cuda.empty_cache()

    print("[vae] Loading VAE on GPU …")
    vae = _load_vae_gpu(args.model, device)

    print("[decode] Decoding latents …")
    img_latents = latents / 0.18215
    rgb = vae.decode(img_latents).sample[0].clamp(-1, 1)
    rgb = ((rgb + 1) / 2).cpu().permute(1, 2, 0).numpy()
    pil = Image.fromarray((rgb * 255).round().astype("uint8"))

    fn = _auto_name(args.output)
    pil.save(fn)
    print("Saved →", fn)

if __name__ == "__main__":
    main()
