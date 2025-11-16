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
import time
from typing import Dict, Iterable

import torch
from PIL import Image
import numpy as np
import copy
from diffusers import AutoencoderKL
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
)
from safetensors.torch import load_file as load_safetensors
from transformers import (
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)
from diffusers.loaders.single_file_utils import create_diffusers_clip_model_from_ldm

from sdxl.unet import SDXLUNet, load_unet_key_mapping


DEFAULT_PROMPT = "A propaganda poster depicting a cat dressed as french emperor napoleon holding a piece of cheese."

# DEFAULT_PROMPT = "a close-up of a fire spitting dragon, cinematic shot."


SD_SCALE = 0.18215
REQUIRED_SUBFOLDERS = (
    "tokenizer",
    "tokenizer_2",
    "text_encoder",
    "text_encoder_2",
    "vae",
    "scheduler",
)
UNET_PREFIXES = ("model.diffusion_model.",)
TEXT_ENCODER_PREFIXES = ("conditioner.embedders.0.",)
TEXT_ENCODER2_PREFIXES = ("conditioner.embedders.1.",)
VAE_PREFIXES = ("first_stage_model.",)


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


def _check_base_layout(base_dir: str) -> None:
    missing = [
        name for name in REQUIRED_SUBFOLDERS if not os.path.isdir(os.path.join(base_dir, name))
    ]
    if missing:
        raise SystemExit(
            f"Base directory {base_dir} is missing required subfolders: {', '.join(missing)}"
        )


def _extract_module_state(
    state_dict: Dict[str, torch.Tensor],
    prefixes: Iterable[str],
) -> Dict[str, torch.Tensor]:
    extracted: Dict[str, torch.Tensor] = {}
    for key in list(state_dict.keys()):
        for prefix in prefixes:
            if key.startswith(prefix):
                extracted[key[len(prefix) :]] = state_dict.pop(key)
                break
    return extracted


def _extract_module_state_with_prefix(
    state_dict: Dict[str, torch.Tensor],
    prefixes: Iterable[str],
) -> Dict[str, torch.Tensor]:
    extracted: Dict[str, torch.Tensor] = {}
    for key in list(state_dict.keys()):
        for prefix in prefixes:
            if key.startswith(prefix):
                extracted[key] = state_dict.pop(key)
                break
    return extracted


def _split_checkpoint(
    state_dict: Dict[str, torch.Tensor],
) -> tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, int]]:
    remaining = dict(state_dict)
    text1 = _extract_module_state(remaining, TEXT_ENCODER_PREFIXES)
    text2 = _extract_module_state(remaining, TEXT_ENCODER2_PREFIXES)
    vae = _extract_module_state(remaining, VAE_PREFIXES)
    unet = _extract_module_state_with_prefix(remaining, UNET_PREFIXES)
    modules = {
        "text_encoder": text1,
        "text_encoder_2": text2,
        "vae": vae,
        "unet": unet,
    }
    stats = {
        "total": len(state_dict),
        "text_encoder": len(text1),
        "text_encoder_2": len(text2),
        "vae": len(vae),
        "unet": len(unet),
        "unassigned": len(remaining),
    }
    return modules, stats


def _require_state(
    state: Dict[str, torch.Tensor],
    module_name: str,
) -> Dict[str, torch.Tensor]:
    if not state:
        raise SystemExit(f"Checkpoint missing weights for {module_name}.")
    return state


def _build_scheduler(base_dir: str, sampler: str, schedule: str):
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

    scheduler = cls.from_pretrained(base_dir, subfolder="scheduler")
    use_karras = schedule_name.startswith("karras")
    config = getattr(scheduler, "config", None)
    if config is not None and hasattr(config, "use_karras_sigmas"):
        config.use_karras_sigmas = use_karras
    elif hasattr(scheduler, "use_karras_sigmas"):
        scheduler.use_karras_sigmas = use_karras
    return scheduler


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
    parser.add_argument("--model", required=True, help="Path to combined SDXL .safetensors checkpoint.")
    parser.add_argument("--output", required=True, help="Directory where the PNG result is written.")
    parser.add_argument(
        "--base",
        required=True,
        help="Directory containing tokenizer/, text_encoder/, vae/, scheduler/ configs.",
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
    parser.add_argument("--height", type=int, default=512, help="Image height (multiple of 8).")
    parser.add_argument("--width", type=int, default=512, help="Image width (multiple of 8).")
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
    if not args.model.endswith(".safetensors"):
        raise SystemExit("--model must point to a .safetensors file.")

    base_dir = args.base
    _check_base_layout(base_dir)
    ckpt_path = args.model

    if args.steps < 1:
        raise SystemExit("--steps must be >= 1")

    width = max(64, (args.width // 8) * 8)
    height = max(64, (args.height // 8) * 8)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA device is required for INT8 SDXL inference.")
    device = torch.device("cuda")
    dtype = torch.float16

    torch.manual_seed(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    print(f"[ckpt] Loading weights from {ckpt_path} …")
    state_dict = load_safetensors(ckpt_path, device="cpu")

    modules, ckpt_stats = _split_checkpoint(state_dict)
    print(
        "[ckpt] tensor counts "
        f"total={ckpt_stats['total']} "
        f"text1={ckpt_stats['text_encoder']} "
        f"text2={ckpt_stats['text_encoder_2']} "
        f"vae={ckpt_stats['vae']} "
        f"unet={ckpt_stats['unet']} "
        f"unassigned={ckpt_stats['unassigned']}"
    )
    unet_state = _require_state(modules.get("unet", {}), "unet")

    print("\n[tokenizer 1]")
    tokenizer = CLIPTokenizer.from_pretrained(os.path.join(base_dir, "tokenizer"))
    tokens = tokenizer(
        [args.negative, args.prompt],
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    for idx, input_ids in enumerate(tokens["input_ids"]):
        print(idx, input_ids.tolist())

    print("\n[tokenizer 2]")
    tokenizer_2 = CLIPTokenizer.from_pretrained(os.path.join(base_dir, "tokenizer_2"))
    tokens_2 = tokenizer_2(
        [args.negative, args.prompt],
        padding="max_length",
        truncation=True,
        max_length=tokenizer_2.model_max_length,
        return_tensors="pt",
    )
    for idx, input_ids in enumerate(tokens_2["input_ids"]):
        print(idx, input_ids.tolist())

    print("\n[text encoder 1]")
    config_path_1 = os.path.join(base_dir, "text_encoder")
    text_encoder = create_diffusers_clip_model_from_ldm(
        CLIPTextModel,
        copy.deepcopy(state_dict),
        config=config_path_1,
        torch_dtype=dtype,
        local_files_only=True,
    ).to(device=device, dtype=dtype).eval()
    
    with torch.no_grad():
        inputs_1 = {k: v.to(device=device) for k, v in tokens.items()}
        hidden_gpu = text_encoder(**inputs_1).last_hidden_state
    print("  mean:", hidden_gpu.abs().mean().item())
    hidden_1 = hidden_gpu.detach().to("cpu", dtype=dtype)
    del text_encoder, hidden_gpu, inputs_1
    torch.cuda.empty_cache()

    print("\n[text encoder 2]")
    config_path_2 = os.path.join(base_dir, "text_encoder_2")
    text_encoder_2 = create_diffusers_clip_model_from_ldm(
        CLIPTextModelWithProjection,
        copy.deepcopy(state_dict),
        config=config_path_2,
        torch_dtype=dtype,
        local_files_only=True,
    ).to(device=device, dtype=dtype).eval()
    
    with torch.no_grad():
        inputs_2 = {k: v.to(device=device) for k, v in tokens_2.items()}
        hidden_gpu = text_encoder_2(**inputs_2).last_hidden_state
    print("  mean:", hidden_gpu.abs().mean().item())
    hidden_2 = hidden_gpu.detach().to("cpu", dtype=dtype)
    del text_encoder_2, hidden_gpu, inputs_2
    torch.cuda.empty_cache()


    prompt_embeds = torch.cat([hidden_2, hidden_1], dim=-1)
    del state_dict

    scheduler = _build_scheduler(base_dir, args.sampler, args.scheduler)
    scheduler.set_timesteps(args.steps, device=device)

    unet_key_mapping = load_unet_key_mapping(base_dir)

    print("\n[unet]")
    unet = SDXLUNet(model=unet_state, key_mapping=unet_key_mapping)
    unet = unet.to(device=device, dtype=dtype).eval()
    p = next(unet.parameters())
    print(
        "unet weight dtype/mean/std:",
        p.dtype,
        p.float().mean().item(),
        p.float().std().item(),
    )
    with torch.no_grad():
        flat = torch.cat([param.detach().float().reshape(-1) for param in unet.parameters()])
    print("unet params mean/std:", flat.mean().item(), flat.std().item())

    latents = _prepare_latents(height, width, generator, device, dtype, scheduler)
    prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)

    print(f"[sample] Running {len(scheduler.timesteps)} denoising steps @ CFG {args.cfg}")
    for step_idx, t in enumerate(scheduler.timesteps, 1):
        step_start = time.perf_counter()
        latent_model_input = latents.repeat(2, 1, 1, 1)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        latent_model_input = latent_model_input.to(dtype=dtype)

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
        step_time = time.perf_counter() - step_start
        print(f"  step {step_idx}/{len(scheduler.timesteps)} {step_time:.2f}s")

    del unet
    torch.cuda.empty_cache()

    print("[check] final latents mean/std:", latents.mean().item(), latents.std().item())

    print("\n[vae]")
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path=base_dir,
        subfolder="vae",
        torch_dtype=torch.float32
    )
    vae.to(device=device, dtype=torch.float32).eval()

    print("[decode] Converting latents to image …")
    latents = latents / SD_SCALE
    vae_param = next(vae.parameters())
    latents = latents.to(device=vae_param.device, dtype=vae_param.dtype)
    test_latents = latents / vae.config.scaling_factor
    test_img = vae.decode(test_latents).sample
    print("[check] decoded image mean/std:", test_img.mean().item(), test_img.std().item())
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image[0].permute(1, 2, 0).detach().cpu().numpy()
    image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
    image = (image * 255).round().clip(0, 255).astype("uint8")
    pil_image = Image.fromarray(image)

    out_path = _auto_name(args.output)
    pil_image.save(out_path)
    print(f"[done] Saved → {out_path}")


if __name__ == "__main__":
    main()
