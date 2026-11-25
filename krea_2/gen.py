#!/usr/bin/env python
"""
Krea 2 generation entrypoint.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

from krea_2.config import (
    DEFAULT_RAW_GUIDANCE,
    DEFAULT_RAW_STEPS,
    DEFAULT_TURBO_STEPS,
    TEXT_MAX_LENGTH,
    image_token_count,
    resolve_model_root,
    resolve_outputs_dir,
)


if TYPE_CHECKING:
    from PIL import Image

    from krea_2.models.denoiser.transformer import Krea2Transformer2DModel

from flux2.gen import _parse_loras_arg

DEFAULT_SEED = 19930625
_PIXEL_TO_TOKEN = 16
_VAE_DTYPES = ("fp32", "fp16")


def _resolve_settings(
    distilled: bool,
    num_inference_steps: int | None,
    guidance_scale: float | None,
) -> tuple[int, float]:
    """Return the steps and the guidance scale that a run uses.

    Args:
        distilled: True for the turbo variant.
        num_inference_steps: The requested steps, or None for the default.
        guidance_scale: The requested guidance scale, or None for the default.

    Returns:
        A tuple of the steps and the guidance scale. The turbo variant always
        returns 0.0, because it does not use guidance.
    """
    if distilled:
        if guidance_scale not in (None, 0.0):
            print(f"  * warning: the turbo variant ignores guidance_scale={guidance_scale}")
        steps = DEFAULT_TURBO_STEPS if num_inference_steps is None else num_inference_steps
        return steps, 0.0
    steps = DEFAULT_RAW_STEPS if num_inference_steps is None else num_inference_steps
    scale = DEFAULT_RAW_GUIDANCE if guidance_scale is None else guidance_scale
    return steps, scale


def _vae_dtype(name: str):
    import torch

    if name == "fp32":
        return torch.float32
    if name == "fp16":
        return torch.float16
    raise ValueError(f"Unknown vae dtype {name!r}. Expected one of: {_VAE_DTYPES}.")


def _inject_loras(denoiser, loras: dict[str, float]) -> int:
    """Inject the LoRA wrappers that a checkpoint list needs."""
    from flux2.lora.model import inject_trainable_lora_modules
    from krea_2.lora.config import KREA2_LORA_TARGETS
    from safetensors import safe_open

    lora_rank = None
    lora_alpha = None
    for lora_path in loras:
        with safe_open(str(lora_path), framework="pt") as handle:
            metadata = handle.metadata() or {}
        lora_rank = int(metadata["rank"])
        lora_alpha = int(metadata["alpha"])
        break
    if lora_rank is None or lora_alpha is None:
        raise RuntimeError("Could not read rank/alpha metadata from LoRA checkpoint.")
    return len(
        inject_trainable_lora_modules(
            denoiser,
            target_linear_names=KREA2_LORA_TARGETS,
            rank=lora_rank,
            alpha=lora_alpha,
        )
    )


@contextlib.contextmanager
def _denoiser_on_device(denoiser, device, *, stream_denoiser: bool):
    """Place a caller denoiser for one denoise pass and restore it on exit."""
    from flux2.gen import _module_device
    from utils.quant.streaming import (
        enable_streaming,
        move_unstreamed_modules_to,
        pin_quantized_buffers,
    )

    original_device = _module_device(denoiser)
    try:
        if stream_denoiser:
            if enable_streaming(denoiser):
                pin_quantized_buffers(denoiser)
            move_unstreamed_modules_to(denoiser, device)
        else:
            denoiser.to(device)
        yield denoiser
    finally:
        if original_device is not None:
            if stream_denoiser:
                move_unstreamed_modules_to(denoiser, original_device)
            else:
                denoiser.to(original_device)


def denoise(
    denoiser,
    latents,
    sigmas,
    *,
    embeds,
    text_mask,
    position_ids,
    neg_embeds=None,
    neg_text_mask=None,
    guidance_scale: float = 0.0,
):
    """Run the Krea 2 Euler denoise loop.

    The denoiser must have its parameters on the device of `latents`. When it
    streams, its packed weights stay in host memory and each layer copies
    itself in for its own multiply.

    Args:
        denoiser: The Krea 2 denoiser.
        latents: The noise tokens with shape `(B, tokens, 64)`, on the device.
        sigmas: The sigmas from `build_sigmas`, on the device. It has one more
            value than the number of steps.
        embeds: The text embeddings with shape `(B, S, 12, 2560)`.
        text_mask: The bool text mask with shape `(B, S)`.
        position_ids: The rotary coordinates of the combined text and image
            sequence, with shape `(S + tokens, 3)`.
        neg_embeds: The negative text embeddings, or None.
        neg_text_mask: The negative text mask, or None.
        guidance_scale: The CFG scale. Used only when `neg_embeds` is not None.

    Returns:
        The denoised tokens in fp32 with the shape of `latents`.

    Raises:
        RuntimeError: If a step produces inf or NaN values.
    """
    import torch

    device = latents.device
    steps = len(sigmas) - 1
    embeds = embeds.to(device=device, dtype=torch.float16)
    text_mask = text_mask.to(device)
    position_ids = position_ids.to(device)
    use_cfg = neg_embeds is not None
    if use_cfg:
        neg_embeds = neg_embeds.to(device=device, dtype=torch.float16)
        neg_text_mask = neg_text_mask.to(device)

    with torch.inference_mode():
        for step_index in range(steps):
            print(f"  * {step_index + 1}/{steps} steps", flush=True)
            step_start = time.perf_counter()
            sigma, sigma_next = sigmas[step_index], sigmas[step_index + 1]
            timestep = sigma.expand(latents.shape[0])
            kwargs = {
                "timestep": timestep,
                "position_ids": position_ids,
                "return_dict": False,
            }
            velocity = denoiser(
                latents.half(),
                encoder_hidden_states=embeds,
                encoder_attention_mask=text_mask,
                **kwargs,
            )[0]
            if use_cfg:
                velocity_uncond = denoiser(
                    latents.half(),
                    encoder_hidden_states=neg_embeds,
                    encoder_attention_mask=neg_text_mask,
                    **kwargs,
                )[0]
                velocity = velocity + guidance_scale * (velocity - velocity_uncond)
            latents = latents + (sigma_next - sigma) * velocity.float()
            if not torch.isfinite(latents).all():
                raise RuntimeError(
                    f"Krea 2 denoise produced inf or NaN values at step {step_index + 1}/{steps}."
                )
            torch.cuda.synchronize(device)
            print(f"    * done in {time.perf_counter() - step_start:.3f}s", flush=True)
    return latents


def _make_noise(device, seed: int, height: int, width: int):
    import torch

    from krea_2.inputs import pack_latents

    shape = (1, 16, height // 8, width // 8)
    generator = torch.Generator(device=device).manual_seed(seed)
    try:
        noise = torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)
    except (RuntimeError, TypeError):
        generator = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn(shape, device=device, dtype=torch.float32, generator=generator).to(torch.bfloat16)
    return pack_latents(noise).to(torch.float16)


def generate_image(
    root: str | Path,
    *,
    prompt: str = "A cat holding a sign that says hello world",
    width: int = 1024,
    height: int = 1024,
    num_inference_steps: int | None = None,
    seed: int = DEFAULT_SEED,
    guidance_scale: float | None = None,
    distilled: bool = True,
    negative_prompt: str = "",
    text_quant_method: str | None = "q8-mini",
    denoiser_quant_method: str | None = "q6-mini",
    max_length: int = TEXT_MAX_LENGTH,
    stream_denoiser: bool = False,
    text_positions: str = "diffusers",
    vae_dtype: str = "fp32",
    tokenizer=None,
    text_encoder=None,
    denoiser: "Krea2Transformer2DModel | None" = None,
    loras: dict[str, float] | None = None,
    vae=None,
    num_images: int = 1,
) -> list["Image.Image"]:
    """Generate images with Krea 2.

    Each model is loaded from disk unless the caller passes it in. A passed
    model is moved to the CUDA device only while it is in use and returns to
    its original device afterwards.

    Args:
        root: Root folder that contains `krea_2/model`.
        prompt: Prompt text to encode.
        width: Output width in pixels. Rounds up to a multiple of 16.
        height: Output height in pixels. Rounds up to a multiple of 16.
        num_inference_steps: Denoising steps. Defaults to 8 (turbo) or 28
            (raw).
        seed: Base noise seed. Image `i` uses `seed + i`.
        guidance_scale: CFG scale. Used only by the raw variant. Defaults to
            4.5.
        distilled: True for the turbo variant, False for raw.
        negative_prompt: Negative prompt. Used only when guidance is on.
        text_quant_method: Text encoder quant method. Defaults to
            `q8-mini`.
        denoiser_quant_method: Denoiser quant method. Defaults to
            `q6-mini`.
        max_length: Number of text tokens after the prefix is dropped.
        stream_denoiser: Keep the quantized weights in pinned host memory and
            copy each layer to the GPU for its own multiply.
        text_positions: `diffusers` or `absolute`.
        vae_dtype: `fp32` or `fp16`.
        tokenizer: Pre-loaded tokenizer. Defaults to None.
        text_encoder: Pre-loaded text encoder. Defaults to None.
        denoiser: Pre-loaded denoiser. Defaults to None.
        loras: Map of LoRA checkpoint path to strength. The checkpoint metadata
            supplies the rank and the alpha. Only valid with a denoiser that
            this call loads.
        vae: Pre-loaded VAE. Defaults to None.
        num_images: Number of images to generate.

    Returns:
        The generated images.

    Raises:
        RuntimeError: If CUDA is not available, or if denoise produces inf or
            NaN values.
    """
    from flux2.gen import _on_device
    from krea_2.denoiser.loader import load_krea2_denoiser
    from krea_2.inputs import build_sigmas, decode_latents, encode_prompt, prepare_position_ids, round_to_multiple
    from krea_2.text_encoder.loader import load_krea2_text_encoder
    from krea_2.vae.loader import load_krea2_vae
    from utils.quant.streaming import (
        enable_streaming,
        move_unstreamed_modules_to,
        pin_quantized_buffers,
    )

    import torch
    from transformers import Qwen2TokenizerFast

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is not available in this environment.")

    model_root = resolve_model_root(root)
    steps, cfg = _resolve_settings(distilled, num_inference_steps, guidance_scale)
    variant = "turbo" if distilled else "raw"
    use_cfg = cfg > 0.0
    vae_compute_dtype = _vae_dtype(vae_dtype)

    aligned_width = round_to_multiple(width, _PIXEL_TO_TOKEN)
    aligned_height = round_to_multiple(height, _PIXEL_TO_TOKEN)
    if aligned_width != width or aligned_height != height:
        print(f"  * rounding size {width}x{height} to {aligned_width}x{aligned_height}")
    width, height = aligned_width, aligned_height

    owns_text_encoder = text_encoder is None
    owns_denoiser = denoiser is None
    owns_vae = vae is None

    print("1. Text Encoding")
    text_start = time.perf_counter()
    if tokenizer is None:
        print("  * loading tokenizer ...")
        tokenizer = Qwen2TokenizerFast.from_pretrained(str(model_root / "text_encoder"))
    else:
        print("  * using caller-provided tokenizer")

    if text_encoder is None:
        print("  * loading text encoder ...")
        print(f"    quantization: {text_quant_method or 'none'}")
        torch.cuda.empty_cache()
        text_encoder = load_krea2_text_encoder(model_root / "text_encoder", text_quant_method)
    else:
        print("  * using caller-provided text encoder")

    negative_embeds = None
    negative_text_mask = None
    with _on_device(text_encoder, device, restore=not owns_text_encoder):
        print("  * text encoding ...")
        torch.cuda.synchronize(device)
        encode_start = time.perf_counter()
        embeds, text_mask = encode_prompt(
            tokenizer,
            text_encoder,
            prompt,
            device=device,
            max_length=max_length,
            text_positions=text_positions,
        )
        if use_cfg:
            negative_embeds, negative_text_mask = encode_prompt(
                tokenizer,
                text_encoder,
                negative_prompt or "",
                device=device,
                max_length=max_length,
                text_positions=text_positions,
            )
        torch.cuda.synchronize(device)
        print(f"    encode time: {time.perf_counter() - encode_start:.3f}s")

    print(f"    text tokens: {embeds.shape[1]}")
    print("  * clearing encoder ...")
    del tokenizer
    del text_encoder
    torch.cuda.empty_cache()
    print(f"  * done in {time.perf_counter() - text_start:.3f}s")

    grid_h = height // _PIXEL_TO_TOKEN
    grid_w = width // _PIXEL_TO_TOKEN
    if grid_h < 1 or grid_w < 1:
        raise ValueError(f"width and height must be at least {_PIXEL_TO_TOKEN}.")
    text_len = embeds.shape[1]
    position_ids = prepare_position_ids(text_len, grid_h, grid_w, device)
    sigmas = build_sigmas(steps, image_token_count(width, height), distilled=distilled).to(device)
    print(f"  * output size: {grid_w * _PIXEL_TO_TOKEN}x{grid_h * _PIXEL_TO_TOKEN}")

    print("2. Denoise")
    denoise_start = time.perf_counter()
    if denoiser is None:
        print("  * loading transformer ...")
        print(f"    variant: {variant}")
        print(f"    quantization: {denoiser_quant_method or 'none'}")
        print(f"    stream denoiser: {'on' if stream_denoiser else 'off'}")
        torch.cuda.empty_cache()
        transformer = load_krea2_denoiser(
            model_root / "denoiser",
            denoiser_quant_method,
            distilled=distilled,
            offloading=stream_denoiser,
            stream_denoiser=stream_denoiser,
        )
    else:
        print("  * using caller-provided transformer")
        transformer = denoiser

    if loras:
        if not owns_denoiser:
            raise ValueError("loras needs a denoiser loaded by this call, not a caller-provided one.")
        from flux2.lora.loader import load_checkpoint as load_lora_checkpoint
        from flux2.lora.model import TrainableLoraLinear

        print("  * applying loras ...")
        print(f"    injected lora modules: {_inject_loras(transformer, loras)}")
        for lora_path, strength in loras.items():
            print(f"    {lora_path} [{strength:.2f}]")
            load_lora_checkpoint(transformer, lora_path)
            if strength != 1.0:
                scale = strength ** 0.5
                for module in transformer.modules():
                    if isinstance(module, TrainableLoraLinear):
                        module.lora_A.data.mul_(scale)
                        module.lora_B.data.mul_(scale)

    if owns_denoiser:
        if stream_denoiser:
            if enable_streaming(transformer):
                pin_quantized_buffers(transformer)
            move_unstreamed_modules_to(transformer, device)
        else:
            transformer = transformer.to(device)
        placement = contextlib.nullcontext()
    else:
        placement = _denoiser_on_device(denoiser, device, stream_denoiser=stream_denoiser)

    latents_list = []
    with placement:
        for image_index in range(num_images):
            print(f"  * image {image_index + 1}/{num_images} (seed {seed + image_index})")
            latents = _make_noise(device, seed + image_index, height, width)
            latents = denoise(
                transformer,
                latents,
                sigmas,
                embeds=embeds,
                text_mask=text_mask,
                position_ids=position_ids,
                neg_embeds=negative_embeds if use_cfg else None,
                neg_text_mask=negative_text_mask if use_cfg else None,
                guidance_scale=cfg,
            )
            latents_list.append(latents)

    torch.cuda.synchronize(device)
    if owns_denoiser:
        print("  * clearing transformer ...")
        del transformer
    torch.cuda.empty_cache()
    print(f"  * done in {time.perf_counter() - denoise_start:.3f}s")

    print("3. VAE")
    vae_start = time.perf_counter()
    if vae is None:
        vae = load_krea2_vae(model_root / "vae")
    images = []
    with _on_device(vae, device, dtype=vae_compute_dtype, restore=not owns_vae):
        for latents in latents_list:
            images.append(decode_latents(vae, latents, height, width, dtype=vae_compute_dtype))
        torch.cuda.synchronize(device)
    print(f"  * done in {time.perf_counter() - vae_start:.3f}s")

    return images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Krea 2 generation.")
    parser.add_argument("--root", required=True, help="Root folder that contains krea_2/model.")
    parser.add_argument(
        "--prompt",
        default="A cat holding a sign that says hello world",
        help="Prompt text to encode.",
    )
    parser.add_argument("--width", type=int, default=1024, help="Output width. Default: 1024.")
    parser.add_argument("--height", type=int, default=1024, help="Output height. Default: 1024.")
    parser.add_argument(
        "--steps",
        type=int,
        help=f"Denoising steps. Default: {DEFAULT_TURBO_STEPS} for turbo, {DEFAULT_RAW_STEPS} with --raw.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Noise seed. Default: {DEFAULT_SEED}.")
    parser.add_argument("--num-images", type=int, default=1, help="Number of images. Default: 1.")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Use the raw variant with CFG instead of the turbo variant.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        help=f"CFG scale for --raw. Default: {DEFAULT_RAW_GUIDANCE}. The turbo variant ignores it.",
    )
    parser.add_argument("--negative-prompt", default="", help="Negative prompt for --raw. Default: empty.")
    parser.add_argument(
        "--loras",
        help='Optional comma-separated LoRA list in the form "path:strength, path2:strength2".',
    )
    parser.add_argument(
        "--text-quant-method",
        default="q8-mini",
        help='Text encoder quantization method. "fp16" reads the fp16 quant file.',
    )
    parser.add_argument(
        "--denoiser-quant-method",
        default="q6-mini",
        help='Denoiser quantization method. "fp16" reads the fp16 quant file.',
    )
    parser.add_argument("--max-length", type=int, default=TEXT_MAX_LENGTH, help="Number of text tokens.")
    parser.add_argument(
        "--stream-denoiser",
        dest="stream_denoiser",
        action="store_true",
        help=(
            "Keep the quantized weights in pinned host memory and copy each layer to the GPU "
            "for its own multiply."
        ),
    )
    parser.add_argument(
        "--vae-dtype",
        choices=_VAE_DTYPES,
        default="fp32",
        help="VAE compute dtype. Default: fp32.",
    )
    parser.add_argument(
        "--text-positions",
        choices=("diffusers", "absolute"),
        default="diffusers",
        help="Text position ids. Default: diffusers (cumulative valid tokens).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from PIL.PngImagePlugin import PngInfo

    distilled = not args.raw
    steps, guidance_scale = _resolve_settings(distilled, args.steps, args.guidance_scale)

    images = generate_image(
        args.root,
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=steps,
        seed=args.seed,
        guidance_scale=guidance_scale,
        distilled=distilled,
        negative_prompt=args.negative_prompt,
        text_quant_method=args.text_quant_method,
        denoiser_quant_method=args.denoiser_quant_method,
        max_length=args.max_length,
        stream_denoiser=args.stream_denoiser,
        text_positions=args.text_positions,
        vae_dtype=args.vae_dtype,
        num_images=args.num_images,
        loras=_parse_loras_arg(args.loras),
    )

    output_dir = resolve_outputs_dir(args.root)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    aligned_width = ((args.width + _PIXEL_TO_TOKEN - 1) // _PIXEL_TO_TOKEN) * _PIXEL_TO_TOKEN
    aligned_height = ((args.height + _PIXEL_TO_TOKEN - 1) // _PIXEL_TO_TOKEN) * _PIXEL_TO_TOKEN
    metadata = {
        "seed": args.seed,
        "steps": steps,
        "variant": "raw" if args.raw else "turbo",
        "guidance_scale": guidance_scale,
        "text_encoder_quant": args.text_quant_method,
        "denoiser_quant": args.denoiser_quant_method,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt if not distilled else "",
        "width": aligned_width,
        "height": aligned_height,
        "text_positions": args.text_positions,
        "vae_dtype": args.vae_dtype,
        "loras": args.loras or "",
    }
    for index, image in enumerate(images):
        suffix = "" if len(images) == 1 else f"-{index + 1}"
        output_path = output_dir / f"{timestamp}{suffix}.png"
        pnginfo = PngInfo()
        pnginfo.add_text("krea_2", json.dumps(metadata))
        image.save(output_path, pnginfo=pnginfo)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
