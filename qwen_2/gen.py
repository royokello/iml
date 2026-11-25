#!/usr/bin/env python
"""Qwen Image 2.1 text-to-image and image-editing entrypoint."""

from __future__ import annotations

import argparse
import gc
import json
import time
from collections.abc import Sequence
from pathlib import Path

from PIL import Image

from flux2.gen import _parse_loras_arg, _split_image_paths
from qwen_2.config import (
    BASE_STEPS,
    DEFAULT_DENOISER_QUANT_METHOD,
    DEFAULT_HEIGHT,
    DEFAULT_REF_SIZE,
    DEFAULT_SEED,
    DEFAULT_TEXT_QUANT_METHOD,
    DEFAULT_WIDTH,
    VARIANTS,
    calculate_shift,
    image_token_count,
    resolve_model_root,
    resolve_outputs_dir,
    resolve_sizes,
    round_to_multiple,
    SIZE_MULTIPLE,
)
from qwen_2.inputs import (
    build_img_mask,
    build_img_shapes,
    build_sigmas,
    decode_latents,
    encode_prompt,
    encode_reference_latents,
    make_noise,
)
from qwen_2.lora.inject import apply_loras

__all__ = ["generate_image"]

_VAE_DTYPES = ("fp16", "fp32")


def _vae_dtype(name: str):
    import torch

    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unknown VAE dtype {name!r}. Expected one of {_VAE_DTYPES}.")


def _build_scheduler(variant: str):
    from diffusers import FlowMatchEulerDiscreteScheduler

    if variant == "turbo":
        return FlowMatchEulerDiscreteScheduler(
            use_dynamic_shifting=False,
            shift_terminal=None,
            base_image_seq_len=256,
            max_image_seq_len=8192,
            base_shift=0.5,
            max_shift=0.9,
        )
    return FlowMatchEulerDiscreteScheduler(
        use_dynamic_shifting=True,
        shift_terminal=0.02,
        base_image_seq_len=256,
        max_image_seq_len=8192,
        base_shift=0.5,
        max_shift=0.9,
    )


def generate_image(
    root: str | Path,
    *,
    prompt: str,
    negative_prompt: str = "",
    width: int | None = None,
    height: int | None = None,
    num_inference_steps: int | None = None,
    guidance_scale: float = 1.0,
    seed: int = DEFAULT_SEED,
    variant: str = "turbo",
    text_quant_method: str,
    denoiser_quant_method: str,
    stream_denoiser: bool = False,
    kv_cache: bool = True,
    stream_text_encoder: bool = False,
    vae_dtype: str = "fp16",
    rgba: bool = True,
    images: Sequence[str] | None = None,
    ref_size: int = DEFAULT_REF_SIZE,
    loras: dict[str, float] | None = None,
    output_dir: str | Path | None = None,
):
    """Generate images from a prompt.

    Args:
        root: Root folder that contains `qwen_2/model`.
        prompt: The prompt text.
        negative_prompt: The negative prompt. Used only when
            `guidance_scale > 1`.
        width: Output width in pixels, a multiple of 32. None takes the last
            reference's width, else `DEFAULT_WIDTH`.
        height: Output height in pixels, a multiple of 32. None takes the last
            reference's height, else `DEFAULT_HEIGHT`.
        num_inference_steps: Denoising steps. Defaults to 8 (turbo) or 40
            (base).
        guidance_scale: Classifier-free guidance scale. Turbo uses 1.0.
        seed: Noise seed.
        variant: `turbo` or `base`.
        text_quant_method: Text encoder quant method, or `fp16`, `bf16`, or
            `fp32` for the unquantized checkpoint. Required: a name must be
            given, and `DEFAULT_TEXT_QUANT_METHOD` holds the CLI value. The
            loader never quantizes on the fly.
        denoiser_quant_method: Denoiser quant method, or `fp16`, `bf16`, or
            `fp32` for the unquantized checkpoint. Required for the same
            reason. `DEFAULT_DENOISER_QUANT_METHOD` holds the CLI value.
        stream_denoiser: Keep the quantized weights in pinned host memory and
            copy each layer to the GPU for its own multiply.
        kv_cache: Reuse the text prefix keys and values across steps.
        stream_text_encoder: Keep the quantized text-encoder weights in pinned
            host memory and copy each layer to the GPU for its own multiply.
        vae_dtype: `fp16` or `fp32`.
        rgba: Keep 4 channels. False flattens alpha over white.
        images: Reference image paths. One or more. None means text-to-image.
        ref_size: The reference area side length, a multiple of 32. Each
            reference is resized to this area at its own aspect ratio.
        loras: Map of LoRA checkpoint path to strength. The LoRA applies to
            the denoiser. None means no LoRA.
        output_dir: Folder for the PNG files. Defaults to
            `<root>/qwen_2/outputs`.

    Returns:
        The generated image.

    Raises:
        RuntimeError: If CUDA is not available, or if denoise produces inf or
            NaN values.
        ValueError: If the variant, a size, or a quant method is invalid.
    """
    import torch
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache
    from transformers import Qwen3VLProcessor

    from qwen_2.denoiser.loader import load_qwen2_denoiser
    from qwen_2.text_encoder.loader import load_qwen2_text_encoder
    from qwen_2.vae.loader import load_qwen2_vae
    from utils.quant.streaming import (
        enable_streaming,
        move_unstreamed_modules_to,
        pin_quantized_buffers,
    )

    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}. Expected one of {VARIANTS}.")
    if not text_quant_method:
        raise ValueError("text_quant_method is required.")
    if not denoiser_quant_method:
        raise ValueError("denoiser_quant_method is required.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this environment.")

    device = torch.device("cuda")
    model_root = resolve_model_root(root)
    steps = num_inference_steps
    if steps is None:
        steps = 8 if variant == "turbo" else BASE_STEPS
    use_cfg = guidance_scale > 1.0 and bool(negative_prompt)

    references = [Image.open(image).convert("RGBA") for image in images] if images else []
    derived_sizes, (width, height) = resolve_sizes(
        [reference.size for reference in references] or None, width, height, ref_size
    )
    # One resize feeds both copies: the vision tower and the VAE must see the
    # same reference at the same size, or the slots and the latent block disagree.
    references = [
        reference.resize(size, Image.LANCZOS)
        for reference, size in zip(references, derived_sizes)
    ]

    aligned_width = round_to_multiple(width, SIZE_MULTIPLE)
    aligned_height = round_to_multiple(height, SIZE_MULTIPLE)
    if (aligned_width, aligned_height) != (width, height):
        print(f"  * rounding size {width}x{height} to {aligned_width}x{aligned_height}")
    width, height = aligned_width, aligned_height

    print("1. Text Encoding")
    text_start = time.perf_counter()
    if references:
        sizes = ", ".join(f"{ref_width}x{ref_height}" for ref_width, ref_height in derived_sizes)
        print(f"  * references: {len(references)} ({sizes})")
    print("  * loading processor ...")
    processor = Qwen3VLProcessor.from_pretrained(
        str(model_root / "text_encoder"), local_files_only=True
    )
    print("  * loading text encoder ...")
    print(f"    quantization: {text_quant_method or 'none'}")
    torch.cuda.empty_cache()
    text_encoder = load_qwen2_text_encoder(
        model_root / "text_encoder", text_quant_method, include_vision=bool(references)
    )
    if stream_text_encoder:
        if enable_streaming(text_encoder):
            pin_quantized_buffers(text_encoder)
        move_unstreamed_modules_to(text_encoder, device)
    else:
        text_encoder = text_encoder.to(device)
    print("  * text encoding ...")
    torch.cuda.synchronize(device)
    encode_start = time.perf_counter()
    with torch.inference_mode():
        embeds, text_mask, image_pad_mask = encode_prompt(
            processor, text_encoder, prompt, device=device, images=references or None
        )
        negative_embeds = None
        negative_text_mask = None
        negative_image_pad_mask = None
        if use_cfg:
            negative_embeds, negative_text_mask, negative_image_pad_mask = encode_prompt(
                processor, text_encoder, negative_prompt, device=device, images=references or None
            )
    torch.cuda.synchronize(device)
    print(f"    encode time: {time.perf_counter() - encode_start:.3f}s")
    embeds = embeds.to(device)
    text_mask = text_mask.to(device).bool()
    if negative_embeds is not None:
        negative_embeds = negative_embeds.to(device)
        negative_text_mask = negative_text_mask.to(device).bool()
    print(f"    text tokens: {embeds.shape[1]}")
    print("  * clearing encoder ...")
    del text_encoder, processor
    gc.collect()
    torch.cuda.empty_cache()

    ref_latents = None
    if references:
        print("  * loading vae for the reference latents ...")
        reference_start = time.perf_counter()
        vae = load_qwen2_vae(model_root / "vae", dtype=_vae_dtype(vae_dtype)).to(device)
        with torch.inference_mode():
            ref_latents = encode_reference_latents(
                vae, references, derived_sizes, dtype=_vae_dtype(vae_dtype)
            )
        torch.cuda.synchronize(device)
        print(f"    reference tokens: {ref_latents.shape[1]}")
        print(f"    encode time: {time.perf_counter() - reference_start:.3f}s")
        del vae
        gc.collect()
        torch.cuda.empty_cache()
    print(f"  * done in {time.perf_counter() - text_start:.3f}s")

    tokens = image_token_count(width, height)
    sigmas = build_sigmas(steps, variant)
    img_shapes = build_img_shapes(height, width, derived_sizes)
    img_mask = build_img_mask(embeds.shape[1], tokens, image_pad_mask).to(device)
    negative_img_mask = (
        build_img_mask(negative_embeds.shape[1], tokens, negative_image_pad_mask).to(device)
        if use_cfg
        else None
    )

    print(f"  * output size: {width}x{height}")
    print("2. Denoise")
    denoise_start = time.perf_counter()
    print("  * loading transformer ...")
    print(f"    variant: {variant}")
    print(f"    quantization: {denoiser_quant_method or 'none'}")
    print(f"    steps: {steps}")
    print(f"    cfg: {guidance_scale}")
    print(f"    stream denoiser: {'on' if stream_denoiser else 'off'}")
    torch.cuda.empty_cache()
    transformer = load_qwen2_denoiser(
        model_root / "denoiser",
        denoiser_quant_method,
        variant=variant,
        offloading=stream_denoiser,
        stream_denoiser=stream_denoiser,
    )
    if loras:
        print("  * applying loras ...")
        print(f"    injected lora modules: {apply_loras(transformer, loras)}")
    if stream_denoiser:
        if enable_streaming(transformer):
            pin_quantized_buffers(transformer)
        move_unstreamed_modules_to(transformer, device)
    else:
        transformer = transformer.to(device)
    torch.cuda.synchronize(device)

    scheduler = _build_scheduler(variant)
    mu = calculate_shift(tokens)
    scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)
    scheduler.set_begin_index(0)

    num_blocks = len(transformer.transformer_blocks)
    cache_enabled = kv_cache and transformer.config.causal_condition
    model_dtype = next(transformer.parameters()).dtype

    latents = make_noise(seed, height, width, device=device, dtype=torch.float32)
    cond_cache = QwenImage21KVCache(num_blocks) if cache_enabled else None
    neg_cache = QwenImage21KVCache(num_blocks) if cache_enabled and use_cfg else None

    with torch.inference_mode():
        for index, t in enumerate(scheduler.timesteps):
            print(f"  * {index + 1}/{steps} steps", flush=True)
            step_start = time.perf_counter()
            kv_mode = "extract" if (cache_enabled and index == 0) else ("cached" if cache_enabled else None)
            timestep = t.expand(latents.shape[0]).to(latents.dtype) / 1000
            latent_input = latents.to(model_dtype)
            if ref_latents is not None:
                latent_input = torch.cat([ref_latents.to(model_dtype), latent_input], dim=1)
            noise_pred = transformer(
                hidden_states=latent_input,
                timestep=timestep,
                encoder_hidden_states=embeds.to(model_dtype),
                encoder_hidden_states_mask=text_mask,
                img_shapes=img_shapes,
                img_mask=img_mask,
                kv_cache=cond_cache,
                kv_cache_mode=kv_mode,
                return_dict=False,
            )[0]
            noise_pred = noise_pred[:, -latents.size(1):]

            if use_cfg:
                neg_noise_pred = transformer(
                    hidden_states=latent_input,
                    timestep=timestep,
                    encoder_hidden_states=negative_embeds.to(model_dtype),
                    encoder_hidden_states_mask=negative_text_mask,
                    img_shapes=img_shapes,
                    img_mask=negative_img_mask,
                    kv_cache=neg_cache,
                    kv_cache_mode=kv_mode,
                    return_dict=False,
                )[0]
                neg_noise_pred = neg_noise_pred[:, -latents.size(1):]
                noise_pred = neg_noise_pred + guidance_scale * (noise_pred - neg_noise_pred)

            latents = scheduler.step(noise_pred.float(), t, latents, return_dict=False)[0]
            if not torch.isfinite(latents).all():
                raise RuntimeError(f"Latents are not finite at step {index + 1}.")
            torch.cuda.synchronize(device)
            print(f"    * done in {time.perf_counter() - step_start:.3f}s", flush=True)

    latents = latents.float().cpu()

    torch.cuda.synchronize(device)
    print("  * clearing transformer ...")
    del transformer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  * done in {time.perf_counter() - denoise_start:.3f}s")

    print("3. VAE")
    vae_start = time.perf_counter()
    print("  * loading vae ...")
    vae = load_qwen2_vae(model_root / "vae", dtype=_vae_dtype(vae_dtype)).to(device)
    torch.cuda.synchronize(device)
    output_folder = Path(output_dir) if output_dir else resolve_outputs_dir(root)
    output_folder.mkdir(parents=True, exist_ok=True)

    from PIL.PngImagePlugin import PngInfo

    metadata = {
        "seed": seed,
        "steps": steps,
        "width": width,
        "height": height,
        "variant": variant,
        "guidance_scale": guidance_scale,
        "text_encoder_quant": text_quant_method or "none",
        "denoiser_quant": denoiser_quant_method or "none",
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "vae_dtype": vae_dtype,
        "reference_images": [str(image) for image in images] if images else [],
        "ref_size": ref_size,
        "loras": list(loras) if loras else [],
        "lora_strengths": list(loras.values()) if loras else [],
    }
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    image = decode_latents(
        vae,
        latents.to(device),
        height,
        width,
        dtype=_vae_dtype(vae_dtype),
    )
    if not rgba:
        image = image.convert("RGB")
    output_path = output_folder / f"{timestamp}.png"
    pnginfo = PngInfo()
    pnginfo.add_text("qwen_2", json.dumps(metadata))
    image.save(output_path, pnginfo=pnginfo)
    print(f"Saved {output_path}")
    torch.cuda.synchronize(device)
    print(f"  * done in {time.perf_counter() - vae_start:.3f}s")
    return image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate images with Qwen Image 2.1.")
    parser.add_argument("--root", required=True, help="Root folder that contains qwen_2/model.")
    parser.add_argument("--prompt", required=True, help="Prompt text.")
    parser.add_argument("--negative-prompt", default="", help="Negative prompt, used when --guidance-scale > 1.")
    parser.add_argument("--width", type=int, default=None, help="Output width (multiple of 32). Defaults to the last reference width, else 1024.")
    parser.add_argument("--height", type=int, default=None, help="Output height (multiple of 32). Defaults to the last reference height, else 1024.")
    parser.add_argument("--images", help='Optional comma-separated reference image paths, for example "a.png, b.png".')
    parser.add_argument("--ref-size", type=int, default=DEFAULT_REF_SIZE, help="Reference area side length (multiple of 32). Default: 512.")
    parser.add_argument("--loras", help='Optional comma-separated LoRA entries, for example "a.safetensors:0.8, b.safetensors".')
    parser.add_argument("--steps", type=int, default=None, help="Denoising steps. Defaults to 8 (turbo) or 40 (base).")
    parser.add_argument("--guidance-scale", type=float, default=1.0, help="CFG scale. Turbo uses 1.0.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Noise seed.")
    parser.add_argument("--variant", choices=VARIANTS, default="turbo", help="Checkpoint variant.")
    parser.add_argument("--text-quant-method", default=DEFAULT_TEXT_QUANT_METHOD, help='Text encoder quant method, or fp16/bf16/fp32 for the unquantized checkpoint.')
    parser.add_argument("--denoiser-quant-method", default=DEFAULT_DENOISER_QUANT_METHOD, help='Denoiser quant method, or fp16/bf16/fp32 for the unquantized checkpoint.')
    parser.add_argument("--stream-denoiser", action="store_true", help="Stream the denoiser weights from host memory.")
    parser.add_argument("--no-kv-cache", action="store_true", help="Disable the prefix KV cache.")
    parser.add_argument("--stream-text-encoder", action="store_true", help="Stream the text-encoder weights from host memory.")
    parser.add_argument("--vae-dtype", choices=_VAE_DTYPES, default="fp16", help="VAE dtype.")
    parser.add_argument("--rgb", action="store_true", help="Flatten alpha over white. Default keeps RGBA.")
    parser.add_argument("--output", default=None, help="Output folder. Defaults to <root>/qwen_2/outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_image(
        args.root,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        variant=args.variant,
        text_quant_method=args.text_quant_method,
        denoiser_quant_method=args.denoiser_quant_method,
        stream_denoiser=args.stream_denoiser,
        kv_cache=not args.no_kv_cache,
        stream_text_encoder=args.stream_text_encoder,
        vae_dtype=args.vae_dtype,
        rgba=not args.rgb,
        images=_split_image_paths(args.images) if args.images else None,
        ref_size=args.ref_size,
        loras=_parse_loras_arg(args.loras),
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
