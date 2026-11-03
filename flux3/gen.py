#!/usr/bin/env python
"""
Flux 3 Klein generation entrypoint.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

from flux2.gen import _module_device, _on_device, _parse_loras_arg, _split_image_paths
from flux3.config import resolve_model_root

if TYPE_CHECKING:
    from diffusers import AutoencoderKLFlux2
    from PIL import Image
    from transformers import Qwen2TokenizerFast

    from flux2.models.text_encoder.model import Qwen3Model
    from flux3.models.denoiser.transformer import Flux3Transformer2DModel

DEFAULT_DISTILL_STEPS = 16
DEFAULT_BASE_STEPS = 50
DEFAULT_BASE_GUIDANCE_SCALE = 3.5
DEFAULT_SEED = 19930625
_LATENT_CHANNELS = 128
_PATCH_PIXELS = 16


def _resolve_settings(
    base: bool,
    num_inference_steps: int | None,
    guidance_scale: float | None,
) -> tuple[int, float]:
    """Return the steps and the guidance scale that a run uses.

    Args:
        base: True for the base variant.
        num_inference_steps: The requested steps, or None for the default.
        guidance_scale: The requested guidance scale, or None for the default.

    Returns:
        A tuple of the steps and the guidance scale. The distill variant always
        returns 1.0, because it does not use guidance.
    """
    if base:
        steps = DEFAULT_BASE_STEPS if num_inference_steps is None else num_inference_steps
        scale = DEFAULT_BASE_GUIDANCE_SCALE if guidance_scale is None else guidance_scale
        return steps, scale
    if guidance_scale not in (None, 1.0):
        print(f"  * warning: the distill variant ignores guidance_scale={guidance_scale}")
    steps = DEFAULT_DISTILL_STEPS if num_inference_steps is None else num_inference_steps
    return steps, 1.0


def _load_vae(vae_path: str | Path):
    from diffusers import AutoencoderKLFlux2

    print("  * loading vae ...")
    return AutoencoderKLFlux2.from_pretrained(str(vae_path), local_files_only=True)


def _inject_loras(denoiser, loras: dict[str, float]) -> int:
    from flux2.lora import inject_trainable_lora_modules
    from flux3.lora.config import FLUX3_LORA_TARGETS
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
    injected = inject_trainable_lora_modules(
        denoiser,
        target_linear_names=FLUX3_LORA_TARGETS,
        rank=lora_rank,
        alpha=lora_alpha,
    )
    return len(injected)


@contextlib.contextmanager
def _denoiser_on_device(denoiser, device, *, offloading: bool):
    """Place a caller denoiser for one denoise pass. Restore it on exit."""
    from flux3.denoiser.loader import move_resident_layers_to

    original_device = _module_device(denoiser)
    try:
        if offloading:
            move_resident_layers_to(denoiser, device)
            denoiser.enable_block_offload(device)
        else:
            denoiser.to(device)
        yield denoiser
    finally:
        if offloading:
            denoiser.disable_block_offload()
            move_resident_layers_to(denoiser, original_device)
        else:
            denoiser.to(original_device)


def denoise(
    denoiser,
    latents,
    sigmas,
    *,
    embeds,
    txt_ids,
    img_ids,
    cond=None,
    cond_ids=None,
    neg=None,
    guidance_scale: float = 1.0,
):
    """Run the Euler denoise loop of Flux 3 Klein.

    Gen and sample use this function, so both use the same sigmas, Euler step,
    and CFG formula. The denoiser must be on the CUDA device, or must have
    block offload on. The function reads `flow_map` from the denoiser config.
    A flow-map denoiser gets the next sigma as `target_timestep`.

    Args:
        denoiser: The Flux 3 denoiser.
        latents: The noise tokens with shape `(1, h * w, 128)`, on the device.
        sigmas: The sigmas from `build_sigmas`, on the device. It has one more
            value than the number of steps.
        embeds: The text embeddings with shape `(1, S, 20480)`.
        txt_ids: The text position ids.
        img_ids: The position ids of the output tokens.
        cond: The reference tokens with shape `(1, C, 128)`, or None.
        cond_ids: The position ids of the reference tokens, or None.
        neg: A tuple of the negative embeddings and the negative text ids, or
            None. Pass it to turn on CFG.
        guidance_scale: The CFG scale. Used only when `neg` is not None.

    Returns:
        The denoised tokens in fp32 with the shape of `latents`.

    Raises:
        RuntimeError: If a step produces inf or NaN values.
    """
    import torch

    device = latents.device
    flow_map = bool(denoiser.config.flow_map)
    steps = len(sigmas) - 1
    embeds = embeds.to(device=device, dtype=torch.float16)
    txt_ids = txt_ids.to(device)
    if neg is not None:
        neg = (neg[0].to(device=device, dtype=torch.float16), neg[1].to(device))

    with torch.inference_mode():
        for step_index in range(steps):
            print(f"  * {step_index + 1}/{steps} steps", flush=True)
            step_start = time.perf_counter()
            sigma, sigma_next = sigmas[step_index], sigmas[step_index + 1]
            kwargs = {
                "img_ids": img_ids,
                "image_cond_hidden_states": cond,
                "image_cond_ids": cond_ids,
                "timestep": sigma.expand(1),
                "target_timestep": sigma_next.expand(1) if flow_map else None,
                "return_dict": False,
            }
            velocity = denoiser(latents.half(), encoder_hidden_states=embeds, txt_ids=txt_ids, **kwargs)[0]
            if neg is not None:
                velocity_uncond = denoiser(latents.half(), encoder_hidden_states=neg[0], txt_ids=neg[1], **kwargs)[0]
                velocity = velocity_uncond + guidance_scale * (velocity - velocity_uncond)
            latents = latents + (sigma_next - sigma) * velocity.float()
            if not torch.isfinite(latents).all():
                raise RuntimeError(
                    f"Flux 3 denoise produced inf or NaN values in fp16 at step {step_index + 1}/{steps}."
                )
            torch.cuda.synchronize(device)
            print(f"    * done in {time.perf_counter() - step_start:.3f}s", flush=True)
    return latents


def generate_image(
    root: str | Path,
    *,
    prompt: str = "A cat holding a sign that says hello world",
    image: str | Path | None = None,
    width: int = 512,
    height: int = 512,
    num_inference_steps: int | None = None,
    seed: int = DEFAULT_SEED,
    guidance_scale: float | None = None,
    base: bool = False,
    negative_prompt: str | None = "",
    ref_size: int = 512,
    text_quant_method: str | None = None,
    denoiser_quant_method: str | None = None,
    loras: dict[str, float] | None = None,
    max_length: int = 512,
    offloading: bool = False,
    tokenizer: Qwen2TokenizerFast | None = None,
    text_encoder: Qwen3Model | None = None,
    denoiser: Flux3Transformer2DModel | None = None,
    vae: AutoencoderKLFlux2 | None = None,
) -> list[Image.Image]:
    """Generate one image with Flux 3 Klein.

    Each model is loaded from disk unless the caller passes it in. Pass the
    models to reuse them across many calls and skip the disk load.

    The caller keeps passed models in host memory. This function moves each
    passed model to the CUDA device only while the model is in use. It moves
    the model back to its original device afterward. The VAE also casts to
    float16 while in use and returns to its original dtype. This function
    never deletes a passed model.

    Args:
        root: Root folder that contains flux3/model.
        prompt: Prompt text to encode.
        image: Comma-separated reference image paths. Defaults to None.
        width: Output width in pixels. Rounds down to a multiple of 16.
        height: Output height in pixels. Rounds down to a multiple of 16.
        num_inference_steps: Denoising steps. Defaults to 16 (distill) or 50
            (base).
        seed: Seed for the latent noise. With reference images, the seed is
            mixed with the file name of the first reference.
        guidance_scale: CFG scale. Used only with base. Defaults to 3.5.
        base: Use the base variant. When a denoiser is passed, this value must
            match the variant of that denoiser.
        negative_prompt: Negative prompt. Used only with base.
        ref_size: Short side of reference images in pixels.
        text_quant_method: Text encoder quantization. Ignored when
            text_encoder is passed.
        denoiser_quant_method: Denoiser quantization. Ignored when denoiser is
            passed.
        loras: Map of LoRA checkpoint path to strength. When a denoiser is
            passed, the function deep-copies it in host memory, injects the
            LoRA into the copy, and moves the copy to the CUDA device. The
            copy needs one extra denoiser-sized block of host memory. The
            passed denoiser is not changed.
        max_length: Maximum tokenizer sequence length.
        offloading: Stream denoiser blocks to the CUDA device one at a time.
        tokenizer: Pre-loaded tokenizer. Defaults to None (load from disk).
        text_encoder: Pre-loaded text encoder. Defaults to None (load from
            disk).
        denoiser: Pre-loaded denoiser in host memory. Defaults to None (load
            from disk).
        vae: Pre-loaded VAE. Defaults to None (load from disk).

    Returns:
        The generated images.

    Raises:
        RuntimeError: If CUDA is not available, if LoRA metadata has no rank
            or alpha, or if the denoise run produces inf or NaN values.
        ValueError: If a passed denoiser does not match `base`.
    """
    from flux3.denoiser.loader import (
        load_flux3_denoiser,
        move_resident_layers_to,
        pin_module_parameters,
    )
    from flux3.inputs import (
        build_sigmas,
        decode_tokens,
        encode_image_latent,
        encode_prompt,
        prepare_image_ids,
        resize_reference,
    )
    from flux3.text_encoder.loader import load_flux3_text_encoder
    from flux2.lora.loader import load_checkpoint as load_lora_checkpoint
    from utils.seed import image_seed

    import torch
    from transformers import Qwen2TokenizerFast

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is not available in this environment.")

    model_root = resolve_model_root(root)
    steps, cfg = _resolve_settings(base, num_inference_steps, guidance_scale)
    variant = "base" if base else "distill"
    use_cfg = base and cfg != 1.0

    if denoiser is not None and bool(denoiser.config.flow_map) == base:
        raise ValueError(f"The passed denoiser does not match base={base}.")

    reference_paths = _split_image_paths(image) if image else []
    effective_seed = seed
    if reference_paths:
        effective_seed = image_seed(seed, Path(reference_paths[0]).stem)

    owns_text_encoder = text_encoder is None
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
        text_encoder = load_flux3_text_encoder(model_root / "text_encoder", text_quant_method)
    else:
        print("  * using caller-provided text encoder")

    negative_embeds = None
    negative_txt_ids = None
    with _on_device(text_encoder, device, restore=not owns_text_encoder):
        print("  * text encoding ...")
        torch.cuda.synchronize(device)
        encode_start = time.perf_counter()
        embeds, txt_ids = encode_prompt(tokenizer, text_encoder, prompt, device=device, max_length=max_length)
        if use_cfg:
            negative_embeds, negative_txt_ids = encode_prompt(
                tokenizer, text_encoder, negative_prompt or "", device=device, max_length=max_length
            )
        torch.cuda.synchronize(device)
        print(f"    encode time: {time.perf_counter() - encode_start:.3f}s")
    print(f"    text tokens: {embeds.shape[1]}")

    print("  * clearing encoder ...")
    del tokenizer
    del text_encoder
    torch.cuda.empty_cache()
    print(f"  * done in {time.perf_counter() - text_start:.3f}s")

    print("2. Image Encoding")
    cond = None
    cond_ids = None
    if reference_paths:
        from PIL import Image

        image_start = time.perf_counter()
        if vae is None:
            vae = _load_vae(model_root / "vae")
        else:
            print("  * using caller-provided vae")

        tokens_list = []
        ids_list = []
        with _on_device(vae, device, dtype=torch.float16, restore=not owns_vae):
            for index, path in enumerate(reference_paths):
                reference = resize_reference(Image.open(path).convert("RGB"), ref_size)
                tokens, ref_h, ref_w = encode_image_latent(vae, reference, device)
                tokens_list.append(tokens)
                ids_list.append(prepare_image_ids(ref_h, ref_w, layer=-(1 + index)))
        cond = torch.cat(tokens_list, dim=0)[None].to(device=device, dtype=torch.float16)
        cond_ids = torch.cat(ids_list, dim=0).to(device)
        print(f"    reference tokens: {tuple(cond.shape)}")
        print(f"  * done in {time.perf_counter() - image_start:.3f}s")
    else:
        print("  * skipped")

    print("3. Prepare latent")
    grid_h = int(height) // _PATCH_PIXELS
    grid_w = int(width) // _PATCH_PIXELS
    if grid_h < 1 or grid_w < 1:
        raise ValueError(f"width and height must be at least {_PATCH_PIXELS}.")
    generator = torch.Generator("cpu").manual_seed(effective_seed)
    latents = torch.randn((grid_h * grid_w, _LATENT_CHANNELS), generator=generator, dtype=torch.float32)
    latents = latents[None].to(device)
    img_ids = prepare_image_ids(grid_h, grid_w).to(device)
    print(f"  * output size: {grid_w * _PATCH_PIXELS}x{grid_h * _PATCH_PIXELS}")

    print("4. Denoise")
    denoise_start = time.perf_counter()
    sigmas = build_sigmas(steps).to(device)
    owns_denoiser = denoiser is None or bool(loras)
    if denoiser is None:
        print("  * loading transformer ...")
        print(f"    variant: {variant}")
        print(f"    quantization: {denoiser_quant_method or 'none'}")
        print(f"    offloading: {offloading}")
        torch.cuda.empty_cache()
        transformer = load_flux3_denoiser(
            model_root / "transformer",
            denoiser_quant_method,
            variant,
            offloading=offloading,
        )
    else:
        print("  * using caller-provided transformer")
        transformer = denoiser
        if loras:
            print("  * copying transformer before lora injection ...")
            transformer = copy.deepcopy(denoiser)
    if loras:
        print("  * applying loras ...")
        print(f"    injected lora modules: {_inject_loras(transformer, loras)}")
    if owns_denoiser:
        if offloading:
            pinned = pin_module_parameters(transformer)
            print(f"    pinned {pinned} cpu tensors for faster block swaps")
            move_resident_layers_to(transformer, device)
            transformer.enable_block_offload(device)
        else:
            transformer = transformer.to(device)
    if loras:
        from flux2.lora import TrainableLoraLinear

        for lora_path, strength in loras.items():
            print(f"    {lora_path} [{strength:.2f}]")
            load_lora_checkpoint(transformer, lora_path)
            if strength != 1.0:
                scale = strength ** 0.5
                for module in transformer.modules():
                    if isinstance(module, TrainableLoraLinear):
                        module.lora_A.data.mul_(scale)
                        module.lora_B.data.mul_(scale)

    torch.cuda.synchronize(device)
    placement = (
        contextlib.nullcontext()
        if owns_denoiser
        else _denoiser_on_device(denoiser, device, offloading=offloading)
    )
    with placement:
        latents = denoise(
            transformer,
            latents,
            sigmas,
            embeds=embeds,
            txt_ids=txt_ids,
            img_ids=img_ids,
            cond=cond,
            cond_ids=cond_ids,
            neg=(negative_embeds, negative_txt_ids) if use_cfg else None,
            guidance_scale=cfg,
        )

    torch.cuda.synchronize(device)
    if owns_denoiser:
        print("  * clearing transformer ...")
        del transformer
    torch.cuda.empty_cache()
    print(f"  * done in {time.perf_counter() - denoise_start:.3f}s")
    if not torch.isfinite(latents).all():
        raise RuntimeError("Flux 3 denoise produced inf or NaN values in fp16.")

    print("5. VAE")
    vae_start = time.perf_counter()
    if vae is None:
        vae = _load_vae(model_root / "vae")
    with _on_device(vae, device, dtype=torch.float16, restore=not owns_vae):
        decoded = decode_tokens(vae, latents, grid_h, grid_w)
        torch.cuda.synchronize(device)
    print(f"  * done in {time.perf_counter() - vae_start:.3f}s")

    return [decoded]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flux 3 Klein generation.")
    parser.add_argument("--root", required=True, help="Root folder that contains flux3/model.")
    parser.add_argument(
        "--prompt",
        default="A cat holding a sign that says hello world",
        help="Prompt text to encode.",
    )
    parser.add_argument(
        "--images",
        help='Optional comma-separated reference image paths, for example "img1.png, img2.png".',
    )
    parser.add_argument("--width", type=int, default=512, help="Output width. Default: 512.")
    parser.add_argument("--height", type=int, default=512, help="Output height. Default: 512.")
    parser.add_argument(
        "--steps",
        type=int,
        help=f"Denoising steps. Default: {DEFAULT_DISTILL_STEPS} for distill, {DEFAULT_BASE_STEPS} with --base.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Noise seed. Default: {DEFAULT_SEED}.")
    parser.add_argument(
        "--base",
        action="store_true",
        help="Use the base variant with CFG instead of the distill variant.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        help=f"CFG scale for --base. Default: {DEFAULT_BASE_GUIDANCE_SCALE}. The distill variant ignores it.",
    )
    parser.add_argument("--negative-prompt", default="", help="Negative prompt for --base. Default: empty.")
    parser.add_argument(
        "--ref-size",
        type=int,
        default=512,
        help="Short side of reference images before encoding. Default: 512.",
    )
    parser.add_argument(
        "--text-quant-method",
        default="sym-high-mini",
        help='Text encoder quantization method. "none" loads the fp16 weights.',
    )
    parser.add_argument(
        "--denoiser-quant-method",
        default="sym-med-nano",
        help='Denoiser quantization method. "none" loads the fp16 weights.',
    )
    parser.add_argument(
        "--loras",
        help='Optional comma-separated LoRA list in the form "path:strength, path2:strength2".',
    )
    parser.add_argument("--max-length", type=int, default=512, help="Maximum tokenizer sequence length.")
    parser.add_argument(
        "--offload",
        action="store_true",
        help="Keep denoiser weights on the CPU and stream blocks to the GPU one at a time.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from PIL.PngImagePlugin import PngInfo

    steps, guidance_scale = _resolve_settings(args.base, args.steps, args.guidance_scale)
    text_quant = None if args.text_quant_method == "none" else args.text_quant_method
    denoiser_quant = None if args.denoiser_quant_method == "none" else args.denoiser_quant_method
    loras = _parse_loras_arg(args.loras)

    images = generate_image(
        args.root,
        prompt=args.prompt,
        image=args.images,
        width=args.width,
        height=args.height,
        num_inference_steps=steps,
        seed=args.seed,
        guidance_scale=guidance_scale,
        base=args.base,
        negative_prompt=args.negative_prompt,
        ref_size=args.ref_size,
        text_quant_method=text_quant,
        denoiser_quant_method=denoiser_quant,
        loras=loras,
        max_length=args.max_length,
        offloading=args.offload,
    )

    output_dir = Path(args.root).expanduser().resolve() / "flux3" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}.png"
    metadata = {
        "seed": args.seed,
        "steps": steps,
        "variant": "base" if args.base else "distill",
        "guidance_scale": guidance_scale,
        "text_encoder_quant": text_quant or "half_precision",
        "denoiser_quant": denoiser_quant or "half_precision",
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt if args.base else "",
        "references": _split_image_paths(args.images) if args.images else [],
        "loras": loras or {},
    }
    pnginfo = PngInfo()
    pnginfo.add_text("flux3", json.dumps(metadata))
    images[0].save(output_path, pnginfo=pnginfo)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
