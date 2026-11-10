#!/usr/bin/env python
"""
Flux 3 Klein training samples. Renders images for every LoRA step checkpoint of a project.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from flux2.sample import _list_step_checkpoints, _parse_sample_indices, _resolution_from_ratio
from flux3.config import resolve_model_root

DEFAULT_SAMPLE_SEED = 19930625
DEFAULT_STEPS = 16
_LATENT_CHANNELS = 128
_PATCH_PIXELS = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Flux 3 training samples for saved LoRA checkpoints.")
    parser.add_argument("--root", type=Path, required=True, help="Root folder that contains flux3/model.")
    parser.add_argument("--project", type=Path, required=True, help="Training project directory.")
    parser.add_argument("--prompt", type=str, help="Prompt that replaces the dataset captions.")
    parser.add_argument("--width", type=int, default=512, help="Prompt-mode sample width.")
    parser.add_argument("--height", type=int, default=512, help="Prompt-mode sample height.")
    parser.add_argument(
        "--target-res",
        dest="target_res",
        type=int,
        default=512,
        help="Short side of the output. Combined with each sample ratio to set the output size.",
    )
    parser.add_argument("--ref-res", dest="ref_res", type=int, default=512, help="Resolution of the references.")
    parser.add_argument(
        "--target-upscale",
        dest="target_upscale",
        action="store_true",
        help="Keep target images below target_res.",
    )
    parser.add_argument(
        "--ref-upscale",
        dest="ref_upscale",
        action="store_true",
        help="Keep reference images below ref_res.",
    )
    parser.add_argument("--start", type=int, default=1, help="First step checkpoint number to render.")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help=f"Denoise steps. Default: {DEFAULT_STEPS}.")
    parser.add_argument(
        "--distill",
        action="store_true",
        help="Use the distill denoiser instead of the base denoiser. It uses no guidance.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        help="CFG scale for the base denoiser. Default: 3.5. 1.0 turns off the negative pass.",
    )
    parser.add_argument("--sample-seed", "--sample_seed", dest="sample_seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--sample-indices", dest="sample_indices", type=str, default=None)
    parser.add_argument("--trigger", type=str, help="Required for captionless datasets.")
    parser.add_argument("--text-quant-method", default="sym-high-mini", help='"none" loads the fp16 weights.')
    parser.add_argument("--denoiser-quant-method", default="sym-med-nano", help='"none" loads the fp16 weights.')
    parser.add_argument("--max-length", type=int, default=512, help="Maximum text tokens.")
    parser.add_argument(
        "--offload",
        action="store_true",
        help="Keep denoiser weights on the CPU and stream blocks to the GPU one at a time.",
    )
    return parser.parse_args()


def _read_lora_metadata(checkpoints: list[tuple[int, Path]]) -> tuple[int, int]:
    from safetensors import safe_open

    for _step_number, checkpoint_path in checkpoints:
        try:
            with safe_open(str(checkpoint_path), framework="pt") as handle:
                metadata = handle.metadata()
        except Exception:
            continue
        if metadata is not None:
            return int(metadata["rank"]), int(metadata["alpha"])
    raise RuntimeError("Could not read safetensors metadata from any checkpoint.")


def main() -> None:
    args = parse_args()

    normalized_prompt = None if args.prompt is None else args.prompt.strip()
    if normalized_prompt == "":
        raise ValueError("--prompt must not be empty.")
    if normalized_prompt is not None and args.trigger is not None:
        raise ValueError("--trigger is only used with captionless datasets.")

    project_path = args.project.expanduser().resolve()
    checkpoints_dir = project_path / "models"
    if not checkpoints_dir.is_dir():
        raise FileNotFoundError(f"Expected models directory at {checkpoints_dir}")

    checkpoints = [
        (step_number, checkpoint_path)
        for step_number, checkpoint_path in _list_step_checkpoints(checkpoints_dir)
        if step_number >= args.start
    ]
    if not checkpoints:
        raise FileNotFoundError(f"No step checkpoints found at or after step {args.start} in {checkpoints_dir}")
    sample_dir = project_path / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    single_sample = args.sample_indices is None

    def output_name(step_number: int, sample_index: int, width: int, height: int, duplicate: int = 1) -> Path:
        suffix = "" if duplicate == 1 else f"_{duplicate}"
        ref = f"_ref{args.ref_res}"
        if single_sample:
            return sample_dir / f"{step_number:06d}_{width}x{height}{ref}{suffix}.png"
        return sample_dir / f"{step_number:06d}_{sample_index}_{width}x{height}{ref}{suffix}.png"

    def next_output_path(step_number: int, sample_index: int, width: int, height: int) -> Path:
        duplicate = 1
        while output_name(step_number, sample_index, width, height, duplicate).is_file():
            duplicate += 1
        return output_name(step_number, sample_index, width, height, duplicate)

    import torch

    from flux2.lora.loader import load_checkpoint as load_lora_checkpoint
    from flux2.lora.model import inject_trainable_lora_modules
    from flux2.gen import _on_device
    from flux2.train.images import normalize_image_latent_resolution
    from flux3.denoiser.loader import (
        load_flux3_denoiser,
        move_resident_layers_to,
        pin_module_parameters,
    )
    from flux3.gen import _load_vae, _resolve_settings, denoise
    from flux3.inputs import build_sigmas, decode_tokens, encode_prompt, prepare_image_ids
    from flux3.lora.config import FLUX3_LORA_TARGETS
    from flux3.train.dataset import Flux3Dataset
    from utils.seed import image_seed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is not available in this environment.")

    model_root = resolve_model_root(args.root)
    text_quant = None if args.text_quant_method == "none" else args.text_quant_method
    denoiser_quant = None if args.denoiser_quant_method == "none" else args.denoiser_quant_method
    variant = "distill" if args.distill else "base"
    steps, guidance_scale = _resolve_settings(not args.distill, args.steps, args.guidance_scale)
    use_cfg = not args.distill and guidance_scale != 1.0

    print("1. Load and encode dataset ...")
    dataset_load_start = time.perf_counter()
    parsed_indices = _parse_sample_indices(args.sample_indices)
    trigger = normalized_prompt if normalized_prompt is not None else args.trigger
    dataset = Flux3Dataset(
        model_root,
        text_quant,
        project_path / "images",
        trigger if trigger is not None else "",
        args.target_res,
        args.ref_res,
        indices=None if parsed_indices is None else [index - 1 for index in parsed_indices],
        load_captions=normalized_prompt is None,
        load_target_images=False,
        load_target_ratios=True,
        target_upscale=args.target_upscale,
        ref_upscale=args.ref_upscale,
        max_length=args.max_length,
    )

    dataset_size = len(dataset.target_ratios)
    if dataset_size == 0:
        print("  * no samples loaded")
        return
    if trigger is None and any(embedding is None for embedding in dataset.text_embeddings):
        raise ValueError("The dataset has samples without a caption. Pass --trigger or --prompt.")

    sample_indices = parsed_indices if parsed_indices is not None else list(range(1, dataset_size + 1))
    selected_index_by_sample = {sample_index: selected for selected, sample_index in enumerate(sample_indices)}

    sample_resolutions: dict[int, tuple[int, int]] = {}
    for sample_index in sample_indices:
        ratio = dataset.target_ratios[selected_index_by_sample[sample_index]]
        if ratio is None:
            continue
        if normalized_prompt is not None and args.sample_indices is None:
            width, height = args.width, args.height
        else:
            width, height = _resolution_from_ratio(ratio=ratio, short_side=args.target_res)
        sample_resolutions[sample_index] = normalize_image_latent_resolution(
            width, height, vae_scale_factor=_PATCH_PIXELS // 2
        )
    sample_indices = list(sample_resolutions.keys())

    print(f"  * dataset mode: {'prompt' if normalized_prompt is not None else 'captioned'}")
    print(f"  * images: {dataset_size}")
    print(f"  * selected samples: {', '.join(str(sample_index) for sample_index in sample_indices)}")
    if trigger is not None:
        print(f"  * trigger: {trigger!r}")
    print(
        "  * sample resolutions: "
        + ", ".join(f"{index}={w}x{h}" for index, (w, h) in sample_resolutions.items())
    )
    print(f"  * done in {time.perf_counter() - dataset_load_start:.3f}s")
    if not sample_indices:
        print("  * no samples requested")
        return
    print(f"  * steps to render: {len(checkpoints)}")

    negative = None
    if use_cfg:
        from transformers import Qwen2TokenizerFast

        from flux3.text_encoder.loader import load_flux3_text_encoder

        print("2. Encode the negative prompt ...")
        negative_start = time.perf_counter()
        tokenizer = Qwen2TokenizerFast.from_pretrained(str(model_root / "text_encoder"))
        text_encoder = load_flux3_text_encoder(model_root / "text_encoder", text_quant).to(device)
        negative_embeds, negative_ids = encode_prompt(
            tokenizer, text_encoder, "", device=device, max_length=args.max_length
        )
        negative = (negative_embeds.to("cpu", torch.float16), negative_ids.to("cpu"))
        del tokenizer, text_encoder
        torch.cuda.empty_cache()
        print(f"  * done in {time.perf_counter() - negative_start:.3f}s")
    else:
        print("2. Encode the negative prompt ... skipped")

    lora_rank, lora_alpha = _read_lora_metadata(checkpoints)
    print(f"  * lora rank/alpha from checkpoint: r={lora_rank}, a={lora_alpha}")

    print("3. Load denoiser ...")
    denoiser_load_start = time.perf_counter()
    print(f"  * variant: {variant}")
    print(f"  * steps: {steps}")
    print(f"  * guidance scale: {guidance_scale if use_cfg else 'off'}")
    denoiser = load_flux3_denoiser(
        model_root / "transformer",
        denoiser_quant,
        variant,
        offloading=args.offload,
    )
    injected = inject_trainable_lora_modules(
        denoiser,
        target_linear_names=FLUX3_LORA_TARGETS,
        rank=lora_rank,
        alpha=lora_alpha,
    )
    denoiser.eval()
    for parameter in denoiser.parameters():
        parameter.requires_grad = False
    if args.offload:
        pinned = pin_module_parameters(denoiser)
        print(f"  * pinned {pinned} cpu tensors for faster block swaps")
        move_resident_layers_to(denoiser, device)
        denoiser.enable_block_offload(device)
    else:
        denoiser = denoiser.to(device)
    print(f"  * injected lora modules: {len(injected)}")
    print(f"  * done in {time.perf_counter() - denoiser_load_start:.3f}s")

    vae = _load_vae(model_root / "vae")
    sigmas = build_sigmas(steps).to(device)

    def denoise_samples(step_number: int) -> list[tuple[int, torch.Tensor]]:
        rendered = []
        for sample_index in sample_indices:
            width, height = sample_resolutions[sample_index]
            if output_name(step_number, sample_index, width, height).is_file():
                print(f"    * skipped sample {sample_index}: {output_name(step_number, sample_index, width, height)}")
                continue

            print(f"    * denoise sample {sample_index} at {width}x{height} ...")
            selected = selected_index_by_sample[sample_index]
            sample_name = dataset.sample_names[selected] or f"index_{sample_index}"

            text_item = None if normalized_prompt is not None else dataset.text_embeddings[selected]
            if text_item is None:
                text_item = dataset.trigger_embedding
            embeds = text_item[0].unsqueeze(0).to(device=device, dtype=torch.float16)
            txt_ids = text_item[1].to(device)

            cond = None
            cond_ids = None
            ref_latents = dataset.ref_latents[selected]
            if ref_latents is not None:
                cond = ref_latents[0].unsqueeze(0).to(device=device, dtype=torch.float16)
                cond_ids = ref_latents[1].to(device)

            grid_h = height // _PATCH_PIXELS
            grid_w = width // _PATCH_PIXELS
            generator = torch.Generator("cpu").manual_seed(image_seed(args.sample_seed, sample_name))
            latents = torch.randn((grid_h * grid_w, _LATENT_CHANNELS), generator=generator, dtype=torch.float32)
            latents = latents[None].to(device)
            img_ids = prepare_image_ids(grid_h, grid_w).to(device)

            latents = denoise(
                denoiser,
                latents,
                sigmas,
                embeds=embeds,
                txt_ids=txt_ids,
                img_ids=img_ids,
                cond=cond,
                cond_ids=cond_ids,
                neg=negative if use_cfg else None,
                guidance_scale=guidance_scale,
            )
            rendered.append((sample_index, latents.cpu()))
            del embeds, txt_ids, cond, cond_ids, latents, img_ids
        return rendered

    def decode_samples(step_number: int, rendered: list[tuple[int, torch.Tensor]]) -> None:
        with _on_device(vae, device, dtype=torch.float16):
            for sample_index, latents in rendered:
                width, height = sample_resolutions[sample_index]
                output_path = next_output_path(step_number, sample_index, width, height)
                image = decode_tokens(vae, latents.to(device), height // _PATCH_PIXELS, width // _PATCH_PIXELS)
                image.save(output_path)
                print(f"    * saved {output_path}")

    print("4. Render checkpoint samples ...")
    render_start = time.perf_counter()
    print(f"  * checkpoints: {len(checkpoints)}")
    print(f"  * sample output dir: {sample_dir}")
    for step_number, checkpoint_path in checkpoints:
        print(f"  * step {step_number}: {checkpoint_path}")
        step_start = time.perf_counter()
        load_lora_checkpoint(denoiser, checkpoint_path)
        rendered = denoise_samples(step_number)
        if not args.offload:
            denoiser.to("cpu")
        torch.cuda.empty_cache()
        print("    * decoding samples ...")
        decode_samples(step_number, rendered)
        if not args.offload:
            denoiser.to(device)
        print(f"    * step sample time: {time.perf_counter() - step_start:.3f}s")

    del denoiser
    del vae
    torch.cuda.empty_cache()
    print(f"  * sampling done in {time.perf_counter() - render_start:.3f}s")


if __name__ == "__main__":
    main()
