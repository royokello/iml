from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

DEFAULT_SAMPLE_SEED = 19930625
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return parsed


def _parse_sample_indices(value: str | None) -> list[int] | None:
    if value is None:
        return None
    return [int(index.strip()) for index in value.split(",") if index.strip()]


def _resolve_sample_indices(
    *,
    dataset_size: int,
    samples: int,
    sample_indices: str | None,
) -> list[int]:
    parsed_indices = _parse_sample_indices(sample_indices)
    if parsed_indices is not None:
        resolved_indices = parsed_indices
    elif samples == 0:
        resolved_indices = []
    elif samples < 0:
        raise ValueError("--samples must be zero or greater.")
    elif samples == 1:
        resolved_indices = [1]
    else:
        resolved_indices = [
            1 + round(index * (dataset_size - 1) / (samples - 1))
            for index in range(samples)
        ]

    invalid_indices = [
        sample_index for sample_index in resolved_indices if sample_index < 1 or sample_index > dataset_size
    ]
    if invalid_indices:
        invalid_text = ", ".join(str(sample_index) for sample_index in invalid_indices)
        raise ValueError(f"Sample indices out of range for {dataset_size} images: {invalid_text}")

    return resolved_indices


def _list_step_checkpoints(checkpoints_dir: Path) -> list[tuple[int, Path]]:
    checkpoints: list[tuple[int, Path]] = []
    for checkpoint_path in checkpoints_dir.glob("*.safetensors"):
        if not checkpoint_path.stem.isdigit():
            continue
        try:
            step_number = int(checkpoint_path.stem)
        except ValueError:
            continue
        checkpoints.append((step_number, checkpoint_path))

    checkpoints.sort(key=lambda item: item[0])
    if not checkpoints:
        raise FileNotFoundError(f"No step checkpoints found in {checkpoints_dir}")
    return checkpoints


def _is_supported_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def _list_image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {directory}")
    image_paths = sorted(
        (path for path in directory.iterdir() if _is_supported_image(path)),
        key=lambda path: path.name,
    )
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in dataset: {directory}")
    return image_paths


def _resolution_from_ratio(*, ratio: float, short_side: int) -> tuple[int, int]:
    if ratio <= 0:
        raise ValueError(f"Image ratio must be positive: {ratio}")
    if ratio >= 1:
        return max(1, round(short_side * ratio)), short_side
    return short_side, max(1, round(short_side / ratio))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Flux 2 training samples for saved LoRA checkpoints.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--version",
        choices=("4b", "9b"),
        required=True,
        help="Model family to load.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        required=True,
        help="Training project directory.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Prompt to render without loading a dataset.",
    )
    parser.add_argument(
        "--width",
        type=_positive_int,
        default=512,
        help="Prompt-only sample width.",
    )
    parser.add_argument(
        "--height",
        type=_positive_int,
        default=512,
        help="Prompt-only sample height.",
    )
    parser.add_argument(
        "--force-size",
        "--force_size",
        dest="force_size",
        type=_positive_int,
        help="Dataset-backed override for the shortest image side. Keeps each sample's aspect ratio.",
    )
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument(
        "--start",
        type=_positive_int,
        default=1,
        help="First step checkpoint number to render samples for.",
    )
    parser.add_argument(
        "--steps",
        type=_positive_int,
        default=16,
        help="Base-model denoising steps per sample. Distilled mode uses its default step count.",
    )
    parser.add_argument(
        "--distill",
        action="store_true",
        help="Use the distilled denoiser checkpoint instead of the base denoiser.",
    )
    parser.add_argument("--sample-seed", "--sample_seed", dest="sample_seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--sample-indices", dest="sample_indices", type=str, default=None)
    parser.add_argument("--text-quant-method", default="sym-high")
    parser.add_argument("--denoiser-quant-method", default="sym-med")
    parser.add_argument(
        "--trigger",
        type=str,
        help="Required for captionless datasets. Encoded once and kept on GPU for all samples.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_path = args.project.expanduser().resolve()
    checkpoints_dir = project_path / "models"
    if not checkpoints_dir.is_dir():
        raise FileNotFoundError(f"Expected models directory at {checkpoints_dir}")

    checkpoints = _list_step_checkpoints(checkpoints_dir)
    checkpoints = [
        (step_number, checkpoint_path)
        for step_number, checkpoint_path in checkpoints
        if step_number >= args.start
    ]
    if not checkpoints:
        raise FileNotFoundError(f"No step checkpoints found at or after step {args.start} in {checkpoints_dir}")
    sample_dir = project_path / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    def next_sample_output_path(step_number: int, sample_index: int) -> Path:
        output_path = sample_dir / f"{step_number:06d}_{sample_index}.png"
        if not output_path.is_file():
            return output_path

        duplicate_index = 2
        while True:
            duplicate_path = sample_dir / f"{step_number:06d}_{sample_index}_{duplicate_index}.png"
            if not duplicate_path.is_file():
                return duplicate_path
            duplicate_index += 1

    model_root = args.root / f"flux2_{args.version}" / "model"
    normalized_prompt = None if args.prompt is None else args.prompt.strip()
    if normalized_prompt == "":
        raise ValueError("--prompt must not be empty.")
    if normalized_prompt is not None and args.trigger is not None:
        raise ValueError("--trigger is only used with captionless datasets.")
    if normalized_prompt is not None and args.force_size is not None and args.sample_indices is None:
        raise ValueError("--force-size requires --sample-indices when used with --prompt.")

    sample_resolutions: dict[int, tuple[int, int]] = {}

    print("1. Load and encode dataset ...")
    from flux2.train.dataset import Flux2Dataset

    dataset_dir = project_path / "images"
    target_images = _list_image_files(dataset_dir / "base")

    sample_indices = _resolve_sample_indices(
        dataset_size=len(target_images),
        samples=1 if normalized_prompt is not None else args.samples,
        sample_indices=args.sample_indices,
    )
    if not sample_indices:
        print("  * no samples requested")
        return

    dataset_load_start = time.perf_counter()
    selected_dataset_indices = [sample_index - 1 for sample_index in sample_indices]
    selected_dataset_index_by_sample = {
        sample_index: selected_index for selected_index, sample_index in enumerate(sample_indices)
    }
    normalized_trigger = normalized_prompt if normalized_prompt is not None else args.trigger
    cached_dataset = Flux2Dataset(
        model_root,
        args.text_quant_method,
        dataset_dir,
        normalized_trigger,
        indices=selected_dataset_indices,
        ratios=True,
        load_target_image=False,
    )

    if normalized_prompt is not None:
        dataset_mode = "prompt"
    else:
        dataset_mode = "captioned"

    if len(cached_dataset.ratios) != len(sample_indices):
        raise ValueError(
            "Collected image ratio count does not match selected sample count: "
            f"{len(cached_dataset.ratios)} != {len(sample_indices)}"
        )

    short_side = args.force_size or 512
    for sample_index in sample_indices:
        selected_index = selected_dataset_index_by_sample[sample_index]
        if normalized_prompt is not None and args.sample_indices is None:
            sample_width = args.width
            sample_height = args.height
        else:
            sample_width, sample_height = _resolution_from_ratio(
                ratio=cached_dataset.ratios[selected_index],
                short_side=short_side,
            )
        sample_resolutions[sample_index] = (sample_width, sample_height)

    print(f"  * dataset mode: {dataset_mode}")
    print(f"  * images: {len(target_images)}")
    print(f"  * selected samples: {', '.join(str(sample_index) for sample_index in sample_indices)}")
    if normalized_trigger is not None:
        print(f"  * trigger: {normalized_trigger!r}")
    if args.force_size is not None:
        print(f"  * forced short side: {args.force_size}")
    if normalized_prompt is not None:
        print(f"  * requested image size: {args.width}x{args.height}")
    print(f"  * short side for ratio samples: {short_side}")
    print(f"  * done in {time.perf_counter() - dataset_load_start:.3f}s")

    if not sample_indices:
        print("  * no samples requested")
        return

    checkpoints_to_render = [
        (step_number, checkpoint_path, sample_indices)
        for step_number, checkpoint_path in checkpoints
    ]
    print(f"  * steps to render: {len(checkpoints_to_render)}")

    from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    import torch

    from flux2.denoiser.loader import _load_flux2_denoiser as load_flux2_denoiser
    from flux2.lora import inject_trainable_lora_modules
    from flux2.lora.config import FLUX2_LORA_ALPHA, FLUX2_LORA_RANK, FLUX2_LORA_TARGETS
    from flux2.lora.loader import load_checkpoint as load_lora_checkpoint
    from flux2.train.images import normalize_image_latent_resolution
    from flux2.gen import (
        DEFAULT_DISTILLED_STEPS,
        _compute_empirical_mu,
        _load_transformer_in_channels,
        _load_vae_scale_factor,
        _pack_latents,
        _prepare_latent_ids,
        _retrieve_timesteps,
        _unpack_latents_with_ids,
        _unpatchify_latents,
    )

    device = torch.device("cuda")
    vae_path = model_root / "vae"
    scheduler_path = model_root / "scheduler"
    transformer_path = model_root / "transformer"
    transformer_variant = "distill" if args.distill else "base"
    sample_steps = DEFAULT_DISTILLED_STEPS if args.distill else args.steps

    print("2. Select cached sample conditioning ...")
    text_encoding_start = time.perf_counter()
    if dataset_mode == "prompt":
        print("  * using prompt encoding from dataset")
    else:
        print("  * using caption encodings from dataset")
    torch.cuda.empty_cache()
    print(f"  * done in {time.perf_counter() - text_encoding_start:.3f}s")

    print("3. Prepare VAE for sample resolutions and decode ...")
    image_latent_prep_start = time.perf_counter()
    vae_scale_factor = _load_vae_scale_factor(vae_path)
    from diffusers import AutoencoderKLFlux2

    vae = AutoencoderKLFlux2.from_pretrained(str(vae_path), local_files_only=True)
    image_processor = Flux2ImageProcessor(vae_scale_factor=vae_scale_factor * 2)

    for sample_index, (sample_width, sample_height) in list(sample_resolutions.items()):
        sample_resolutions[sample_index] = normalize_image_latent_resolution(
            sample_width,
            sample_height,
            vae_scale_factor=vae_scale_factor,
        )
    sample_resolution_text = ", ".join(
        f"{sample_index}={width}x{height}" for sample_index, (width, height) in sample_resolutions.items()
    )
    print(f"  * sample resolutions: {sample_resolution_text}")


    torch.cuda.empty_cache()
    print(f"  * done in {time.perf_counter() - image_latent_prep_start:.3f}s")

    print("4. Load denoiser ...")
    denoiser_load_start = time.perf_counter()
    print(f"  * variant: {transformer_variant}")
    print(f"  * steps: {sample_steps}")
    transformer = load_flux2_denoiser(
        str(transformer_path),
        quant_method=args.denoiser_quant_method,
        variant=transformer_variant,
        version=args.version,
    )
    transformer = transformer.to(device)
    for parameter in transformer.parameters():
        parameter.requires_grad = False
    lora_module_names = inject_trainable_lora_modules(
        transformer,
        target_linear_names=FLUX2_LORA_TARGETS,
        rank=FLUX2_LORA_RANK,
        alpha=FLUX2_LORA_ALPHA,
    )
    transformer.eval()
    num_channels_latents = _load_transformer_in_channels(transformer_path) // 4
    print(f"  * injected lora modules: {len(lora_module_names)}")
    print(f"  * done in {time.perf_counter() - denoiser_load_start:.3f}s")

    def denoise_samples(step_number: int, pending_sample_indices: list[int]) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(scheduler_path), local_files_only=True)
        sigmas = torch.linspace(1.0, 1 / sample_steps, sample_steps, dtype=torch.float32).tolist()
        if hasattr(scheduler.config, "use_flow_sigmas") and scheduler.config.use_flow_sigmas:
            sigmas = None

        sample_latents = []
        with torch.inference_mode():
            for sample_index in pending_sample_indices:
                existing_output_path = sample_dir / f"{step_number:06d}_{sample_index}.png"
                if existing_output_path.is_file():
                    print(f"    * skipped sample {sample_index}: {existing_output_path}")
                    continue

                selected_index = selected_dataset_index_by_sample[sample_index]
                sample_width, sample_height = sample_resolutions[sample_index]
                print(f"    * denoise sample {sample_index} at {sample_width}x{sample_height} ...")
                if dataset_mode == "prompt":
                    text_embed = cached_dataset.trigger_embed
                    text_id = cached_dataset.trigger_id
                else:
                    text_embed = cached_dataset.text_embeds_list[selected_index]
                    text_id = cached_dataset.text_ids_list[selected_index]
                    if text_embed is None or text_id is None:
                        text_embed = cached_dataset.trigger_embed
                        text_id = cached_dataset.trigger_id
                prompt_embeds_batch = text_embed.unsqueeze(0).to(
                    device=device,
                    dtype=transformer.dtype,
                )
                text_ids_batch = text_id.unsqueeze(0).to(device=device)

                latent_height = 2 * (sample_height // (vae_scale_factor * 2))
                latent_width = 2 * (sample_width // (vae_scale_factor * 2))
                latent_shape = (1, num_channels_latents * 4, latent_height // 2, latent_width // 2)

                latent_generator = torch.Generator(device=device)
                latent_generator.manual_seed(args.sample_seed + sample_index)
                latents = torch.randn(
                    latent_shape,
                    device=device,
                    dtype=prompt_embeds_batch.dtype,
                    generator=latent_generator,
                )
                latent_ids = _prepare_latent_ids(torch, latents).to(device)
                latents = _pack_latents(latents)

                image_seq_len = latents.shape[1]
                mu = _compute_empirical_mu(image_seq_len=image_seq_len, num_steps=sample_steps)
                timesteps, _ = _retrieve_timesteps(
                    scheduler,
                    sample_steps,
                    device=device,
                    sigmas=sigmas,
                    mu=mu,
                )
                scheduler.set_begin_index(0)

                reference_latents_batch = None
                reference_latent_ids_batch = None
                ref_latents = cached_dataset.high_ref_latents[selected_index]
                ref_ids = cached_dataset.high_ref_latent_ids[selected_index]
                if ref_latents is None or ref_ids is None:
                    ref_latents = cached_dataset.base_ref_latents[selected_index]
                    ref_ids = cached_dataset.base_ref_latent_ids[selected_index]
                if ref_latents is not None and ref_ids is not None:
                    reference_latents_batch = ref_latents.unsqueeze(0).to(
                        device=device,
                        dtype=latents.dtype,
                    )
                    reference_latent_ids_batch = ref_ids.unsqueeze(0).to(
                        device=device
                    )

                for timestep_value in timesteps:
                    timestep = timestep_value.expand(latents.shape[0]).to(latents.dtype)
                    latent_model_input = latents.to(transformer.dtype)
                    latent_image_ids = latent_ids
                    if reference_latents_batch is not None and reference_latent_ids_batch is not None:
                        latent_model_input = torch.cat([latents, reference_latents_batch], dim=1).to(
                            transformer.dtype
                        )
                        latent_image_ids = torch.cat([latent_ids, reference_latent_ids_batch], dim=1)

                    with transformer.cache_context("cond"):
                        noise_pred = transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep / 1000,
                            guidance=None,
                            encoder_hidden_states=prompt_embeds_batch,
                            txt_ids=text_ids_batch,
                            img_ids=latent_image_ids,
                            joint_attention_kwargs=None,
                            return_dict=False,
                        )[0]
                    noise_pred = noise_pred[:, : latents.size(1)]

                    latents_dtype = latents.dtype
                    latents = scheduler.step(noise_pred, timestep_value, latents, return_dict=False)[0]
                    if latents.dtype != latents_dtype:
                        latents = latents.to(latents_dtype)

                sample_latents.append((sample_index, latents.cpu(), latent_ids.cpu()))
                del (
                    prompt_embeds_batch,
                    text_ids_batch,
                    latents,
                    latent_ids,
                    reference_latents_batch,
                    reference_latent_ids_batch,
                    noise_pred,
                )

        return sample_latents

    def decode_samples(step_number: int, sample_latents: list[tuple[int, torch.Tensor, torch.Tensor]]) -> None:
        vae.to(device, dtype=torch.float16)
        with torch.inference_mode():
            for sample_index, latents_cpu, latent_ids_cpu in sample_latents:
                output_path = next_sample_output_path(step_number, sample_index)
                latents = latents_cpu.to(device=device)
                latent_ids = latent_ids_cpu.to(device=device)
                latents = _unpack_latents_with_ids(torch, latents, latent_ids)

                latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
                latents_bn_std = torch.sqrt(
                    vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
                ).to(latents.device, latents.dtype)
                latents = latents * latents_bn_std + latents_bn_mean
                latents = _unpatchify_latents(latents)
                decoded = vae.decode(latents.to(dtype=vae.dtype), return_dict=False)[0]
                image = image_processor.postprocess(decoded, output_type="pil")[0]
                image.save(output_path)
                print(f"    * saved {output_path}")
                del latents, latent_ids, decoded, image
        vae.to("cpu")
        torch.cuda.empty_cache()

    print("5. Render checkpoint samples ...")
    render_start = time.perf_counter()
    print(f"  * checkpoints: {len(checkpoints)}")
    print(f"  * sample output dir: {sample_dir}")
    for step_number, checkpoint_path, pending_sample_indices in checkpoints_to_render:
        print(f"  * step {step_number}: {checkpoint_path}")
        step_start = time.perf_counter()
        load_lora_checkpoint(transformer, checkpoint_path)
        sample_latents = denoise_samples(step_number, pending_sample_indices)
        transformer.to("cpu")
        torch.cuda.empty_cache()
        print("    * decoding samples ...")
        decode_samples(step_number, sample_latents)
        transformer.to(device)
        print(f"    * step sample time: {time.perf_counter() - step_start:.3f}s")

    del transformer
    del vae
    torch.cuda.empty_cache()
    print(f"  * sampling done in {time.perf_counter() - render_start:.3f}s")


if __name__ == "__main__":
    main()
