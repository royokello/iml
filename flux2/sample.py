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


def _resize_to_max_side(*, image_width: int, image_height: int, max_side: int) -> tuple[int, int]:
    if image_width >= image_height:
        scaled_width = max_side
        scaled_height = round(image_height * max_side / image_width)
    else:
        scaled_height = max_side
        scaled_width = round(image_width * max_side / image_height)

    return max(1, scaled_width), max(1, scaled_height)


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
        help="Dataset-only override for the longest image side. Keeps each sample's aspect ratio.",
    )
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument(
        "--start",
        type=_positive_int,
        default=1,
        help="First step checkpoint number to render samples for.",
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

    def sample_output_path(step_number: int, sample_index: int) -> Path:
        return sample_dir / f"{step_number:06d}_{sample_index}.png"

    def next_sample_output_path(step_number: int, sample_index: int) -> Path:
        output_path = sample_output_path(step_number, sample_index)
        if not output_path.is_file():
            return output_path

        duplicate_index = 2
        while True:
            duplicate_path = sample_dir / f"{step_number:06d}_{sample_index}_{duplicate_index}.png"
            if not duplicate_path.is_file():
                return duplicate_path
            duplicate_index += 1

    resolved_version = args.version.strip().lower()
    model_root = args.root / f"flux2_{resolved_version}" / "model"
    normalized_prompt = None if args.prompt is None else args.prompt.strip()
    if normalized_prompt == "":
        raise ValueError("--prompt must not be empty.")
    if normalized_prompt is not None and args.trigger is not None:
        raise ValueError("--trigger is only used with captionless datasets.")
    if normalized_prompt is not None and args.force_size is not None:
        raise ValueError("--force-size is only used with datasets.")

    dataset_mode = "prompt"
    reference_images: list[list[Path]] = []
    sample_resolutions: dict[int, tuple[int, int]] = {}
    cached_dataset = None

    print("1. Load and encode dataset ...")
    from flux2.train.dataset import Flux2Dataset

    target_images = _list_image_files(project_path / "images" / "high")
    reference_images = []
    for target_image in target_images:
        references_dir = target_image.parent / target_image.stem
        reference_images.append(_list_image_files(references_dir) if references_dir.is_dir() else [])

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
        project_path,
        normalized_trigger,
        indices=selected_dataset_indices,
    )

    if normalized_prompt is not None:
        dataset_mode = "prompt"
    elif cached_dataset.shared_prompt_embeds is not None:
        dataset_mode = "captionless"
    else:
        dataset_mode = "captioned"

    first_image_size = cached_dataset.high_res_encodings.target_image_resolutions[0]

    selected_reference_count = 0
    for sample_index in sample_indices:
        selected_index = selected_dataset_index_by_sample[sample_index]
        sample_width, sample_height = cached_dataset.high_res_encodings.target_image_resolutions[selected_index]
        if args.force_size is not None:
            sample_width, sample_height = _resize_to_max_side(
                image_width=sample_width,
                image_height=sample_height,
                max_side=args.force_size,
            )
        elif normalized_prompt is not None:
            sample_width = args.width
            sample_height = args.height
        sample_resolutions[sample_index] = (sample_width, sample_height)
        selected_reference_count += len(reference_images[sample_index - 1])

    print(f"  * dataset mode: {dataset_mode}")
    print(f"  * images: {len(target_images)}")
    print(f"  * selected samples: {', '.join(str(sample_index) for sample_index in sample_indices)}")
    print(f"  * selected reference images: {selected_reference_count}")
    if normalized_trigger is not None:
        print(f"  * trigger: {normalized_trigger!r}")
    if args.force_size is not None:
        print(f"  * forced max side: {args.force_size}")
    if normalized_prompt is not None:
        print(f"  * requested image size: {args.width}x{args.height}")
    print(f"  * first image size: {first_image_size[0]}x{first_image_size[1]}")
    print(f"  * done in {time.perf_counter() - dataset_load_start:.3f}s")

    if not sample_indices:
        print("  * no samples requested")
        return

    checkpoints_to_render = []
    existing_step_outputs = 0
    for step_number, checkpoint_path in checkpoints:
        has_existing_outputs = any(sample_output_path(step_number, sample_index).is_file() for sample_index in sample_indices)
        if has_existing_outputs:
            existing_step_outputs += 1
        checkpoints_to_render.append((step_number, checkpoint_path, sample_indices))

    if existing_step_outputs:
        print(f"  * steps with existing samples: {existing_step_outputs}")
        print("  * existing samples will be preserved; new renders use the next available suffix")
    print(f"  * steps to render: {len(checkpoints_to_render)}")

    from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    from safetensors.torch import load_file as safe_load_file
    import torch

    from flux2.denoiser.loader import _load_flux2_denoiser as load_flux2_denoiser
    from flux2.lora import TrainableLoraLinear, inject_trainable_lora_modules
    from flux2.train.main import INITIAL_LORA_ALPHA, INITIAL_LORA_RANK, INITIAL_LORA_TARGET_LINEAR_NAMES
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

    print("2. Select cached sample conditioning ...")
    text_encoding_start = time.perf_counter()
    current_encodings = cached_dataset.high_res_encodings
    if current_encodings.prompt_embeds is not None and current_encodings.text_ids is not None:
        print("  * using high-res caption encodings from dataset")
    else:
        print("  * using shared trigger encoding from dataset")
    torch.cuda.empty_cache()
    print(f"  * done in {time.perf_counter() - text_encoding_start:.3f}s")

    print("3. Prepare VAE for sample resolutions and decode ...")
    image_latent_prep_start = time.perf_counter()
    vae_scale_factor = _load_vae_scale_factor(vae_path)
    from diffusers import AutoencoderKLFlux2

    vae = AutoencoderKLFlux2.from_pretrained(str(vae_path), local_files_only=True)
    image_processor = Flux2ImageProcessor(vae_scale_factor=vae_scale_factor * 2)

    if normalized_prompt is not None or args.force_size is not None:
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

    reference_latent_token_count = sum(
        latent.shape[0] for latent in current_encodings.reference_latents if latent is not None
    )
    print(f"  * cached reference latent tokens on cpu: {reference_latent_token_count}")
    torch.cuda.empty_cache()
    print(f"  * done in {time.perf_counter() - image_latent_prep_start:.3f}s")

    print("4. Load denoiser ...")
    denoiser_load_start = time.perf_counter()
    transformer = load_flux2_denoiser(
        str(transformer_path),
        quant_method=args.denoiser_quant_method,
        variant="distill",
        version=resolved_version,
    )
    transformer = transformer.to(device)
    for parameter in transformer.parameters():
        parameter.requires_grad = False
    lora_module_names = inject_trainable_lora_modules(
        transformer,
        target_linear_names=INITIAL_LORA_TARGET_LINEAR_NAMES,
        rank=INITIAL_LORA_RANK,
        alpha=INITIAL_LORA_ALPHA,
    )
    transformer.eval()
    num_channels_latents = _load_transformer_in_channels(transformer_path) // 4
    torch.cuda.synchronize(device)
    print(f"  * injected lora modules: {len(lora_module_names)}")
    print(f"  * done in {time.perf_counter() - denoiser_load_start:.3f}s")

    def load_checkpoint(checkpoint_path: Path) -> None:
        checkpoint_state = safe_load_file(str(checkpoint_path), device="cpu")
        for module_name, child in transformer.named_modules():
            if not isinstance(child, TrainableLoraLinear):
                continue

            lora_a_key = f"{module_name}.lora_A.weight"
            lora_b_key = f"{module_name}.lora_B.weight"
            if lora_a_key not in checkpoint_state or lora_b_key not in checkpoint_state:
                raise KeyError(f"Missing LoRA weights for {module_name} in checkpoint {checkpoint_path}")

            child.lora_A.data.copy_(
                checkpoint_state[lora_a_key].to(device=child.lora_A.device, dtype=child.lora_A.dtype)
            )
            child.lora_B.data.copy_(
                checkpoint_state[lora_b_key].to(device=child.lora_B.device, dtype=child.lora_B.dtype)
            )

    def denoise_samples(step_number: int, pending_sample_indices: list[int]) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
        sample_steps = DEFAULT_DISTILLED_STEPS
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(scheduler_path), local_files_only=True)
        sigmas = torch.linspace(1.0, 1 / sample_steps, sample_steps, dtype=torch.float32).tolist()
        if hasattr(scheduler.config, "use_flow_sigmas") and scheduler.config.use_flow_sigmas:
            sigmas = None

        sample_latents = []
        with torch.inference_mode():
            for sample_index in pending_sample_indices:
                selected_index = selected_dataset_index_by_sample[sample_index]
                sample_width, sample_height = sample_resolutions[sample_index]
                print(f"    * denoise sample {sample_index} at {sample_width}x{sample_height} ...")
                if current_encodings.prompt_embeds is not None and current_encodings.text_ids is not None:
                    prompt_embeds_batch = current_encodings.prompt_embeds[
                        selected_index : selected_index + 1
                    ].to(device=device, dtype=transformer.dtype)
                    text_ids_batch = current_encodings.text_ids[selected_index : selected_index + 1].to(device=device)
                else:
                    prompt_embeds_batch = cached_dataset.shared_prompt_embeds.to(
                        device=device,
                        dtype=transformer.dtype,
                    )
                    text_ids_batch = cached_dataset.shared_text_ids.to(device=device)

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
                if (
                    current_encodings.reference_latents[selected_index] is not None
                    and current_encodings.reference_latent_ids[selected_index] is not None
                ):
                    reference_latents_batch = current_encodings.reference_latents[selected_index].unsqueeze(0).to(
                        device=device,
                        dtype=latents.dtype,
                    )
                    reference_latent_ids_batch = current_encodings.reference_latent_ids[selected_index].unsqueeze(0).to(
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

                torch.cuda.synchronize(device)
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
        torch.cuda.synchronize(device)
        vae.to("cpu")
        torch.cuda.empty_cache()

    print("5. Render checkpoint samples ...")
    render_start = time.perf_counter()
    print(f"  * checkpoints: {len(checkpoints)}")
    print(f"  * sample output dir: {sample_dir}")
    for step_number, checkpoint_path, pending_sample_indices in checkpoints_to_render:
        print(f"  * step {step_number}: {checkpoint_path}")
        step_start = time.perf_counter()
        load_checkpoint(checkpoint_path)
        sample_latents = denoise_samples(step_number, pending_sample_indices)
        transformer.to("cpu")
        torch.cuda.empty_cache()
        print("    * decoding samples ...")
        decode_samples(step_number, sample_latents)
        transformer.to(device)
        torch.cuda.synchronize(device)
        print(f"    * step sample time: {time.perf_counter() - step_start:.3f}s")

    del transformer
    del vae
    torch.cuda.empty_cache()
    print(f"  * sampling done in {time.perf_counter() - render_start:.3f}s")


if __name__ == "__main__":
    main()
