from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

DEFAULT_SAMPLE_SEED = 19930625
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def _parse_sample_indices(value: str | None) -> list[int] | None:
    if value is None:
        return None
    return [int(index.strip()) for index in value.split(",") if index.strip()]


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
        type=int,
        default=512,
        help="Prompt-only sample width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Prompt-only sample height.",
    )
    parser.add_argument(
        "--target-res",
        dest="target_res",
        type=int,
        default=512,
        help="Short side for generation output. Combined with each sample's aspect ratio to determine output dimensions.",
    )
    parser.add_argument(
        "--ref-res",
        dest="ref_res",
        type=int,
        default=512,
        help="Resolution for reference image encoding.",
    )
    parser.add_argument(
        "--target-upscale",
        dest="target_upscale",
        action="store_true",
        help="Upscale target images below target_res to meet the target resolution.",
    )
    parser.add_argument(
        "--ref-upscale",
        dest="ref_upscale",
        action="store_true",
        help="Upscale reference images below ref_res to meet the target resolution.",
    )
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="First step checkpoint number to render samples for.",
    )
    parser.add_argument(
        "--steps",
        type=int,
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
    parser.add_argument("--text-quant-method", default="sym-med-nano")
    parser.add_argument("--denoiser-quant-method", default="sym-med-nano")
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

    def next_sample_output_path(step_number: int, sample_index: int | None, sample_width: int, sample_height: int) -> Path:
        if single_sample:
            output_path = sample_dir / f"{step_number:06d}_{sample_width}x{sample_height}.png"
        else:
            output_path = sample_dir / f"{step_number:06d}_{sample_index}_{sample_width}x{sample_height}.png"
        if not output_path.is_file():
            return output_path

        duplicate_index = 2
        while True:
            if single_sample:
                duplicate_path = sample_dir / f"{step_number:06d}_{sample_width}x{sample_height}_{duplicate_index}.png"
            else:
                duplicate_path = sample_dir / f"{step_number:06d}_{sample_index}_{sample_width}x{sample_height}_{duplicate_index}.png"
            if not duplicate_path.is_file():
                return duplicate_path
            duplicate_index += 1

    model_root = args.root / "flux2" / args.version / "model"
    normalized_prompt = None if args.prompt is None else args.prompt.strip()
    if normalized_prompt == "":
        raise ValueError("--prompt must not be empty.")
    if normalized_prompt is not None and args.trigger is not None:
        raise ValueError("--trigger is only used with captionless datasets.")

    sample_resolutions: dict[int, tuple[int, int]] = {}

    print("1. Load and encode dataset ...")
    from flux2.train.dataset import Flux2Dataset

    dataset_dir = project_path / "images"

    dataset_load_start = time.perf_counter()
    parsed_indices = _parse_sample_indices(args.sample_indices)
    normalized_trigger = normalized_prompt if normalized_prompt is not None else args.trigger
    cached_dataset = Flux2Dataset(
        model_rootpath=model_root,
        text_quant_method=args.text_quant_method,
        dataset_dirpath=dataset_dir,
        trigger_str=normalized_trigger,
        target_resolution=args.target_res,
        reference_resolution=args.ref_res,
        indices=None if parsed_indices is None else [i - 1 for i in parsed_indices],
        load_target_images=False,
        load_target_ratios=True,
        target_upscale=args.target_upscale,
        ref_upscale=args.ref_upscale,
    )

    dataset_size = len(cached_dataset.target_ratios)
    if dataset_size == 0:
        print("  * no samples loaded")
        return

    sample_indices = parsed_indices if parsed_indices is not None else list(range(1, dataset_size + 1))

    selected_dataset_index_by_sample = {
        sample_index: selected_index for selected_index, sample_index in enumerate(sample_indices)
    }

    if normalized_prompt is not None:
        dataset_mode = "prompt"
    else:
        dataset_mode = "captioned"

    short_side = args.target_res
    for sample_index in sample_indices:
        selected_index = selected_dataset_index_by_sample[sample_index]
        if cached_dataset.target_ratios[selected_index] is None:
            continue
        if normalized_prompt is not None and args.sample_indices is None:
            sample_width = args.width
            sample_height = args.height
        else:
            sample_width, sample_height = _resolution_from_ratio(
                ratio=cached_dataset.target_ratios[selected_index],
                short_side=short_side,
            )
        sample_resolutions[sample_index] = (sample_width, sample_height)

    sample_indices = list(sample_resolutions.keys())

    single_sample = args.sample_indices is None

    print(f"  * dataset mode: {dataset_mode}")
    print(f"  * images: {dataset_size}")
    print(f"  * selected samples: {', '.join(str(sample_index) for sample_index in sample_indices)}")
    if normalized_trigger is not None:
        print(f"  * trigger: {normalized_trigger!r}")
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
    from flux2.lora.config import FLUX2_LORA_TARGETS
    from safetensors import safe_open

    # Read rank/alpha from the first checkpoint so modules are created at the
    # right size regardless of what the config defaults say.
    if not checkpoints:
        raise FileNotFoundError("No checkpoints to load metadata from.")
    first_cp_meta: dict[str, str] | None = None
    for _step_num, cp_path in checkpoints:
        try:
            with safe_open(str(cp_path), framework="pt") as f:
                first_cp_meta = f.metadata()
        except Exception:
            continue
        break
    if first_cp_meta is None:
        raise RuntimeError("Could not read safetensors metadata from any checkpoint.")
    lora_rank = int(first_cp_meta["rank"])
    lora_alpha = int(first_cp_meta["alpha"])
    print(f"  * lora rank/alpha from checkpoint: r={lora_rank}, a={lora_alpha}")
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
        rank=lora_rank,
        alpha=lora_alpha,
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
                sample_width, sample_height = sample_resolutions[sample_index]
                if single_sample:
                    existing_output_path = sample_dir / f"{step_number:06d}_{sample_width}x{sample_height}.png"
                else:
                    existing_output_path = sample_dir / f"{step_number:06d}_{sample_index}_{sample_width}x{sample_height}.png"
                if existing_output_path.is_file():
                    print(f"    * skipped sample {sample_index}: {existing_output_path}")
                    continue

                selected_index = selected_dataset_index_by_sample[sample_index]
                sample_width, sample_height = sample_resolutions[sample_index]
                print(f"    * denoise sample {sample_index} at {sample_width}x{sample_height} ...")
                if dataset_mode == "prompt":
                    text_embed, text_id = cached_dataset.trigger_embedding
                else:
                    text_item = cached_dataset.text_embeddings[selected_index]
                    if text_item is not None:
                        text_embed, text_id = text_item
                    else:
                        text_embed, text_id = cached_dataset.trigger_embedding
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

                ref_latents = cached_dataset.ref_latents[selected_index]
                reference_latents_batch = None
                reference_latent_ids_batch = None
                if ref_latents is not None:
                    ref_latent_tensor, ref_ids_tensor = ref_latents
                    reference_latents_batch = ref_latent_tensor.unsqueeze(0).to(
                        device=device,
                        dtype=latents.dtype,
                    )
                    reference_latent_ids_batch = ref_ids_tensor.unsqueeze(0).to(device=device)

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
                sample_width, sample_height = sample_resolutions[sample_index]
                output_path = next_sample_output_path(step_number, sample_index, sample_width, sample_height)
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
