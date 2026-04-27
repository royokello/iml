from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.quant.validators import CLI_QUANT_METHODS

DEFAULT_SAMPLE_STEPS = 10
DEFAULT_SAMPLE_SEED = 19930625


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


def _list_epoch_checkpoints(models_dir: Path) -> list[tuple[int, Path]]:
    checkpoints: list[tuple[int, Path]] = []
    for checkpoint_path in models_dir.glob("epoch_*.safetensors"):
        epoch_text = checkpoint_path.stem.removeprefix("epoch_")
        try:
            epoch_number = int(epoch_text)
        except ValueError:
            continue
        checkpoints.append((epoch_number, checkpoint_path))

    checkpoints.sort(key=lambda item: item[0])
    if not checkpoints:
        raise FileNotFoundError(f"No epoch checkpoints found in {models_dir}")
    return checkpoints


def _normalize_image_latent_resolution(
    *,
    image_width: int,
    image_height: int,
    vae_scale_factor: int,
) -> tuple[int, int]:
    original_width = image_width
    original_height = image_height
    multiple_of = vae_scale_factor * 2
    image_width = (image_width // multiple_of) * multiple_of
    image_height = (image_height // multiple_of) * multiple_of
    if image_width <= 0 or image_height <= 0:
        raise ValueError(
            f"Image is smaller than required multiple {multiple_of}: {original_width}x{original_height}"
        )
    return image_width, image_height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Flux 2 training samples for saved LoRA epochs.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--version",
        choices=("4b", "9b"),
        required=True,
        help="Model family to load.",
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Training output directory that contains the models subdirectory.",
    )
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument(
        "--start",
        type=_positive_int,
        default=1,
        help="First epoch number to render samples for.",
    )
    parser.add_argument("--sample-seed", "--sample_seed", dest="sample_seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--sample-indices", "--sample_indices", dest="sample_indices", type=str, default=None)
    parser.add_argument("--text-quant-method", choices=CLI_QUANT_METHODS, default="sym-high")
    parser.add_argument("--denoiser-quant-method", choices=CLI_QUANT_METHODS, default="sym-low")
    parser.add_argument(
        "--trigger",
        type=str,
        help="Required for captionless datasets. Encoded once and kept on GPU for all samples.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from flux2.loaders import load_flux2_dataset

    output_dir = args.output.expanduser().resolve()
    models_dir = output_dir / "models"
    if not models_dir.is_dir():
        raise FileNotFoundError(f"Expected training models directory at {models_dir}")

    checkpoints = _list_epoch_checkpoints(models_dir)
    checkpoints = [
        (epoch_number, checkpoint_path)
        for epoch_number, checkpoint_path in checkpoints
        if epoch_number >= args.start
    ]
    if not checkpoints:
        raise FileNotFoundError(f"No epoch checkpoints found at or after epoch {args.start} in {models_dir}")
    sample_dir = output_dir / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)

    def sample_output_path(epoch_number: int, sample_index: int) -> Path:
        return sample_dir / f"epoch_{epoch_number}_{sample_index}.png"

    print("1. Load dataset into cpu ram ...")
    dataset_load_start = time.perf_counter()
    dataset = load_flux2_dataset(args.dataset)
    target_images = dataset["target_images"]
    text_prompts = dataset["text_prompts"]
    reference_images = dataset["reference_images"]
    if not (len(target_images) == len(text_prompts) == len(reference_images)):
        raise ValueError("Dataset loader returned misaligned target, prompt, and reference lists.")

    sample_indices = _resolve_sample_indices(
        dataset_size=len(target_images),
        samples=args.samples,
        sample_indices=args.sample_indices,
    )
    if not sample_indices:
        print("  * no samples requested")
        return

    checkpoints_to_render = []
    skipped_epochs = 0
    for epoch_number, checkpoint_path in checkpoints:
        pending_sample_indices = [
            sample_index for sample_index in sample_indices if not sample_output_path(epoch_number, sample_index).is_file()
        ]
        if pending_sample_indices:
            checkpoints_to_render.append((epoch_number, checkpoint_path, pending_sample_indices))
        else:
            skipped_epochs += 1

    if skipped_epochs:
        print(f"  * epochs with existing samples: {skipped_epochs}")
    if not checkpoints_to_render:
        print("  * all requested epoch samples already exist")
        return
    print(f"  * epochs to render: {len(checkpoints_to_render)}")

    normalized_trigger = None if args.trigger is None else args.trigger.strip()
    if normalized_trigger == "":
        raise ValueError("--trigger must not be empty.")

    missing_prompt_count = sum(prompt is None for prompt in text_prompts)
    if missing_prompt_count and normalized_trigger is None:
        raise ValueError(f"{missing_prompt_count} samples without .txt prompts require --trigger.")
    if not missing_prompt_count and normalized_trigger is not None:
        raise ValueError("--trigger is only used for samples without .txt prompts.")

    dataset_mode = "captionless" if missing_prompt_count == len(text_prompts) else "captioned"
    trigger_text = normalized_trigger if dataset_mode == "captionless" else None
    captions = [prompt if prompt is not None else normalized_trigger for prompt in text_prompts]
    if any(caption is None for caption in captions):
        raise ValueError("Internal dataset prompt resolution failed.")

    from PIL import Image

    sample_resolutions: dict[int, tuple[int, int]] = {}
    with Image.open(target_images[0]) as first_image:
        first_image_size = first_image.size

    selected_reference_count = 0
    for sample_index in sample_indices:
        dataset_index = sample_index - 1
        with Image.open(target_images[dataset_index]) as sample_image:
            sample_resolutions[sample_index] = sample_image.size
        selected_reference_count += len(reference_images[dataset_index])

    print(f"  * dataset mode: {dataset_mode}")
    print(f"  * images: {len(target_images)}")
    print(f"  * selected samples: {', '.join(str(sample_index) for sample_index in sample_indices)}")
    print(f"  * selected reference images: {selected_reference_count}")
    if trigger_text is not None:
        print(f"  * trigger: {trigger_text!r}")
    print(f"  * first image size: {first_image_size[0]}x{first_image_size[1]}")
    print(f"  * done in {time.perf_counter() - dataset_load_start:.3f}s")

    from diffusers import AutoencoderKLFlux2
    from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    from safetensors.torch import load_file as safe_load_file
    import torch
    from transformers import Qwen2TokenizerFast

    from flux2.loaders import load_flux2_denoiser, load_flux2_text_encoder
    from flux2.lora import TrainableLoraLinear, inject_trainable_lora_modules
    from flux2.train import INITIAL_LORA_ALPHA, INITIAL_LORA_RANK, INITIAL_LORA_TARGET_LINEAR_NAMES
    from flux2.gen import (
        DEFAULT_BASE_GUIDANCE_SCALE,
        _compute_empirical_mu,
        _encode_prompt_embeddings,
        _load_transformer_in_channels,
        _load_vae_scale_factor,
        _pack_latents,
        _patchify_latents,
        _prepare_image_ids,
        _prepare_latent_ids,
        _resolve_model_root,
        _retrieve_latents,
        _retrieve_timesteps,
        _unpack_latents_with_ids,
        _unpatchify_latents,
    )

    device = torch.device("cuda")
    resolved_version = args.version.strip().lower()
    model_root = _resolve_model_root(args.root, resolved_version)
    tokenizer_path = model_root / "tokenizer"
    text_encoder_path = model_root / "text_encoder"
    vae_path = model_root / "vae"
    scheduler_path = model_root / "scheduler"
    transformer_path = model_root / "transformer"

    print("2. Load text encoder into gpu ...")
    text_encoder_load_start = time.perf_counter()
    tokenizer = Qwen2TokenizerFast.from_pretrained(str(tokenizer_path))
    text_encoder = load_flux2_text_encoder(
        str(text_encoder_path),
        quant_method=args.text_quant_method,
    )
    text_encoder = text_encoder.to(device)
    torch.cuda.synchronize(device)
    print(f"  * done in {time.perf_counter() - text_encoder_load_start:.3f}s")

    print("3. Prepare text encodings ...")
    text_encoding_start = time.perf_counter()
    prompt_embeds_by_sample: dict[int, torch.Tensor] = {}
    text_ids_by_sample: dict[int, torch.Tensor] = {}
    shared_prompt_embeds = None
    shared_text_ids = None
    negative_prompt_embeds = None
    negative_text_ids = None

    if dataset_mode == "captioned":
        for sample_index in sample_indices:
            dataset_index = sample_index - 1
            print(f"  * encode sample {sample_index} ...")
            encode_start = time.perf_counter()
            sample_prompt_embeds, sample_text_ids = _encode_prompt_embeddings(
                torch,
                tokenizer,
                text_encoder,
                prompt=captions[dataset_index],
                device=device,
                max_length=512,
            )
            torch.cuda.synchronize(device)
            prompt_embeds_by_sample[sample_index] = sample_prompt_embeds.cpu()
            text_ids_by_sample[sample_index] = sample_text_ids.cpu()
            print(f"    encode time: {time.perf_counter() - encode_start:.3f}s")
    else:
        encode_start = time.perf_counter()
        shared_prompt_embeds, shared_text_ids = _encode_prompt_embeddings(
            torch,
            tokenizer,
            text_encoder,
            prompt=trigger_text,
            device=device,
            max_length=512,
        )
        torch.cuda.synchronize(device)
        print(f"  * trigger encode time: {time.perf_counter() - encode_start:.3f}s")

    encode_start = time.perf_counter()
    negative_prompt_embeds, negative_text_ids = _encode_prompt_embeddings(
        torch,
        tokenizer,
        text_encoder,
        prompt="",
        device=device,
        max_length=512,
    )
    torch.cuda.synchronize(device)
    negative_prompt_embeds = negative_prompt_embeds.cpu()
    negative_text_ids = negative_text_ids.cpu()
    print(f"  * negative prompt encode time: {time.perf_counter() - encode_start:.3f}s")

    del tokenizer
    del text_encoder
    torch.cuda.empty_cache()
    print(f"  * done in {time.perf_counter() - text_encoding_start:.3f}s")

    print("4. Prepare reference image latents ...")
    image_latent_prep_start = time.perf_counter()
    vae_scale_factor = _load_vae_scale_factor(vae_path)
    vae = AutoencoderKLFlux2.from_pretrained(str(vae_path), local_files_only=True)
    vae = vae.to(device, dtype=torch.float16)
    image_processor = Flux2ImageProcessor(vae_scale_factor=vae_scale_factor * 2)

    for sample_index, (sample_width, sample_height) in list(sample_resolutions.items()):
        sample_resolutions[sample_index] = _normalize_image_latent_resolution(
            image_width=sample_width,
            image_height=sample_height,
            vae_scale_factor=vae_scale_factor,
        )
    sample_resolution_text = ", ".join(
        f"{sample_index}={width}x{height}" for sample_index, (width, height) in sample_resolutions.items()
    )
    print(f"  * sample resolutions: {sample_resolution_text}")

    def encode_reference_image(image) -> tuple[torch.Tensor, torch.Tensor]:
        image_processor.check_image_input(image)
        image_width, image_height = _normalize_image_latent_resolution(
            image_width=image.size[0],
            image_height=image.size[1],
            vae_scale_factor=vae_scale_factor,
        )

        image_tensor = image_processor.preprocess(image, height=image_height, width=image_width, resize_mode="crop")
        image_tensor = image_tensor.to(device=device, dtype=torch.float16)
        with torch.inference_mode():
            latent = _retrieve_latents(vae.encode(image_tensor))
        latent = _patchify_latents(latent)

        latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latent.device, latent.dtype)
        latents_bn_std = torch.sqrt(
            vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
        ).to(latent.device, latent.dtype)
        latent = (latent - latents_bn_mean) / latents_bn_std

        return _pack_latents(latent).squeeze(0).cpu(), latent

    reference_latents_by_sample: dict[int, torch.Tensor] = {}
    reference_latent_ids_by_sample: dict[int, torch.Tensor] = {}
    for sample_index in sample_indices:
        dataset_index = sample_index - 1
        sample_references = reference_images[dataset_index]
        if not sample_references:
            continue

        print(f"  * sample {sample_index} references: {len(sample_references)} ...")
        reference_encode_start = time.perf_counter()
        encoded_reference_latents = []
        raw_reference_latents = []
        for reference_index, reference_path in enumerate(sample_references, start=1):
            print(f"    * reference {reference_index} / {len(sample_references)} ...")
            reference_image = Image.open(reference_path).convert("RGB")
            reference_latent, raw_reference_latent = encode_reference_image(reference_image)
            encoded_reference_latents.append(reference_latent)
            raw_reference_latents.append(raw_reference_latent)

        reference_latents_by_sample[sample_index] = torch.cat(encoded_reference_latents, dim=0)
        reference_latent_ids_by_sample[sample_index] = _prepare_image_ids(torch, raw_reference_latents).squeeze(0).cpu()
        torch.cuda.synchronize(device)
        print(f"    encode time: {time.perf_counter() - reference_encode_start:.3f}s")

    reference_latent_token_count = sum(latent.shape[0] for latent in reference_latents_by_sample.values())
    print(f"  * cached reference latent tokens on cpu: {reference_latent_token_count}")
    vae.to("cpu")
    torch.cuda.empty_cache()
    print(f"  * done in {time.perf_counter() - image_latent_prep_start:.3f}s")

    print("5. Load denoiser ...")
    denoiser_load_start = time.perf_counter()
    transformer = load_flux2_denoiser(
        str(transformer_path),
        quant_method=args.denoiser_quant_method,
        variant="base",
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

    def denoise_samples(epoch_number: int, pending_sample_indices: list[int]) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
        sample_steps = DEFAULT_SAMPLE_STEPS
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(scheduler_path), local_files_only=True)
        sigmas = torch.linspace(1.0, 1 / sample_steps, sample_steps, dtype=torch.float32).tolist()
        if hasattr(scheduler.config, "use_flow_sigmas") and scheduler.config.use_flow_sigmas:
            sigmas = None

        sample_latents = []
        with torch.inference_mode():
            for sample_index in pending_sample_indices:
                sample_width, sample_height = sample_resolutions[sample_index]
                print(f"    * denoise sample {sample_index} at {sample_width}x{sample_height} ...")
                if prompt_embeds_by_sample and text_ids_by_sample:
                    prompt_embeds_batch = prompt_embeds_by_sample[sample_index].to(device=device, dtype=transformer.dtype)
                    text_ids_batch = text_ids_by_sample[sample_index].to(device=device)
                else:
                    prompt_embeds_batch = shared_prompt_embeds.to(device=device, dtype=transformer.dtype)
                    text_ids_batch = shared_text_ids.to(device=device)

                negative_prompt_embeds_batch = negative_prompt_embeds.to(device=device, dtype=transformer.dtype)
                negative_text_ids_batch = negative_text_ids.to(device=device)

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
                if sample_index in reference_latents_by_sample:
                    reference_latents_batch = reference_latents_by_sample[sample_index].unsqueeze(0).to(
                        device=device,
                        dtype=latents.dtype,
                    )
                    reference_latent_ids_batch = reference_latent_ids_by_sample[sample_index].unsqueeze(0).to(
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

                    with transformer.cache_context("uncond"):
                        neg_noise_pred = transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep / 1000,
                            guidance=None,
                            encoder_hidden_states=negative_prompt_embeds_batch,
                            txt_ids=negative_text_ids_batch,
                            img_ids=latent_image_ids,
                            joint_attention_kwargs=None,
                            return_dict=False,
                        )[0]
                    neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
                    noise_pred = neg_noise_pred + DEFAULT_BASE_GUIDANCE_SCALE * (noise_pred - neg_noise_pred)

                    latents_dtype = latents.dtype
                    latents = scheduler.step(noise_pred, timestep_value, latents, return_dict=False)[0]
                    if latents.dtype != latents_dtype:
                        latents = latents.to(latents_dtype)

                torch.cuda.synchronize(device)
                sample_latents.append((sample_index, latents.cpu(), latent_ids.cpu()))
                del (
                    prompt_embeds_batch,
                    text_ids_batch,
                    negative_prompt_embeds_batch,
                    negative_text_ids_batch,
                    latents,
                    latent_ids,
                    reference_latents_batch,
                    reference_latent_ids_batch,
                    noise_pred,
                    neg_noise_pred,
                )

        return sample_latents

    def decode_samples(epoch_number: int, sample_latents: list[tuple[int, torch.Tensor, torch.Tensor]]) -> None:
        vae.to(device, dtype=torch.float16)
        with torch.inference_mode():
            for sample_index, latents_cpu, latent_ids_cpu in sample_latents:
                output_path = sample_output_path(epoch_number, sample_index)
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

    print("6. Render checkpoint samples ...")
    render_start = time.perf_counter()
    print(f"  * checkpoints: {len(checkpoints)}")
    print(f"  * sample output dir: {sample_dir}")
    for epoch_number, checkpoint_path, pending_sample_indices in checkpoints_to_render:
        print(f"  * epoch {epoch_number}: {checkpoint_path}")
        epoch_start = time.perf_counter()
        load_checkpoint(checkpoint_path)
        sample_latents = denoise_samples(epoch_number, pending_sample_indices)
        transformer.to("cpu")
        torch.cuda.empty_cache()
        print("    * decoding samples ...")
        decode_samples(epoch_number, sample_latents)
        transformer.to(device)
        torch.cuda.synchronize(device)
        print(f"    * epoch sample time: {time.perf_counter() - epoch_start:.3f}s")

    del transformer
    del vae
    torch.cuda.empty_cache()
    print(f"  * sampling done in {time.perf_counter() - render_start:.3f}s")


if __name__ == "__main__":
    main()
