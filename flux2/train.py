from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from utils.quant.validators import CLI_QUANT_METHODS

INITIAL_LORA_TARGET_LINEAR_NAMES = (
    "transformer_blocks.attn.to_q",
    "transformer_blocks.attn.to_k",
    "transformer_blocks.attn.to_v",
    "transformer_blocks.attn.to_out",
    "transformer_blocks.ff.linear_in",
    "transformer_blocks.ff.linear_out",
    "single_transformer_blocks.attn.to_qkv_mlp_proj",
    "single_transformer_blocks.attn.to_out",
)
INITIAL_LORA_RANK = 32
INITIAL_LORA_ALPHA = 16
INITIAL_LORA_LEARNING_RATE = 1e-5


LORA_LEARNING_RATE_UPTO_256_STEPS = 1e-4
LORA_LEARNING_RATE_UPTO_512_STEPS = 5e-5
LORA_LEARNING_RATE_ABOVE_512_STEPS = 2e-5


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument(
        "--version",
        choices=("4b", "9b"),
        required=True,
        help="Model family to load.",
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=_positive_int, default=1500)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--sample_seed", type=int, default=19930625)
    parser.add_argument("--sample_indices", type=str, default=None)
    parser.add_argument("--text-quant-method", choices=CLI_QUANT_METHODS, default="sym-high")
    parser.add_argument("--denoiser-quant-method", choices=CLI_QUANT_METHODS, default="sym-low")
    parser.add_argument(
        "--trigger",
        type=str,
        help="Required for captionless datasets. Encoded once and kept on GPU for all training images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from flux2.loaders import load_flux2_dataset, load_flux2_denoiser, load_flux2_text_encoder
    from flux2.lora import TrainableLoraLinear, build_lora_state_dict, inject_trainable_lora_modules
    from flux2.gen import (
        DEFAULT_BASE_GUIDANCE_SCALE,
        _compute_empirical_mu,
        _encode_prompt_embeddings,
        _load_transformer_in_channels,
        _load_vae_scale_factor,
        _pack_latents,
        _prepare_image_ids,
        _patchify_latents,
        _prepare_latent_ids,
        _retrieve_latents,
        _retrieve_timesteps,
        _resolve_model_root,
        _unpack_latents_with_ids,
        _unpatchify_latents,
    )
    from PIL import Image
    from safetensors.torch import load_file as safe_load_file, save_file
    import torch
    from diffusers import AutoencoderKLFlux2
    from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    from transformers import Qwen2TokenizerFast

    device = torch.device("cuda")

    print("1. Load dataset into cpu ram ...")
    dataset_load_start = time.perf_counter()
    dataset = load_flux2_dataset(args.dataset)
    target_images = dataset["target_images"]
    text_prompts = dataset["text_prompts"]
    reference_images = dataset["reference_images"]
    if not (len(target_images) == len(text_prompts) == len(reference_images)):
        raise ValueError("Dataset loader returned misaligned target, prompt, and reference lists.")
    if args.samples == 0:
        sample_indices = []
    elif args.samples == 1:
        sample_indices = [1]
    elif args.sample_indices is None:
        sample_indices = [
            1 + round(index * (len(target_images) - 1) / (args.samples - 1))
            for index in range(args.samples)
        ]
    else:
        sample_indices = [int(index.strip()) for index in args.sample_indices.split(",") if index.strip()]
    invalid_sample_indices = [
        sample_index for sample_index in sample_indices if sample_index < 1 or sample_index > len(target_images)
    ]
    if invalid_sample_indices:
        invalid_text = ", ".join(str(sample_index) for sample_index in invalid_sample_indices)
        raise ValueError(f"Sample indices out of range for {len(target_images)} images: {invalid_text}")
    sample_resolutions: list[tuple[int, int]] = []
    for sample_index in sample_indices:
        with Image.open(target_images[sample_index - 1]) as sample_image:
            sample_resolutions.append(sample_image.size)

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

    images = [Image.open(image_path).convert("RGB") for image_path in target_images]
    reference_image_count = sum(len(sample_references) for sample_references in reference_images)
    print(f"  * dataset mode: {dataset_mode}")
    print(f"  * images: {len(images)}")
    print(f"  * reference images: {reference_image_count}")
    if trigger_text is not None:
        print(f"  * trigger: {trigger_text!r}")
    print(f"  * done in {time.perf_counter() - dataset_load_start:.3f}s")

    print("2. Load text encoder into gpu ...")
    text_encoder_load_start = time.perf_counter()
    resolved_version = args.version.strip().lower()
    model_root = _resolve_model_root(args.root, resolved_version)
    tokenizer_path = model_root / "tokenizer"
    text_encoder_path = model_root / "text_encoder"
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
    prompt_embeds = None
    text_ids = None
    shared_prompt_embeds = None
    shared_text_ids = None
    negative_prompt_embeds = None
    negative_text_ids = None
    if dataset_mode == "captioned":
        prompt_embeds_list = []
        text_ids_list = []
        for caption_index, caption in enumerate(captions, start=1):
            print(f"  * {caption_index} / {len(captions)} ...")
            encode_start = time.perf_counter()
            sample_prompt_embeds, sample_text_ids = _encode_prompt_embeddings(
                torch,
                tokenizer,
                text_encoder,
                prompt=caption,
                device=device,
                max_length=512,
            )
            torch.cuda.synchronize(device)
            prompt_embeds_list.append(sample_prompt_embeds.squeeze(0).cpu())
            text_ids_list.append(sample_text_ids.squeeze(0).cpu())
            print(f"    encode time: {time.perf_counter() - encode_start:.3f}s")

        prompt_embeds = torch.stack(prompt_embeds_list, dim=0)
        text_ids = torch.stack(text_ids_list, dim=0)
        print(f"  * cached caption encodings on cpu: {tuple(prompt_embeds.shape)}")
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
        print(f"  * shared trigger embeddings on gpu: {tuple(shared_prompt_embeds.shape)}")
        print(f"  * shared trigger text ids on gpu: {tuple(shared_text_ids.shape)}")

    if sample_indices:
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

    print("4. Load vae into gpu ...")
    vae_load_start = time.perf_counter()
    vae_path = model_root / "vae"
    vae_scale_factor = _load_vae_scale_factor(vae_path)
    vae = AutoencoderKLFlux2.from_pretrained(str(vae_path), local_files_only=True)
    vae = vae.to(device, dtype=torch.float16)
    torch.cuda.synchronize(device)
    print(f"  * done in {time.perf_counter() - vae_load_start:.3f}s")

    print("5. Prepare image latents ...")
    image_latent_prep_start = time.perf_counter()
    image_processor = Flux2ImageProcessor(vae_scale_factor=vae_scale_factor * 2)

    def normalize_image_latent_resolution(image_width: int, image_height: int) -> tuple[int, int]:
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

    def image_latent_resolution(image) -> tuple[int, int]:
        return normalize_image_latent_resolution(*image.size)

    for resolution_index, (sample_width, sample_height) in enumerate(sample_resolutions):
        sample_resolutions[resolution_index] = normalize_image_latent_resolution(sample_width, sample_height)
    if sample_resolutions:
        sample_resolution_text = ", ".join(
            f"{sample_index}={width}x{height}"
            for sample_index, (width, height) in zip(sample_indices, sample_resolutions)
        )
        print(f"  * sample resolutions: {sample_resolution_text}")

    def encode_image_latent(image) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_processor.check_image_input(image)

        image_width, image_height = image_latent_resolution(image)

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

        return _pack_latents(latent).squeeze(0).cpu(), _prepare_latent_ids(torch, latent).squeeze(0).cpu(), latent

    image_latents = []
    image_latent_ids = []
    for image_index, image in enumerate(images, start=1):
        print(f"  * {image_index} / {len(images)} ...")
        image_encode_start = time.perf_counter()
        image_latent, image_latent_id, _ = encode_image_latent(image)
        image_latents.append(image_latent)
        image_latent_ids.append(image_latent_id)
        torch.cuda.synchronize(device)
        print(f"    encode time: {time.perf_counter() - image_encode_start:.3f}s")

    reference_latents = []
    reference_latent_ids = []
    for sample_index, sample_references in enumerate(reference_images, start=1):
        if not sample_references:
            reference_latents.append(None)
            reference_latent_ids.append(None)
            continue

        print(f"  * sample {sample_index} references: {len(sample_references)} ...")
        reference_encode_start = time.perf_counter()
        encoded_reference_latents = []
        raw_reference_latents = []
        for reference_index, reference_path in enumerate(sample_references, start=1):
            print(f"    * reference {reference_index} / {len(sample_references)} ...")
            reference_image = Image.open(reference_path).convert("RGB")
            reference_latent, _, raw_reference_latent = encode_image_latent(reference_image)
            encoded_reference_latents.append(reference_latent)
            raw_reference_latents.append(raw_reference_latent)

        reference_latents.append(torch.cat(encoded_reference_latents, dim=0))
        reference_latent_ids.append(_prepare_image_ids(torch, raw_reference_latents).squeeze(0).cpu())
        torch.cuda.synchronize(device)
        print(f"    encode time: {time.perf_counter() - reference_encode_start:.3f}s")

    reference_latent_token_count = sum(latent.shape[0] for latent in reference_latents if latent is not None)
    print(f"  * cached reference latent tokens on cpu: {reference_latent_token_count}")

    del vae
    torch.cuda.empty_cache()
    print(f"  * done in {time.perf_counter() - image_latent_prep_start:.3f}s")

    print("6. Load denoiser ...")
    denoiser_load_start = time.perf_counter()
    transformer_path = model_root / "transformer"
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
    transformer.train()
    transformer.enable_gradient_checkpointing()
    print(f"  * initial lora targets: {', '.join(INITIAL_LORA_TARGET_LINEAR_NAMES)}")
    print(f"  * injected lora modules: {len(lora_module_names)}")
    print(f"  * lora rank/alpha: r={INITIAL_LORA_RANK}, alpha={INITIAL_LORA_ALPHA}")
    print(f"  * gradient checkpointing: {transformer.is_gradient_checkpointing}")
    trainable_params = sum(parameter.numel() for parameter in transformer.parameters() if parameter.requires_grad)
    print(f"  * trainable params: {trainable_params:,}")
    lora_parameters = [parameter for parameter in transformer.parameters() if parameter.requires_grad]
    print(f"  * lora parameter dtype: {lora_parameters[0].dtype}")
    optimizer = torch.optim.AdamW(lora_parameters, lr=INITIAL_LORA_LEARNING_RATE)
    print(f"  * optimizer: AdamW lr={INITIAL_LORA_LEARNING_RATE}")
    if shared_prompt_embeds is not None:
        shared_prompt_embeds = shared_prompt_embeds.to(dtype=transformer.dtype)
        print("  * using one shared trigger conditioning tensor on gpu")
    torch.cuda.synchronize(device)
    print(f"  * done in {time.perf_counter() - denoiser_load_start:.3f}s")

    output_dir = args.output
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_path = output_dir / "logs.csv"
    dataset_size = len(image_latents)

    def find_latest_checkpoint() -> tuple[int, Path] | None:
        checkpoints = sorted(
            models_dir.glob("epoch_*.safetensors"),
            key=lambda path: int(path.stem.removeprefix("epoch_")),
        )
        if not checkpoints:
            return None
        latest_checkpoint = checkpoints[-1]
        latest_epoch = int(latest_checkpoint.stem.removeprefix("epoch_"))
        return latest_epoch, latest_checkpoint

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

    def optimizer_checkpoint_path(epoch_number: int) -> Path:
        return models_dir / f"epoch_{epoch_number}.optimizer.pt"

    def load_optimizer_checkpoint(epoch_number: int) -> None:
        optimizer_path = optimizer_checkpoint_path(epoch_number)
        if not optimizer_path.is_file():
            raise FileNotFoundError(f"No optimizer checkpoint found for epoch {epoch_number}: {optimizer_path}")

        optimizer_checkpoint = torch.load(str(optimizer_path), map_location=device)
        if not isinstance(optimizer_checkpoint, dict) or "optimizer" not in optimizer_checkpoint:
            raise ValueError(f"Invalid optimizer checkpoint format: {optimizer_path}")

        saved_epoch = optimizer_checkpoint.get("epoch")
        if saved_epoch != epoch_number:
            raise ValueError(
                f"Optimizer checkpoint {optimizer_path} is for epoch {saved_epoch}, expected epoch {epoch_number}"
            )

        optimizer.load_state_dict(optimizer_checkpoint["optimizer"])
        print(f"  * loaded optimizer from {optimizer_path}")

    def trim_logs(max_epoch: int) -> None:
        if not logs_path.is_file():
            return

        retained_rows: list[list[str]] = []
        removed_rows = 0
        with logs_path.open("r", encoding="utf-8", newline="") as logs_handle:
            logs_reader = csv.reader(logs_handle)
            next(logs_reader, None)
            for row in logs_reader:
                if len(row) < 3:
                    continue

                try:
                    epoch_number = int(row[0])
                    loss_value = float(row[2])
                except ValueError:
                    continue

                if epoch_number <= max_epoch:
                    retained_rows.append([row[0], row[1], row[2]])
                else:
                    removed_rows += 1

        with logs_path.open("w", encoding="utf-8", newline="") as logs_handle:
            logs_writer = csv.writer(logs_handle)
            logs_writer.writerow(["epoch", "image", "loss"])
            logs_writer.writerows(retained_rows)

        if removed_rows:
            print(f"  * removed {removed_rows} log rows above epoch {max_epoch}")

    def prune_optimizer_checkpoints(latest_optimizer_path: Path) -> None:
        for optimizer_path in models_dir.glob("epoch_*.optimizer.pt"):
            if optimizer_path == latest_optimizer_path:
                continue

            optimizer_path.unlink()
            print(f"  * removed old optimizer checkpoint {optimizer_path}")

    def save_checkpoint(epoch_number: int) -> None:
        checkpoint_save_start = time.perf_counter()
        output_path = models_dir / f"epoch_{epoch_number}.safetensors"
        lora_state_dict = build_lora_state_dict(transformer)
        save_file(lora_state_dict, str(output_path))
        print(f"  * saved lora to {output_path}")

        optimizer_path = optimizer_checkpoint_path(epoch_number)
        temporary_optimizer_path = optimizer_path.with_name(f"{optimizer_path.name}.tmp")
        torch.save(
            {
                "epoch": epoch_number,
                "optimizer": optimizer.state_dict(),
            },
            str(temporary_optimizer_path),
        )
        temporary_optimizer_path.replace(optimizer_path)
        print(f"  * saved optimizer to {optimizer_path}")
        prune_optimizer_checkpoints(optimizer_path)
        print(f"  * checkpoint save time: {time.perf_counter() - checkpoint_save_start:.3f}s")

    def render_samples(epoch_number: int) -> None:
        if not sample_indices:
            return

        sample_start = time.perf_counter()
        sample_dir = output_dir / "sample"
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample_steps = 25

        print(f"  * rendering {len(sample_indices)} sample(s) ...")
        scheduler_path = model_root / "scheduler"
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(scheduler_path), local_files_only=True)
        num_channels_latents = _load_transformer_in_channels(transformer_path) // 4

        sigmas = torch.linspace(1.0, 1 / sample_steps, sample_steps, dtype=torch.float32).tolist()
        if hasattr(scheduler.config, "use_flow_sigmas") and scheduler.config.use_flow_sigmas:
            sigmas = None

        was_training = transformer.training
        transformer.eval()
        sample_latents = []
        try:
            with torch.inference_mode():
                for sample_index, (sample_width, sample_height) in zip(sample_indices, sample_resolutions):
                    dataset_index = sample_index - 1
                    print(f"    * denoise sample {sample_index} at {sample_width}x{sample_height} ...")
                    if prompt_embeds is not None and text_ids is not None:
                        prompt_embeds_batch = prompt_embeds[dataset_index : dataset_index + 1].to(
                            device=device,
                            dtype=transformer.dtype,
                        )
                        text_ids_batch = text_ids[dataset_index : dataset_index + 1].to(device=device)
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
                    if reference_latents[dataset_index] is not None and reference_latent_ids[dataset_index] is not None:
                        reference_latents_batch = reference_latents[dataset_index].unsqueeze(0).to(
                            device=device,
                            dtype=latents.dtype,
                        )
                        reference_latent_ids_batch = reference_latent_ids[dataset_index].unsqueeze(0).to(device=device)

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
        finally:
            transformer.to("cpu")
            torch.cuda.empty_cache()

        print("    * decoding samples ...")
        vae = AutoencoderKLFlux2.from_pretrained(str(vae_path), local_files_only=True)
        vae = vae.to(device, dtype=torch.float16)
        image_processor = Flux2ImageProcessor(vae_scale_factor=vae_scale_factor * 2)
        with torch.inference_mode():
            for sample_index, latents_cpu, latent_ids_cpu in sample_latents:
                output_path = sample_dir / f"epoch_{epoch_number}_{sample_index}.png"
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

        del vae
        torch.cuda.empty_cache()
        transformer.to(device)
        if was_training:
            transformer.train()
        else:
            transformer.eval()
        torch.cuda.synchronize(device)
        print(f"  * sample render time: {time.perf_counter() - sample_start:.3f}s")

    print("7. Run training epochs ...")
    training_start = time.perf_counter()
    print(f"  * logs: {logs_path}")
    print(f"  * max steps: {args.steps}")
    steps_done = 0
    start_epoch = 1
    logs_mode = "w"
    if args.resume:
        latest_checkpoint = find_latest_checkpoint()
        if latest_checkpoint is None:
            raise FileNotFoundError(f"No checkpoints found in {models_dir}")

        latest_epoch, checkpoint_path = latest_checkpoint
        print(f"  * resuming from {checkpoint_path}")
        load_checkpoint(checkpoint_path)
        load_optimizer_checkpoint(latest_epoch)
        trim_logs(latest_epoch)
        steps_done = latest_epoch * dataset_size
        start_epoch = latest_epoch + 1
        logs_mode = "a"
        print(f"  * resume epoch: {latest_epoch}")
        print(f"  * resumed steps: {steps_done}")

    with logs_path.open(logs_mode, encoding="utf-8", newline="") as logs_handle:
        logs_writer = csv.writer(logs_handle)
        if logs_mode == "w":
            logs_writer.writerow(["epoch", "image", "loss"])

        if steps_done > args.steps:
            print(f"  * stopping before training: resumed steps {steps_done} already exceed limit {args.steps}")

        epoch_number = start_epoch
        while steps_done <= args.steps:
            epoch_start = time.perf_counter()
            print(f"  * epoch {epoch_number} ...")
            for sample_index in range(dataset_size):
                sample_start = time.perf_counter()
                print(f"    * sample {sample_index + 1} / {dataset_size} ...")
                optimizer.zero_grad(set_to_none=True)

                if prompt_embeds is not None and text_ids is not None:
                    prompt_embeds_batch = prompt_embeds[sample_index : sample_index + 1].to(
                        device=device,
                        dtype=transformer.dtype,
                    )
                    text_ids_batch = text_ids[sample_index : sample_index + 1].to(device=device)
                else:
                    prompt_embeds_batch = shared_prompt_embeds
                    text_ids_batch = shared_text_ids
                image_latents_batch = image_latents[sample_index].unsqueeze(0).to(device=device)
                image_latent_ids_batch = image_latent_ids[sample_index].unsqueeze(0).to(device=device)
                reference_latents_batch = None
                reference_latent_ids_batch = None
                if reference_latents[sample_index] is not None and reference_latent_ids[sample_index] is not None:
                    reference_latents_batch = reference_latents[sample_index].unsqueeze(0).to(
                        device=device,
                        dtype=image_latents_batch.dtype,
                    )
                    reference_latent_ids_batch = reference_latent_ids[sample_index].unsqueeze(0).to(device=device)

                noise = torch.randn_like(image_latents_batch)
                timestep = torch.rand((1,), device=device, dtype=image_latents_batch.dtype) * 1000.0
                sigma = (timestep / 1000.0).view(-1, 1, 1)

                noisy_latents = (1.0 - sigma) * image_latents_batch + sigma * noise
                target = noise - image_latents_batch
                model_hidden_states = noisy_latents
                model_img_ids = image_latent_ids_batch
                if reference_latents_batch is not None and reference_latent_ids_batch is not None:
                    model_hidden_states = torch.cat([noisy_latents, reference_latents_batch], dim=1)
                    model_img_ids = torch.cat([image_latent_ids_batch, reference_latent_ids_batch], dim=1)

                torch.cuda.synchronize(device)
                noise_pred = transformer(
                    hidden_states=model_hidden_states.to(dtype=transformer.dtype),
                    timestep=timestep / 1000,
                    guidance=None,
                    encoder_hidden_states=prompt_embeds_batch,
                    txt_ids=text_ids_batch,
                    img_ids=model_img_ids,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred[:, : image_latents_batch.size(1)]
                torch.cuda.synchronize(device)

                loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float())
                loss_value = loss.item()
                print(f"      loss: {loss_value:.6f}")
                logs_writer.writerow([epoch_number, sample_index + 1, loss_value])
                logs_handle.flush()

                loss.backward()
                torch.cuda.synchronize(device)

                optimizer.step()
                torch.cuda.synchronize(device)
                steps_done += 1
                print(f"      done in {time.perf_counter() - sample_start:.3f}s")

                del (
                    prompt_embeds_batch,
                    text_ids_batch,
                    image_latents_batch,
                    image_latent_ids_batch,
                    reference_latents_batch,
                    reference_latent_ids_batch,
                    noise,
                    timestep,
                    sigma,
                    noisy_latents,
                    model_hidden_states,
                    model_img_ids,
                    target,
                    noise_pred,
                    loss,
                )

            print(f"  * total steps: {steps_done}")
            print(f"  * epoch done in {time.perf_counter() - epoch_start:.3f}s")
            save_checkpoint(epoch_number)
            render_samples(epoch_number)
            if steps_done > args.steps:
                print(f"  * stopping after epoch {epoch_number}: steps {steps_done} exceeded limit {args.steps}")
                break
            epoch_number += 1

    print(f"  * training done in {time.perf_counter() - training_start:.3f}s")
    del transformer


if __name__ == "__main__":
    main()
