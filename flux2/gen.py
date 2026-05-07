#!/usr/bin/env python
"""
Unified Flux 2 Klein generation entrypoint.
"""

from __future__ import annotations

import argparse
import inspect
import json
import time
from pathlib import Path

DEFAULT_DISTILLED_STEPS = 4
DEFAULT_BASE_STEPS = 50
DEFAULT_SEED = 19930625
DEFAULT_DISTILLED_GUIDANCE_SCALE = 1.0
DEFAULT_BASE_GUIDANCE_SCALE = 4.0
DEFAULT_TEXT_ENCODER_OUT_LAYERS = (9, 18, 27)
_MODEL_DIRS = {
    "4b": "flux2_4b",
    "9b": "flux2_9b",
}


def _split_image_paths(image: str | Path) -> list[str]:
    return [part.strip() for part in str(image).split(",") if part.strip()]


def _load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_vae_scale_factor(vae_path: str | Path) -> int:
    config = _load_json(Path(vae_path) / "config.json")
    return 2 ** (len(config["block_out_channels"]) - 1)


def _load_transformer_in_channels(transformer_path: str | Path) -> int:
    config = _load_json(Path(transformer_path) / "config.json")
    return int(config["in_channels"])


def _load_transformer_joint_attention_dim(transformer_path: str | Path) -> int:
    config = _load_json(Path(transformer_path) / "config.json")
    return int(config["joint_attention_dim"])


def _resolve_version_dir(version: str) -> str:
    resolved_version = version.strip().lower()
    if resolved_version not in _MODEL_DIRS:
        raise ValueError(f"Unsupported version {version!r}. Expected one of: {', '.join(sorted(_MODEL_DIRS))}")
    return _MODEL_DIRS[resolved_version]


def _resolve_model_root(root: str | Path, version: str) -> Path:
    return Path(root).expanduser().resolve() / _resolve_version_dir(version) / "model"


def _resolve_output_dir(root: str | Path, version: str) -> Path:
    return Path(root).expanduser().resolve() / _resolve_version_dir(version) / "output"


def _parse_loras_arg(value: str | None) -> dict[str, float] | None:
    if value is None:
        return None

    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        return None

    loras: dict[str, float] = {}
    for item in items:
        path_text, sep, strength_text = item.rpartition(":")
        if not sep:
            path_text = item
            strength = 1.0
        else:
            if not path_text:
                raise ValueError(f"Invalid LoRA entry: {item!r}")
            try:
                strength = float(strength_text)
            except ValueError as exc:
                raise ValueError(f"Invalid LoRA strength in entry: {item!r}") from exc

        path_text = path_text.strip()
        if not path_text:
            raise ValueError(f"Invalid LoRA entry: {item!r}")

        loras[path_text] = strength

    return loras


def _retrieve_latents(encoder_output):
    if hasattr(encoder_output, "latent_dist"):
        return encoder_output.latent_dist.mode()
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not access latents of provided encoder_output")


def _resolve_inference_settings(
    *,
    base: bool,
    num_inference_steps: int | None,
    guidance_scale: float | None,
) -> tuple[bool, int, float]:
    if base:
        return (
            True,
            DEFAULT_BASE_STEPS if num_inference_steps is None else num_inference_steps,
            DEFAULT_BASE_GUIDANCE_SCALE if guidance_scale is None else guidance_scale,
        )
    return (
        False,
        DEFAULT_DISTILLED_STEPS if num_inference_steps is None else num_inference_steps,
        DEFAULT_DISTILLED_GUIDANCE_SCALE if guidance_scale is None else guidance_scale,
    )


def _resolve_transformer_variant(*, base: bool) -> str:
    return "base" if base else "distill"


def _encode_prompt_embeddings(
    torch_module,
    tokenizer,
    text_encoder,
    *,
    prompt: str,
    device,
    max_length: int,
) -> tuple[object, object]:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch_module.inference_mode():
        encoder_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "output_hidden_states": True,
            "use_cache": False,
        }
        if hasattr(text_encoder, "lm_head"):
            encoder_kwargs["compute_logits"] = False
        output = text_encoder(**encoder_kwargs)

    hidden_states = torch_module.stack(
        [output.hidden_states[index] for index in DEFAULT_TEXT_ENCODER_OUT_LAYERS],
        dim=1,
    )
    prompt_embeds = hidden_states.permute(0, 2, 1, 3).reshape(hidden_states.shape[0], hidden_states.shape[2], -1)
    text_ids = _prepare_text_ids(torch_module, prompt_embeds).to(device)
    return prompt_embeds, text_ids


def _validate_text_encoder_layers(text_encoder) -> None:
    hidden_layer_count = int(getattr(text_encoder.config, "num_hidden_layers", 0))
    if hidden_layer_count <= 0:
        raise ValueError("Text encoder config is missing num_hidden_layers.")

    hidden_state_count = hidden_layer_count + 1
    invalid_layers = [
        index for index in DEFAULT_TEXT_ENCODER_OUT_LAYERS if index < 0 or index >= hidden_state_count
    ]
    if invalid_layers:
        raise ValueError(
            "text_encoder_out_layers contains out-of-range indices "
            f"for a model with {hidden_state_count} hidden-state outputs: {invalid_layers}"
        )


def _validate_joint_attention_dim(
    *,
    text_encoder,
    transformer_path: str | Path,
) -> None:
    hidden_size = int(getattr(text_encoder.config, "hidden_size", 0))
    if hidden_size <= 0:
        raise ValueError("Text encoder config is missing hidden_size.")

    expected_joint_attention_dim = hidden_size * len(DEFAULT_TEXT_ENCODER_OUT_LAYERS)
    actual_joint_attention_dim = _load_transformer_joint_attention_dim(transformer_path)
    if expected_joint_attention_dim != actual_joint_attention_dim:
        raise ValueError(
            "Text encoder output width does not match denoiser joint_attention_dim: "
            f"{expected_joint_attention_dim} != {actual_joint_attention_dim}. "
            f"hidden_size={hidden_size}, layers={DEFAULT_TEXT_ENCODER_OUT_LAYERS}, transformer={transformer_path}"
        )


def _resize_to_max_side(pil_image, max_side: int):
    width, height = pil_image.size
    longest_side = max(width, height)
    if longest_side <= max_side:
        return pil_image

    scale = max_side / float(longest_side)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    from PIL import Image

    return pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def _patchify_latents(latents):
    batch_size, num_channels_latents, height, width = latents.shape
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    latents = latents.reshape(batch_size, num_channels_latents * 4, height // 2, width // 2)
    return latents


def _pack_latents(latents):
    batch_size, num_channels, height, width = latents.shape
    return latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)


def _unpack_latents_with_ids(torch_module, packed_latents, latent_ids):
    unpacked = []
    for packed, positions in zip(packed_latents, latent_ids):
        _, channels = packed.shape
        h_ids = positions[:, 1].to(torch_module.int64)
        w_ids = positions[:, 2].to(torch_module.int64)

        height = torch_module.max(h_ids) + 1
        width = torch_module.max(w_ids) + 1
        flat_ids = h_ids * width + w_ids

        output = torch_module.zeros((height * width, channels), device=packed.device, dtype=packed.dtype)
        output.scatter_(0, flat_ids.unsqueeze(1).expand(-1, channels), packed)
        unpacked.append(output.view(height, width, channels).permute(2, 0, 1))

    return torch_module.stack(unpacked, dim=0)


def _unpatchify_latents(latents):
    batch_size, num_channels_latents, height, width = latents.shape
    latents = latents.reshape(batch_size, num_channels_latents // 4, 2, 2, height, width)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    latents = latents.reshape(batch_size, num_channels_latents // 4, height * 2, width * 2)
    return latents


def _prepare_latent_ids(torch_module, latents):
    batch_size, _, height, width = latents.shape
    latent_ids = torch_module.cartesian_prod(
        torch_module.arange(1),
        torch_module.arange(height),
        torch_module.arange(width),
        torch_module.arange(1),
    )
    return latent_ids.unsqueeze(0).expand(batch_size, -1, -1)


def _prepare_image_ids(torch_module, image_latents: list, scale: int = 10):
    t_coords = [scale + scale * t for t in torch_module.arange(0, len(image_latents))]
    t_coords = [t.view(-1) for t in t_coords]

    image_latent_ids = []
    for latent, t in zip(image_latents, t_coords):
        latent = latent.squeeze(0)
        _, height, width = latent.shape
        coords = torch_module.cartesian_prod(
            t,
            torch_module.arange(height),
            torch_module.arange(width),
            torch_module.arange(1),
        )
        image_latent_ids.append(coords)

    return torch_module.cat(image_latent_ids, dim=0).unsqueeze(0)


def _prepare_text_ids(torch_module, prompt_embeds, t_coord=None):
    batch_size, seq_len, _ = prompt_embeds.shape
    out_ids = []

    for index in range(batch_size):
        t = torch_module.arange(1) if t_coord is None else t_coord[index]
        h = torch_module.arange(1)
        w = torch_module.arange(1)
        l = torch_module.arange(seq_len)
        out_ids.append(torch_module.cartesian_prod(t, h, w, l))

    return torch_module.stack(out_ids)


def _compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


def _retrieve_timesteps(
    scheduler,
    num_inference_steps,
    *,
    device,
    timesteps=None,
    sigmas=None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(f"{scheduler.__class__} does not support custom timesteps.")
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accepts_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_sigmas:
            raise ValueError(f"{scheduler.__class__} does not support custom sigmas.")
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps

    return timesteps, num_inference_steps


def generate_image(
    root: str | Path,
    *,
    version: str,
    image: str | Path | None = None,
    prompt: str = "A cat holding a sign that says hello world",
    width: int = 512,
    height: int = 512,
    num_inference_steps: int | None = None,
    seed: int = DEFAULT_SEED,
    guidance_scale: float | None = None,
    base: bool = False,
    ref_size: int = 512,
    text_quant_method: str | None = None,
    denoiser_quant_method: str | None = None,
    loras: dict[str, float] | None = None,
    max_length: int = 512,
) -> None:
    from flux2.denoiser.loader import _load_flux2_denoiser as load_flux2_denoiser
    from flux2.text_encoder.loader import _load_flux2_text_encoder as load_flux2_text_encoder
    from flux2.lora import apply_lora

    print("1. Text Encoding")
    text_encoding_start = time.perf_counter()
    import torch
    from transformers import Qwen2TokenizerFast

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is not available in this environment.")

    resolved_version = version.strip().lower()
    model_root = Path(root) / f"flux2_{resolved_version}" / "model"
    tokenizer_path = model_root / "tokenizer"
    text_encoder_path = model_root / "text_encoder"
    vae_path = model_root / "vae"
    scheduler_path = model_root / "scheduler"
    transformer_path = model_root / "transformer"

    is_base, num_inference_steps, guidance_scale = _resolve_inference_settings(
        base=base,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )
    transformer_variant = _resolve_transformer_variant(base=is_base)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    latent_generator = torch.Generator(device=device)
    latent_generator.manual_seed(seed)

    print("  * loading tokenizer ...")
    tokenizer = Qwen2TokenizerFast.from_pretrained(str(tokenizer_path))

    print("  * loading text encoder ...")
    print(f"    quantization: {text_quant_method or 'none'}")
    torch.cuda.empty_cache()
    text_encoder = load_flux2_text_encoder(
        str(text_encoder_path),
        quant_method=text_quant_method,
    )
    _validate_text_encoder_layers(text_encoder)
    _validate_joint_attention_dim(
        text_encoder=text_encoder,
        transformer_path=transformer_path,
    )
    text_encoder = text_encoder.to(device)

    print("  * text encoding ...")
    torch.cuda.synchronize(device)
    encode_start = time.perf_counter()
    prompt_embeds, text_ids = _encode_prompt_embeddings(
        torch,
        tokenizer,
        text_encoder,
        prompt=prompt,
        device=device,
        max_length=max_length,
    )
    negative_prompt_embeds = None
    negative_text_ids = None
    if is_base:
        negative_prompt_embeds, negative_text_ids = _encode_prompt_embeddings(
            torch,
            tokenizer,
            text_encoder,
            prompt="",
            device=device,
            max_length=max_length,
        )
    torch.cuda.synchronize(device)
    encode_seconds = time.perf_counter() - encode_start
    print(f"    encode time: {encode_seconds:.3f}s")

    print("  * clearing encoder ...")
    del tokenizer
    del text_encoder
    torch.cuda.empty_cache()
    text_encoding_seconds = time.perf_counter() - text_encoding_start
    print(f"  * done in {text_encoding_seconds:.3f}s")

    batch_size = prompt_embeds.shape[0]
    vae_scale_factor = _load_vae_scale_factor(vae_path)
    image_latents = None
    image_latent_ids = None
    vae = None
    scheduler = None

    print("2. Image Encoding")
    if image:
        image_encoding_start = time.perf_counter()
        from diffusers import AutoencoderKLFlux2
        from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
        from PIL import Image

        print("  * loading vae ...")
        vae = AutoencoderKLFlux2.from_pretrained(str(vae_path), local_files_only=True)
        vae = vae.to(device, dtype=torch.float16)

        print("  * loading image processor ...")
        image_processor = Flux2ImageProcessor(vae_scale_factor=vae_scale_factor * 2)

        image_paths = _split_image_paths(image)
        encoded_image_latents = []

        for image_path in image_paths:
            pil_image = Image.open(image_path).convert("RGB")
            image_processor.check_image_input(pil_image)
            pil_image = _resize_to_max_side(pil_image, ref_size)

            image_width, image_height = pil_image.size

            multiple_of = vae_scale_factor * 2
            image_width = (image_width // multiple_of) * multiple_of
            image_height = (image_height // multiple_of) * multiple_of

            image_tensor = image_processor.preprocess(pil_image, height=image_height, width=image_width, resize_mode="crop")
            image_tensor = image_tensor.to(device=device, dtype=torch.float16)

            latent = _retrieve_latents(vae.encode(image_tensor))
            latent = _patchify_latents(latent)

            latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latent.device, latent.dtype)
            latents_bn_std = torch.sqrt(
                vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
            ).to(latent.device, latent.dtype)
            latent = (latent - latents_bn_mean) / latents_bn_std
            encoded_image_latents.append(latent)

        image_latent_ids = _prepare_image_ids(torch, encoded_image_latents).to(device)
        image_latents = torch.cat([_pack_latents(latent).squeeze(0) for latent in encoded_image_latents], dim=0)
        image_latents = image_latents.unsqueeze(0).repeat(batch_size, 1, 1)
        image_latent_ids = image_latent_ids.repeat(batch_size, 1, 1)

        print(f"    image latent tokens: {tuple(image_latents.shape)}")
        print(f"    image latent ids: {tuple(image_latent_ids.shape)}")
        print(f"    conditioned size: {width}x{height}")

        image_encoding_seconds = time.perf_counter() - image_encoding_start
        print(f"  * done in {image_encoding_seconds:.3f}s")
    else:
        print("  * skipped")

    print("3. Prepare latent")
    latent_prep_start = time.perf_counter()
    num_channels_latents = _load_transformer_in_channels(transformer_path) // 4

    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))
    latent_shape = (batch_size, num_channels_latents * 4, height // 2, width // 2)

    latents = torch.randn(latent_shape, device=device, dtype=prompt_embeds.dtype, generator=latent_generator)
    latent_ids = _prepare_latent_ids(torch, latents).to(device)
    latents = _pack_latents(latents)

    latent_prep_seconds = time.perf_counter() - latent_prep_start
    print(f"  * done in {latent_prep_seconds:.3f}s")

    print("4. Prepare timesteps")
    timestep_prep_start = time.perf_counter()
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

    print("  * loading scheduler ...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(scheduler_path), local_files_only=True)

    sigmas = torch.linspace(1.0, 1 / num_inference_steps, num_inference_steps, dtype=torch.float32).tolist()
    if hasattr(scheduler.config, "use_flow_sigmas") and scheduler.config.use_flow_sigmas:
        sigmas = None

    image_seq_len = latents.shape[1]
    mu = _compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
    timesteps, num_inference_steps = _retrieve_timesteps(
        scheduler,
        num_inference_steps,
        device=device,
        sigmas=sigmas,
        mu=mu,
    )

    timestep_prep_seconds = time.perf_counter() - timestep_prep_start
    print(f"  * done in {timestep_prep_seconds:.3f}s")

    print("5. Denoise")
    denoise_start = time.perf_counter()
    print("  * loading transformer ...")
    print(f"    variant: {transformer_variant}")
    print(f"    quantization: {denoiser_quant_method or 'none'}")
    torch.cuda.empty_cache()
    transformer = load_flux2_denoiser(
        str(transformer_path),
        quant_method=denoiser_quant_method,
        variant=transformer_variant,
        version=resolved_version,
    )
    transformer = transformer.to(device)
    if loras:
        print("  * applying loras ...")
        apply_lora(transformer, loras)
    prompt_embeds = prompt_embeds.to(device=device, dtype=transformer.dtype)
    if negative_prompt_embeds is not None:
        negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=transformer.dtype)

    scheduler.set_begin_index(0)
    torch.cuda.synchronize(device)
    with torch.inference_mode():
        for step_index, timestep_value in enumerate(timesteps):
            print(f"  * {step_index + 1}/{num_inference_steps} steps", flush=True)
            step_start = time.perf_counter()

            timestep = timestep_value.expand(latents.shape[0]).to(latents.dtype)
            latent_model_input = latents.to(transformer.dtype)
            latent_image_ids = latent_ids

            if image_latents is not None:
                latent_model_input = torch.cat([latents, image_latents], dim=1).to(transformer.dtype)
                latent_image_ids = torch.cat([latent_ids, image_latent_ids], dim=1)

            with transformer.cache_context("cond"):
                noise_pred = transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=None,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]

            noise_pred = noise_pred[:, : latents.size(1) :]
            if is_base:
                with transformer.cache_context("uncond"):
                    neg_noise_pred = transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=None,
                        encoder_hidden_states=negative_prompt_embeds,
                        txt_ids=negative_text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=None,
                        return_dict=False,
                    )[0]
                neg_noise_pred = neg_noise_pred[:, : latents.size(1) :]
                noise_pred = neg_noise_pred + guidance_scale * (noise_pred - neg_noise_pred)

            latents_dtype = latents.dtype
            latents = scheduler.step(noise_pred, timestep_value, latents, return_dict=False)[0]
            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)
            torch.cuda.synchronize(device)
            step_seconds = time.perf_counter() - step_start
            print(f"    * done in {step_seconds:.3f}s", flush=True)

    torch.cuda.synchronize(device)
    print("  * clearing transformer ...")
    del transformer
    torch.cuda.empty_cache()
    denoise_seconds = time.perf_counter() - denoise_start
    print(f"  * done in {denoise_seconds:.3f}s")

    print("6. VAE")
    vae_start = time.perf_counter()
    from diffusers import AutoencoderKLFlux2
    from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor

    if vae is None:
        print("  * loading vae ...")
        vae = AutoencoderKLFlux2.from_pretrained(str(vae_path), local_files_only=True)
        vae = vae.to(device, dtype=torch.float16)

    image_processor = Flux2ImageProcessor(vae_scale_factor=vae_scale_factor * 2)

    torch.cuda.synchronize(device)
    with torch.inference_mode():
        latents = _unpack_latents_with_ids(torch, latents, latent_ids)

        latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(
            latents.device, latents.dtype
        )
        latents = latents * latents_bn_std + latents_bn_mean
        latents = _unpatchify_latents(latents)
        decoded = vae.decode(latents.to(dtype=vae.dtype), return_dict=False)[0]
        image = image_processor.postprocess(decoded, output_type="pil")[0]
    torch.cuda.synchronize(device)

    vae_seconds = time.perf_counter() - vae_start
    print(f"  * done in {vae_seconds:.3f}s")

    print("7. Saving")
    saving_start = time.perf_counter()
    output_dir = Path(root) / f"flux2_{version}" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}.png"
    image.save(output_path)
    saving_seconds = time.perf_counter() - saving_start
    print(f"  * saved to {output_path}")
    print(f"  * done in {saving_seconds:.3f}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified Flux 2 Klein generation flow.")
    parser.add_argument(
        "--root",
        required=True,
        help="Root folder that contains flux_2_klein_4b/model or flux2_9b/model.",
    )
    parser.add_argument(
        "--version",
        choices=("4b", "9b"),
        required=True,
        help="Model family to load.",
    )
    parser.add_argument(
        "--images",
        help='Optional comma-separated image paths to encode, for example "img1.png, img2.png".',
    )
    parser.add_argument(
        "--prompt",
        default="A cat holding a sign that says hello world",
        help="Prompt text to encode.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Output image width. Defaults to 384.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Output image height. Defaults to 768.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        help="Number of denoising steps. Defaults to 4 for distilled mode or 50 with --base.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Seed for latent noise initialization. Default: {DEFAULT_SEED}.",
    )
    parser.add_argument(
        "--base",
        action="store_true",
        help="Use base-model inference defaults and CFG-style denoising instead of distilled defaults.",
    )
    parser.add_argument(
        "--ref-size",
        type=int,
        default=512,
        help="Maximum longest side for reference images before encoding.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        help="Guidance scale override for --base CFG mixing. Ignored in distilled mode.",
    )
    parser.add_argument(
        "--text-quant-method",
        default="sym-high",
        help='Text encoder quantization method. "none" keeps checkpoint weights as loaded.',
    )
    parser.add_argument(
        "--denoiser-quant-method",
        default="sym-med",
        help='Denoiser quantization method. Use "none" to load the fp16 checkpoint directly.',
    )
    parser.add_argument(
        "--loras",
        help='Optional comma-separated LoRA list in the form "path:strength, path2:strength2".',
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum tokenizer sequence length.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_image(
        args.root,
        version=args.version,
        image=args.images,
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        seed=args.seed,
        guidance_scale=args.guidance_scale,
        base=args.base,
        ref_size=args.ref_size,
        text_quant_method=None if args.text_quant_method == "none" else args.text_quant_method,
        denoiser_quant_method=None if args.denoiser_quant_method == "none" else args.denoiser_quant_method,
        loras=_parse_loras_arg(args.loras),
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
