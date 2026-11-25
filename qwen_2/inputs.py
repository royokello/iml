from __future__ import annotations

from collections.abc import Sequence

import torch
from PIL import Image

from qwen_2.config import (
    LATENT_CHANNELS,
    TEXT_SYSTEM_PROMPT,
    TEXT_TEMPLATE_T2I,
    TURBO_SIGMAS,
    VAE_SCALE_FACTOR,
    build_edit_text,
)

__all__ = [
    "build_img_mask",
    "build_img_shapes",
    "build_sigmas",
    "decode_latents",
    "encode_prompt",
    "encode_reference_latents",
    "make_noise",
    "pack_latents",
    "unpack_latents",
]


def build_sigmas(steps: int, variant: str) -> torch.Tensor:
    """Build the noise schedule for a variant.

    Args:
        steps: The number of denoising steps, used by the base variant.
        variant: `turbo` or `base`.

    Returns:
        The sigma values, one per step.

    Raises:
        ValueError: If the variant is unknown.
    """
    if variant == "turbo":
        return torch.tensor(TURBO_SIGMAS, dtype=torch.float32)
    if variant == "base":
        return torch.linspace(1.0, 1.0 / steps, steps, dtype=torch.float32)
    raise ValueError(f"Unknown variant {variant!r}. Expected 'turbo' or 'base'.")


def build_img_mask(
    sequence_length: int,
    target_tokens: int,
    image_pad_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the joint-sequence image mask.

    The first `sequence_length` entries describe the prompt. Text-to-image has
    no image tokens there, so they are all False. With a reference image the
    encoder reports which prompt positions hold `<|image_pad|>` slots, and that
    mask is used instead. The appended entries mark the target image, one slot
    per 2x2 group of latent tokens.

    Args:
        sequence_length: The number of prompt tokens.
        target_tokens: The number of target latent tokens.
        image_pad_mask: A bool tensor of shape `(1, sequence_length)` that marks
            the prompt positions holding reference image slots. None means no
            reference image.

    Returns:
        A bool tensor of shape `(1, sequence_length + target_tokens // 4)`.

    Raises:
        ValueError: If `target_tokens` is not a multiple of 4, or if
            `image_pad_mask` does not have shape `(1, sequence_length)`.
    """
    if target_tokens % 4 != 0:
        raise ValueError(f"target_tokens={target_tokens} must be a multiple of 4.")
    if image_pad_mask is None:
        prompt = torch.zeros(1, sequence_length, dtype=torch.bool)
    else:
        if tuple(image_pad_mask.shape) != (1, sequence_length):
            raise ValueError(
                f"image_pad_mask shape {tuple(image_pad_mask.shape)} does not match "
                f"sequence_length={sequence_length}."
            )
        prompt = image_pad_mask.bool()
    target = torch.ones(1, target_tokens // 4, dtype=torch.bool)
    return torch.cat([prompt, target], dim=1)


def build_img_shapes(
    height: int,
    width: int,
    ref_sizes: Sequence[tuple[int, int]] = (),
) -> list[list[tuple[int, int, int]]]:
    """Build the per-sample image layout in latent tokens.

    Args:
        height: The output height in pixels.
        width: The output width in pixels.
        ref_sizes: The `(width, height)` of each reference image, in order. One
            block is emitted per reference before the target block.

    Returns:
        `[[(1, ref_h // 16, ref_w // 16), ..., (1, height // 16, width // 16)]]`.

    Raises:
        ValueError: If a side is not a positive multiple of 16.
    """
    sizes = [*ref_sizes, (width, height)]
    blocks: list[tuple[int, int, int]] = []
    for block_width, block_height in sizes:
        if (
            block_height <= 0
            or block_width <= 0
            or block_height % VAE_SCALE_FACTOR
            or block_width % VAE_SCALE_FACTOR
        ):
            raise ValueError(
                f"size {(block_width, block_height)} must be positive multiples of "
                f"{VAE_SCALE_FACTOR}."
            )
        blocks.append(
            (1, block_height // VAE_SCALE_FACTOR, block_width // VAE_SCALE_FACTOR)
        )
    return [blocks]


def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """Flatten a latent tensor into tokens.

    Args:
        latents: A tensor of shape `(B, C, H, W)` or `(B, C, 1, H, W)`.

    Returns:
        A tensor of shape `(B, H * W, C)`.
    """
    if latents.ndim == 5:
        batch, channels, frames, height, width = latents.shape
        if frames != 1:
            raise ValueError(f"Expected a single frame, got {frames}.")
        latents = latents[:, :, 0]
    elif latents.ndim != 4:
        raise ValueError(f"Expected 4 or 5 dimensions, got {latents.ndim}.")
    batch, channels, height, width = latents.shape
    return latents.reshape(batch, channels, height * width).transpose(1, 2)


def unpack_latents(latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Restore tokens into a latent tensor.

    Args:
        latents: A tensor of shape `(B, tokens, C)`.
        height: The output height in pixels.
        width: The output width in pixels.

    Returns:
        A tensor of shape `(B, C, 1, height // 16, width // 16)`.
    """
    batch, _, channels = latents.shape
    latent_height = height // VAE_SCALE_FACTOR
    latent_width = width // VAE_SCALE_FACTOR
    return latents.transpose(1, 2).reshape(batch, channels, 1, latent_height, latent_width)


def make_noise(
    seed: int,
    height: int,
    width: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample the initial packed noise.

    Args:
        seed: The random seed.
        height: The output height in pixels.
        width: The output width in pixels.
        device: The device that holds the generator and the tensor.
        dtype: The tensor dtype.

    Returns:
        A tensor of shape `(1, (height // 16) * (width // 16), 64)`.
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(
        1,
        LATENT_CHANNELS,
        height // VAE_SCALE_FACTOR,
        width // VAE_SCALE_FACTOR,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    return pack_latents(noise)


def _system_prefix_length(processor) -> int:
    system_message = [{"role": "system", "content": [{"type": "text", "text": TEXT_SYSTEM_PROMPT}]}]
    tokens = processor.apply_chat_template(system_message, tokenize=True, return_dict=False)
    if tokens and isinstance(tokens[0], int):
        return len(tokens)
    return len(tokens[0])


def encode_prompt(
    processor,
    text_encoder,
    prompt: str,
    *,
    device: torch.device | str,
    images: Sequence[Image.Image] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode a text prompt for the denoiser, with optional reference images.

    The hidden states come from the last decoder layer before the final RMS
    norm, which is what the denoiser was trained on.

    Args:
        processor: The `Qwen3VLProcessor`.
        text_encoder: The text encoder, text-only or full. A run with reference
            images needs the vision tower, so pass the `include_vision=True`
            model.
        prompt: The user prompt.
        device: The device that holds the text encoder.
        images: The reference images. One or more. None means text-to-image.

    Returns:
        A tuple of `(embeds, attention_mask, image_pad_mask)`. The embeds have
        shape `(1, tokens, 4096)`; both masks have shape `(1, tokens)`.
        `image_pad_mask` marks the positions that hold reference image slots and
        is all False for text-to-image.
    """
    references = [_composite_over_white(image) for image in images] if images else []
    if references:
        text = build_edit_text(prompt, len(references))
    else:
        text = TEXT_TEMPLATE_T2I.format(prompt=prompt or " ")

    processor_kwargs = {
        "text": [text],
        "padding": True,
        "padding_side": "left",
        "return_tensors": "pt",
    }
    if references:
        processor_kwargs["images"] = references
    inputs = processor(**processor_kwargs).to(device)

    forward_kwargs = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "output_hidden_states": True,
    }
    if references and hasattr(inputs, "pixel_values"):
        forward_kwargs["pixel_values"] = inputs.pixel_values
        forward_kwargs["image_grid_thw"] = inputs.image_grid_thw
    if hasattr(inputs, "mm_token_type_ids"):
        forward_kwargs["mm_token_type_ids"] = inputs.mm_token_type_ids

    text_model = getattr(text_encoder, "language_model", text_encoder)
    handle = text_model.norm.register_forward_hook(lambda module, args, output: args[0])
    try:
        outputs = text_encoder(**forward_kwargs)
    finally:
        handle.remove()

    hidden = outputs.hidden_states[-1]
    mask = inputs.attention_mask
    drop = _system_prefix_length(processor)

    valid = hidden[mask.bool()]
    valid = valid[drop:]
    embeds = valid.unsqueeze(0)

    valid_mask = torch.ones(embeds.shape[1], dtype=torch.long, device=embeds.device)

    image_token_id = processor.tokenizer.encode("<|image_pad|>")[0]
    slots = (inputs.input_ids[0] == image_token_id) & mask[0].bool()
    image_pad_mask = slots[drop:].unsqueeze(0)
    return embeds, valid_mask.unsqueeze(0), image_pad_mask


def _composite_over_white(image: Image.Image) -> Image.Image:
    """Flatten an image onto white for the vision tower.

    The VAE still reads all four channels; only this copy loses the alpha.

    Args:
        image: The reference image.

    Returns:
        An RGB image.
    """
    if image.mode != "RGBA":
        return image.convert("RGB")
    white = Image.new("RGB", image.size, (255, 255, 255))
    white.paste(image, mask=image.getchannel("A"))
    return white


def decode_latents(
    vae,
    latents: torch.Tensor,
    height: int,
    width: int,
    *,
    dtype: torch.dtype | None = None,
) -> Image.Image:
    """Decode packed latents into an image.

    Args:
        vae: The `AutoencoderKLQwenImage21`.
        latents: Packed latents of shape `(B, tokens, 64)`.
        height: The output height in pixels.
        width: The output width in pixels.
        dtype: The dtype to decode in. None uses the VAE dtype.

    Returns:
        The decoded PIL image. It has 4 channels when the VAE decodes 4.
    """
    target_dtype = dtype if dtype is not None else vae.dtype
    unpacked = unpack_latents(latents, height, width).to(target_dtype)

    channels = unpacked.shape[1]
    mean = torch.tensor(vae.config.latents_mean[:channels], device=unpacked.device, dtype=target_dtype)
    std = torch.tensor(vae.config.latents_std[:channels], device=unpacked.device, dtype=target_dtype)
    unpacked = unpacked * std.view(1, channels, 1, 1, 1) + mean.view(1, channels, 1, 1, 1)

    with torch.inference_mode():
        image = vae.decode(unpacked, return_dict=False)[0][:, :, 0]
        image = (image / 2 + 0.5).clamp(0, 1)
        array = image.cpu().permute(0, 2, 3, 1).float().numpy()
    array = (array * 255).round().astype("uint8")
    return Image.fromarray(array[0])


def encode_reference_latents(
    vae,
    images: Sequence[Image.Image],
    sizes: Sequence[tuple[int, int]],
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Encode reference images into normalized packed latents.

    The latents are normalized the same way `decode_latents` undoes them, so
    the denoiser sees latent values on the scale it was trained on.

    Args:
        vae: The `AutoencoderKLQwenImage21`.
        images: The reference images, in order. All four channels are kept.
        sizes: The derived `(width, height)` for each image, from
            `resolve_sizes`.
        dtype: The compute dtype. None uses the VAE dtype.

    Returns:
        Packed latents of shape `(1, total_tokens, 64)`, with the references
        concatenated in order.

    Raises:
        ValueError: If `images` and `sizes` differ in length.
    """
    if len(images) != len(sizes):
        raise ValueError(f"Got {len(images)} images but {len(sizes)} sizes.")
    from diffusers.image_processor import VaeImageProcessor

    target_dtype = dtype if dtype is not None else vae.dtype
    device = next(vae.parameters()).device
    channels = LATENT_CHANNELS
    processor = VaeImageProcessor(
        vae_scale_factor=VAE_SCALE_FACTOR, vae_latent_channels=channels
    )
    mean = torch.tensor(
        vae.config.latents_mean[:channels], device=device, dtype=target_dtype
    ).view(1, channels, 1, 1, 1)
    std = torch.tensor(
        vae.config.latents_std[:channels], device=device, dtype=target_dtype
    ).view(1, channels, 1, 1, 1)

    packed: list[torch.Tensor] = []
    with torch.inference_mode():
        for image, (width, height) in zip(images, sizes):
            pixels = processor.preprocess(image, width=width, height=height)
            pixels = pixels.unsqueeze(2).to(device=device, dtype=target_dtype)
            output = vae.encode(pixels)
            latent_dist = getattr(output, "latent_dist", None)
            latents = output.latents if latent_dist is None else latent_dist.mode()
            latents = (latents.to(dtype=target_dtype) - mean) / std
            latent_height, latent_width = latents.shape[3], latents.shape[4]
            packed.append(
                latents.view(1, channels, latent_height * latent_width).transpose(1, 2)
            )
    return torch.cat(packed, dim=1)
