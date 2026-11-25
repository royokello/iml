from __future__ import annotations

import torch
from PIL import Image

from qwen_2.config import (
    LATENT_CHANNELS,
    TEXT_SYSTEM_PROMPT,
    TEXT_TEMPLATE_T2I,
    TURBO_SIGMAS,
    VAE_SCALE_FACTOR,
)

__all__ = [
    "build_img_mask",
    "build_img_shapes",
    "build_sigmas",
    "decode_latents",
    "encode_prompt",
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


def build_img_mask(sequence_length: int, target_tokens: int) -> torch.Tensor:
    """Build the joint-sequence image mask.

    The first `sequence_length` entries describe the prompt, which has no image
    tokens in text-to-image, so they are all False. The appended entries mark
    the target image, one slot per 2x2 group of latent tokens.

    Args:
        sequence_length: The number of prompt tokens.
        target_tokens: The number of target latent tokens.

    Returns:
        A bool tensor of shape `(1, sequence_length + target_tokens // 4)`.

    Raises:
        ValueError: If `target_tokens` is not a multiple of 4.
    """
    if target_tokens % 4 != 0:
        raise ValueError(f"target_tokens={target_tokens} must be a multiple of 4.")
    prompt = torch.zeros(1, sequence_length, dtype=torch.bool)
    target = torch.ones(1, target_tokens // 4, dtype=torch.bool)
    return torch.cat([prompt, target], dim=1)


def build_img_shapes(height: int, width: int) -> list[list[tuple[int, int, int]]]:
    """Build the per-sample image layout in latent tokens.

    Args:
        height: The output height in pixels.
        width: The output width in pixels.

    Returns:
        `[[(1, height // 16, width // 16)]]`.

    Raises:
        ValueError: If a side is not a positive multiple of 16.
    """
    if height <= 0 or width <= 0 or height % VAE_SCALE_FACTOR or width % VAE_SCALE_FACTOR:
        raise ValueError(
            f"height={height} and width={width} must be positive multiples of {VAE_SCALE_FACTOR}."
        )
    return [[(1, height // VAE_SCALE_FACTOR, width // VAE_SCALE_FACTOR)]]


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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a text prompt for the denoiser.

    The hidden states come from the last decoder layer before the final RMS
    norm, which is what the denoiser was trained on.

    Args:
        processor: The `Qwen3VLProcessor`.
        text_encoder: The text encoder, text-only or full.
        prompt: The user prompt.
        device: The device that holds the text encoder.

    Returns:
        A tuple of `(embeds, attention_mask)`. The embeds have shape
        `(1, tokens, 4096)`; the mask has shape `(1, tokens)` and is bool.
    """
    text = TEXT_TEMPLATE_T2I.format(prompt=prompt or " ")
    inputs = processor(
        text=[text],
        padding=True,
        padding_side="left",
        return_tensors="pt",
    ).to(device)

    text_model = getattr(text_encoder, "language_model", text_encoder)
    handle = text_model.norm.register_forward_hook(lambda module, args, output: args[0])
    try:
        outputs = text_encoder(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            output_hidden_states=True,
        )
    finally:
        handle.remove()

    hidden = outputs.hidden_states[-1]
    mask = inputs.attention_mask
    drop = _system_prefix_length(processor)

    valid = hidden[mask.bool()]
    valid = valid[drop:]
    embeds = valid.unsqueeze(0)

    valid_mask = torch.ones(embeds.shape[1], dtype=torch.long, device=embeds.device)
    return embeds, valid_mask.unsqueeze(0)


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
