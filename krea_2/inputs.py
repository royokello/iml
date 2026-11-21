"""Input, sampler, and decode helpers for Krea 2 generation.

Gen uses these helpers, so the text window, position ids, sigmas, latent pack,
and decode all come from one place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from krea_2.config import (
    BASE_IMAGE_SEQ_LEN,
    BASE_SHIFT,
    DEFAULT_TURBO_MU,
    LATENT_CHANNELS,
    LATENTS_MEAN,
    LATENTS_STD,
    MAX_IMAGE_SEQ_LEN,
    MAX_SHIFT,
    PATCH_SIZE,
    PROMPT_TEMPLATE_PREFIX,
    PROMPT_TEMPLATE_START_IDX,
    PROMPT_TEMPLATE_SUFFIX,
    PROMPT_TEMPLATE_SUFFIX_TOKENS,
    TEXT_ENCODER_SELECT_LAYERS,
    TEXT_MAX_LENGTH,
    VAE_SCALE_FACTOR,
)

if TYPE_CHECKING:
    from PIL import Image

_TEXT_POSITION_CHOICES = ("diffusers", "absolute")
_PIXEL_TO_TOKEN = VAE_SCALE_FACTOR * PATCH_SIZE


def tokenize_prompt(tokenizer, prompt: str, *, max_length: int = TEXT_MAX_LENGTH) -> dict[str, torch.Tensor]:
    """Tokenize a prompt in the fixed Krea 2 template layout.

    The prompt is padded to a fixed length and the assistant suffix is appended
    after the padding.

    Args:
        tokenizer: The Qwen2 tokenizer.
        prompt: The prompt text.
        max_length: The number of text tokens after the prefix is dropped.

    Returns:
        A dict with `input_ids` and a bool `attention_mask`, each with shape
        `(1, max_length + start_idx)`.
    """
    text = PROMPT_TEMPLATE_PREFIX + prompt
    prompt_length = max_length + PROMPT_TEMPLATE_START_IDX - PROMPT_TEMPLATE_SUFFIX_TOKENS
    encoded = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=prompt_length,
        return_tensors="pt",
    )
    suffix = tokenizer(PROMPT_TEMPLATE_SUFFIX, return_tensors="pt")

    return {
        "input_ids": torch.cat([encoded["input_ids"], suffix["input_ids"]], dim=1),
        "attention_mask": torch.cat([encoded["attention_mask"], suffix["attention_mask"]], dim=1).bool(),
    }


def encode_prompt(
    tokenizer,
    text_encoder,
    prompt: str,
    *,
    device: torch.device,
    max_length: int = TEXT_MAX_LENGTH,
    text_positions: str = "diffusers",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a prompt into the text conditioning.

    Args:
        tokenizer: The Qwen2 tokenizer.
        text_encoder: The text model, already on `device`.
        prompt: The prompt text.
        device: The device of the text model.
        max_length: The number of text tokens after the prefix is dropped.
        text_positions: `diffusers` uses valid-token positions
            (`cumsum(mask) - 1`). `absolute` uses raw indices, as the official
            `encoder.py` does.

    Returns:
        The embeddings with shape `(1, max_length, 12, 2560)` and the bool text
        mask with shape `(1, max_length)`.

    Raises:
        ValueError: If `text_positions` is not a known choice.
    """
    if text_positions not in _TEXT_POSITION_CHOICES:
        raise ValueError(f"Unknown text_positions {text_positions!r}. Expected one of: {_TEXT_POSITION_CHOICES}.")

    tokens = tokenize_prompt(tokenizer, prompt, max_length=max_length)
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    if text_positions == "diffusers":
        position_ids = (attention_mask.long().cumsum(dim=-1) - 1).clamp(min=0)
    else:
        position_ids = torch.arange(attention_mask.shape[-1], device=device)[None].expand_as(attention_mask)

    with torch.inference_mode():
        output = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            use_cache=False,
        )

    hidden = torch.stack([output.hidden_states[index] for index in TEXT_ENCODER_SELECT_LAYERS], dim=2)
    hidden = hidden[:, PROMPT_TEMPLATE_START_IDX:]
    attention_mask = attention_mask[:, PROMPT_TEMPLATE_START_IDX:]
    return hidden, attention_mask


def prepare_position_ids(
    text_len: int,
    grid_h: int,
    grid_w: int,
    device: torch.device,
) -> torch.Tensor:
    """Build the rotary coordinates of the combined text and image sequence.

    Args:
        text_len: The number of text tokens.
        grid_h: The latent grid height in tokens.
        grid_w: The latent grid width in tokens.
        device: The device for the result.

    Returns:
        A float tensor with shape `(text_len + grid_h * grid_w, 3)`. Text rows
        are zero; image rows are `(0, h, w)`.
    """
    text_ids = torch.zeros(text_len, 3, device=device)
    image_ids = torch.zeros(grid_h, grid_w, 3, device=device)
    image_ids[..., 1] = torch.arange(grid_h, device=device)[:, None]
    image_ids[..., 2] = torch.arange(grid_w, device=device)[None, :]
    return torch.cat([text_ids, image_ids.reshape(-1, 3)], dim=0)


def calculate_shift(image_tokens: int) -> float:
    """Return the resolution-dependent timestep shift for the raw model.

    Args:
        image_tokens: The number of image tokens.

    Returns:
        The `mu` value on the line from (256, 0.5) to (6400, 1.15).
    """
    slope = (MAX_SHIFT - BASE_SHIFT) / (MAX_IMAGE_SEQ_LEN - BASE_IMAGE_SEQ_LEN)
    intercept = BASE_SHIFT - slope * BASE_IMAGE_SEQ_LEN
    return slope * image_tokens + intercept


def build_sigmas(steps: int, image_tokens: int, *, distilled: bool) -> torch.Tensor:
    """Return the shifted sigma schedule.

    Args:
        steps: The number of denoise steps.
        image_tokens: The number of image tokens, not counting text.
        distilled: True pins `mu` to 1.15, as the turbo model expects. False
            derives `mu` from `image_tokens`.

    Returns:
        A float32 tensor with `steps + 1` values. The last value is 0.
    """
    ts = torch.linspace(1.0, 1.0 / steps, steps, dtype=torch.float64)
    mu = DEFAULT_TURBO_MU if distilled else calculate_shift(image_tokens)
    sigma = torch.exp(torch.tensor(mu, dtype=torch.float64)) / (
        torch.exp(torch.tensor(mu, dtype=torch.float64)) + (1.0 / ts - 1.0)
    )
    return torch.cat([sigma, torch.zeros(1, dtype=torch.float64)]).float()


def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """Patchify a latent tensor into tokens.

    Args:
        latents: A tensor with shape `(B, C, H, W)`.

    Returns:
        The tokens with shape `(B, (H // 2) * (W // 2), C * 4)`.
    """
    batch, channels, height, width = latents.shape
    patch = PATCH_SIZE
    tokens = latents.reshape(batch, channels, height // patch, patch, width // patch, patch)
    tokens = tokens.permute(0, 2, 4, 1, 3, 5)
    return tokens.reshape(batch, (height // patch) * (width // patch), channels * patch * patch)


def unpack_latents(latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Unpatchify tokens into a latent tensor.

    Args:
        latents: The tokens with shape `(B, (H // 16) * (W // 16), 64)`.
        height: The output height in pixels.
        width: The output width in pixels.

    Returns:
        A tensor with shape `(B, 16, 1, H // 8, W // 8)`.
    """
    batch = latents.shape[0]
    patch = PATCH_SIZE
    grid_h = height // _PIXEL_TO_TOKEN
    grid_w = width // _PIXEL_TO_TOKEN
    tokens = latents.reshape(batch, grid_h, grid_w, LATENT_CHANNELS, patch, patch)
    tokens = tokens.permute(0, 3, 1, 4, 2, 5)
    latents = tokens.reshape(batch, LATENT_CHANNELS, grid_h * patch, grid_w * patch)
    return latents.unsqueeze(2)


def decode_latents(
    vae,
    latents: torch.Tensor,
    height: int,
    width: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> "Image.Image":
    """Decode latent tokens into an image.

    Args:
        vae: The Qwen-Image VAE, on the device of `latents`.
        latents: The tokens with shape `(1, (H // 16) * (W // 16), 64)`.
        height: The output height in pixels.
        width: The output width in pixels.
        dtype: The dtype for the VAE input.

    Returns:
        The decoded RGB image.
    """
    from PIL import Image

    device = latents.device
    mean = torch.tensor(LATENTS_MEAN, device=device, dtype=torch.float32).view(1, LATENT_CHANNELS, 1, 1, 1)
    std = torch.tensor(LATENTS_STD, device=device, dtype=torch.float32).view(1, LATENT_CHANNELS, 1, 1, 1)
    latents = unpack_latents(latents, height, width).to(device=device, dtype=torch.float32)
    latents = latents * std + mean

    with torch.inference_mode():
        decoded = vae.decode(latents.to(dtype=dtype), return_dict=False)[0][:, :, 0]
    decoded = (decoded / 2 + 0.5).clamp(0, 1)
    array = decoded[0].mul(255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array)


def round_to_multiple(value: int, multiple: int = _PIXEL_TO_TOKEN) -> int:
    """Round `value` up to the nearest multiple.

    Args:
        value: The value to round.
        multiple: The alignment.

    Returns:
        The rounded value.
    """
    return ((int(value) + multiple - 1) // multiple) * multiple


__all__ = [
    "build_sigmas",
    "calculate_shift",
    "decode_latents",
    "encode_prompt",
    "pack_latents",
    "prepare_position_ids",
    "round_to_multiple",
    "tokenize_prompt",
    "unpack_latents",
]
