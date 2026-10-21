"""Shared input and output helpers for Flux 3 Klein.

Gen, train, and sample use these helpers, so all three use the same text
window, position ids, sigmas, and latent normalization.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from flux2.gen import _pack_latents, _patchify_latents, _retrieve_latents, _unpatchify_latents
from flux3.config import IMAGE_ROPE_SCALE, SCHEDULER_SHIFT, TEXT_ENCODER_OUT_LAYERS, TEXT_PAD_MULTIPLE

if TYPE_CHECKING:
    from PIL import Image

_PADDING_POSITION_ID = 1
_LATENT_CHANNELS = 128


def prepare_text_ids(length: int) -> torch.Tensor:
    """Return the position ids of the text tokens.

    Args:
        length: The number of text tokens.

    Returns:
        A float tensor of shape `(length, 4)`. Each row is `(0, 0, 0, i)`.
    """
    return torch.cartesian_prod(
        torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(length)
    ).float()


def prepare_image_ids(height: int, width: int, layer: int = 0) -> torch.Tensor:
    """Return the position ids of the image tokens.

    Args:
        height: The token grid height.
        width: The token grid width.
        layer: The `l` coordinate. The output image uses 0. The reference
            images use -1, -2, and so on.

    Returns:
        A float tensor of shape `(height * width, 4)`. Each row is
        `(0, h * 0.5, w * 0.5, layer)`.
    """
    ids = torch.cartesian_prod(
        torch.arange(1), torch.arange(height), torch.arange(width), torch.tensor([layer])
    ).float()
    ids[:, 1:3] *= IMAGE_ROPE_SCALE
    return ids


def build_sigmas(num_steps: int) -> torch.Tensor:
    """Return the shifted sigmas for a run.

    Args:
        num_steps: The number of denoise steps.

    Returns:
        A float32 tensor with `num_steps + 1` values. The first is 1.0 after
        the shift. The last is 0.0.
    """
    sigmas = torch.linspace(1.0, 1.0 / num_steps, num_steps, dtype=torch.float64)
    sigmas = SCHEDULER_SHIFT * sigmas / (1 + (SCHEDULER_SHIFT - 1) * sigmas)
    return torch.cat([sigmas, torch.zeros(1, dtype=torch.float64)]).float()


def _text_window(tokenizer, text: str, max_length: int) -> int:
    real_length = len(tokenizer(text, truncation=True, max_length=max_length)["input_ids"])
    return min(math.ceil(real_length / TEXT_PAD_MULTIPLE) * TEXT_PAD_MULTIPLE, max_length)


def encode_prompt(
    tokenizer,
    text_encoder,
    prompt: str,
    *,
    device: torch.device,
    max_length: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a prompt into the text conditioning.

    The text is padded to a multiple of `TEXT_PAD_MULTIPLE` tokens, up to
    `max_length`. The padding position ids are 1. The 8 hidden states of
    `TEXT_ENCODER_OUT_LAYERS` are stacked along the channel axis.

    Args:
        tokenizer: The Qwen tokenizer.
        text_encoder: The text encoder. It must be on `device`.
        prompt: The prompt text.
        device: The device of the text encoder.
        max_length: The maximum number of tokens.

    Returns:
        A tuple of the embeddings with shape `(1, S, 20480)` and the text ids
        with shape `(S, 4)`, both on `device`.
    """
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    window = _text_window(tokenizer, text, max_length)
    encoded = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        padding_side="right",
        truncation=True,
        max_length=window,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    position_ids = torch.arange(window, device=device)[None].masked_fill(attention_mask == 0, _PADDING_POSITION_ID)

    with torch.inference_mode():
        output = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            use_cache=False,
        )

    hidden = torch.stack([output.hidden_states[index] for index in TEXT_ENCODER_OUT_LAYERS], dim=1)
    embeds = hidden.permute(0, 2, 1, 3).reshape(1, window, -1)
    return embeds, prepare_text_ids(window).to(device)


def resize_reference(image: Image.Image, ref_size: int) -> Image.Image:
    """Resize a reference image.

    The short side becomes `ref_size`. Both sides round to multiples of 16,
    with a minimum of 16.

    Args:
        image: The reference image.
        ref_size: The short side in pixels.

    Returns:
        The resized image.
    """
    from PIL import Image

    width, height = image.size
    scale = ref_size / min(width, height)
    new_width = max(16, round(width * scale / 16) * 16)
    new_height = max(16, round(height * scale / 16) * 16)
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def _vae_scale_factor(vae) -> int:
    return 2 ** (len(vae.config.block_out_channels) - 1)


def _bn_stats(vae, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = vae.bn.running_mean.view(1, -1, 1, 1).to(like.device, like.dtype)
    std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(like.device, like.dtype)
    return mean, std


def encode_image_latent(vae, image: Image.Image, device: torch.device) -> tuple[torch.Tensor, int, int]:
    """Encode an image into normalized latent tokens.

    The VAE must be on `device` in fp16. The image sides must be multiples of 16.

    Args:
        vae: The Flux 2 VAE.
        image: The RGB image.
        device: The VAE device.

    Returns:
        A tuple of the tokens with shape `(h * w, 128)`, `h`, and `w`.
    """
    from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor

    processor = Flux2ImageProcessor(vae_scale_factor=_vae_scale_factor(vae) * 2)
    processor.check_image_input(image)
    tensor = processor.preprocess(image).to(device=device, dtype=torch.float16)

    with torch.inference_mode():
        latent = _patchify_latents(_retrieve_latents(vae.encode(tensor)))
        mean, std = _bn_stats(vae, latent)
        latent = (latent - mean) / std
    return _pack_latents(latent)[0], latent.shape[-2], latent.shape[-1]


def decode_tokens(vae, tokens: torch.Tensor, height: int, width: int) -> Image.Image:
    """Decode latent tokens into an image.

    The VAE must be on the device of `tokens`, in fp16.

    Args:
        vae: The Flux 2 VAE.
        tokens: The normalized tokens with shape `(1, height * width, 128)`.
        height: The token grid height.
        width: The token grid width.

    Returns:
        The decoded image.
    """
    from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor

    processor = Flux2ImageProcessor(vae_scale_factor=_vae_scale_factor(vae) * 2)
    with torch.inference_mode():
        latent = tokens.view(1, height, width, _LATENT_CHANNELS).permute(0, 3, 1, 2)
        mean, std = _bn_stats(vae, latent)
        latent = _unpatchify_latents(latent * std + mean)
        decoded = vae.decode(latent.to(dtype=vae.dtype), return_dict=False)[0]
        return processor.postprocess(decoded, output_type="pil")[0]
