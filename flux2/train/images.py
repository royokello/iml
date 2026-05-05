from __future__ import annotations

import torch

from flux2.gen import _pack_latents, _patchify_latents, _prepare_latent_ids, _retrieve_latents


def normalize_image_latent_resolution(
    image_width: int,
    image_height: int,
    *,
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


def image_latent_resolution(image, *, vae_scale_factor: int) -> tuple[int, int]:
    return normalize_image_latent_resolution(*image.size, vae_scale_factor=vae_scale_factor)


def prepare_image_latent(
    vae,
    processor,
    image,
    *,
    device: torch.device,
    vae_scale_factor: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    processor.check_image_input(image)

    image_width, image_height = image_latent_resolution(image, vae_scale_factor=vae_scale_factor)

    image_tensor = processor.preprocess(image, height=image_height, width=image_width, resize_mode="crop")
    image_tensor = image_tensor.to(device=device, dtype=torch.float16)
    with torch.inference_mode():
        latent = _retrieve_latents(vae.encode(image_tensor))
    latent = _patchify_latents(latent)

    latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latent.device, latent.dtype)
    latents_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(
        latent.device, latent.dtype
    )
    latent = (latent - latents_bn_mean) / latents_bn_std

    return _pack_latents(latent).squeeze(0).cpu(), _prepare_latent_ids(torch, latent).squeeze(0).cpu(), latent
