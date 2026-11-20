from __future__ import annotations

import json
from pathlib import Path

import torch

DEFAULT_NUM_TRAIN_TIMESTEPS = 1000


def betas_from_config(scheduler_config_path: str | Path) -> torch.Tensor:
    """Build the training betas from a scheduler config file.

    Args:
        scheduler_config_path: The `scheduler_config.json` of the model folder.

    Returns:
        A float32 tensor of `num_train_timesteps` betas.

    Raises:
        FileNotFoundError: If the config file is missing.
        ValueError: If the schedule is not `scaled_linear`, or a beta is out of
            range.
    """
    path = Path(scheduler_config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Scheduler config not found: {path}")

    config = json.loads(path.read_text(encoding="utf-8"))
    schedule = config.get("beta_schedule", "scaled_linear")
    if schedule != "scaled_linear":
        raise ValueError(f"Unsupported beta schedule {schedule!r} in {path}")

    beta_start = float(config["beta_start"])
    beta_end = float(config["beta_end"])
    num_train_timesteps = int(config.get("num_train_timesteps", DEFAULT_NUM_TRAIN_TIMESTEPS))
    if not 0.0 < beta_start < beta_end < 1.0:
        raise ValueError(f"Invalid beta range {beta_start} to {beta_end} in {path}")

    return torch.linspace(
        beta_start**0.5,
        beta_end**0.5,
        num_train_timesteps,
        dtype=torch.float32,
    ) ** 2


def alphas_cumprod_from_betas(betas: torch.Tensor) -> torch.Tensor:
    """Return the cumulative product of `1 - betas`.

    Args:
        betas: A tensor of betas, one per training timestep.

    Returns:
        A tensor with the same length as `betas`.
    """
    return torch.cumprod(1.0 - betas, dim=0)


def add_noise(
    latents: torch.Tensor,
    noise: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    timesteps: torch.Tensor,
    noise_offset: float = 0.0,
) -> torch.Tensor:
    """Return the noised latent for a batch.

    The offset is added to the noise per channel and is not renormalized. This
    matches sd-scripts, which produced the reference checkpoint.

    Args:
        latents: Clean latents, shape `(batch, channels, height, width)`.
        noise: Standard normal noise with the shape of `latents`.
        alphas_cumprod: The cumulative alpha product, one value per timestep.
        timesteps: One timestep per batch element.
        noise_offset: Standard deviation of the per-channel offset. Zero
            disables the offset.

    Returns:
        The noised latents.
    """
    if noise_offset:
        offset_shape = (latents.shape[0], latents.shape[1], 1, 1)
        offset = torch.randn(offset_shape, device=noise.device, dtype=noise.dtype)
        noise = noise + noise_offset * offset

    # Work in fp32. `1 - alpha` at the last timesteps is too small for fp16,
    # and the error would show up as a wrong SNR.
    alpha = alphas_cumprod.to(device=latents.device)[timesteps].float()
    alpha = alpha.view(-1, 1, 1, 1)
    noised = alpha.sqrt() * latents.float() + (1.0 - alpha).sqrt() * noise.float()
    return noised.to(dtype=latents.dtype)


def min_snr_weight(
    alphas_cumprod: torch.Tensor,
    timesteps: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Return the min-SNR loss weight for a batch.

    Args:
        alphas_cumprod: The cumulative alpha product, one value per timestep.
        timesteps: One timestep per batch element.
        gamma: The SNR cap. Use 0 or less to disable the weighting.

    Returns:
        One weight per batch element, at most 1.
    """
    if gamma <= 0:
        return torch.ones(len(timesteps), device=timesteps.device)

    alpha = alphas_cumprod.to(device=timesteps.device)[timesteps]
    snr = alpha / (1.0 - alpha)
    return torch.clamp(gamma / snr, max=1.0)


__all__ = [
    "add_noise",
    "alphas_cumprod_from_betas",
    "betas_from_config",
    "min_snr_weight",
]
