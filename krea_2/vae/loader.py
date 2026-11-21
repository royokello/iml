from __future__ import annotations

from pathlib import Path


def load_krea2_vae(path: str | Path, *, dtype=None):
    """Load the Krea 2 VAE.

    The VAE is the Qwen-Image autoencoder. The model folder holds a
    `diffusion_pytorch_model.safetensors` file and a `config.json` written by
    `krea_2.vae.copy_files`.

    Args:
        path: The `vae` folder, or the weight file inside it.
        dtype: The dtype to cast the weights to, or None to keep the file dtype.

    Returns:
        The VAE in eval mode.

    Raises:
        FileNotFoundError: If the config or the weight file is missing.
    """
    from diffusers import AutoencoderKLQwenImage

    model_path = Path(path).expanduser().resolve()
    folder = model_path.parent if model_path.is_file() else model_path
    config_path = folder / "config.json"
    weights_path = folder / "diffusion_pytorch_model.safetensors"
    if not config_path.is_file():
        raise FileNotFoundError(f"VAE config not found: {config_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"VAE weights not found: {weights_path}")

    vae = AutoencoderKLQwenImage.from_pretrained(str(folder), local_files_only=True)
    if dtype is not None:
        vae = vae.to(dtype=dtype)
    vae.eval()
    return vae


__all__ = ["load_krea2_vae"]
