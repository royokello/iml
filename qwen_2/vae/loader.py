from __future__ import annotations

from pathlib import Path

import torch
from diffusers import AutoencoderKLQwenImage21

__all__ = ["load_qwen2_vae"]

_REQUIRED_FILES = ("config.json", "diffusion_pytorch_model.safetensors")


def _resolve_folder(path: str | Path) -> Path:
    model_path = Path(path).expanduser()
    return model_path.parent if model_path.is_file() else model_path


def load_qwen2_vae(path: str | Path, *, dtype: torch.dtype | None = None) -> AutoencoderKLQwenImage21:
    """Load the Qwen Image 2.1 VAE.

    Args:
        path: The `vae` folder, or a file inside it.
        dtype: The dtype to cast the weights to. None keeps the checkpoint dtype.

    Returns:
        The VAE in eval mode.

    Raises:
        FileNotFoundError: If the folder or a required file is missing.
    """
    folder = _resolve_folder(path).resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"VAE folder not found: {folder}")
    missing = [name for name in _REQUIRED_FILES if not (folder / name).is_file()]
    if missing:
        raise FileNotFoundError(f"VAE files not found in {folder}: {', '.join(missing)}")

    kwargs = {"local_files_only": True}
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    vae = AutoencoderKLQwenImage21.from_pretrained(str(folder), **kwargs)
    if dtype is not None:
        vae = vae.to(dtype=dtype)
    vae.eval()
    return vae
