from __future__ import annotations

from typing import Any

__all__ = [
    "load_flux2_dataset",
    "load_flux2_denoiser",
    "load_flux2_text_encoder",
]


def load_flux2_dataset(*args: Any, **kwargs: Any) -> Any:
    from .dataset import load_flux2_dataset as _load_flux2_dataset

    return _load_flux2_dataset(*args, **kwargs)


def load_flux2_denoiser(*args: Any, **kwargs: Any) -> Any:
    from .denoiser import _load_flux2_denoiser

    return _load_flux2_denoiser(*args, **kwargs)


def load_flux2_text_encoder(*args: Any, **kwargs: Any) -> Any:
    from .text_encoder import _load_flux2_text_encoder

    return _load_flux2_text_encoder(*args, **kwargs)
