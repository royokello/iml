from __future__ import annotations


def quantize_text_encoder(*args, **kwargs):
    from .text_encoder import quantize_text_encoder as _quantize_text_encoder

    return _quantize_text_encoder(*args, **kwargs)


def quantize_denoiser(*args, **kwargs):
    from .denoiser import quantize_denoiser as _quantize_denoiser

    return _quantize_denoiser(*args, **kwargs)


__all__ = ["quantize_text_encoder", "quantize_denoiser"]
