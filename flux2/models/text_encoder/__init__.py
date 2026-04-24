from .model import Qwen3Model, Qwen3PreTrainedModel, Qwen3TextEncoder


def load_flux2_text_encoder(*args, **kwargs):
    from flux2.loaders import load_flux2_text_encoder as _load

    return _load(*args, **kwargs)

__all__ = [
    "Qwen3Model",
    "Qwen3PreTrainedModel",
    "Qwen3TextEncoder",
    "load_flux2_text_encoder",
]
