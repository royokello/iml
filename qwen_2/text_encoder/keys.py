from __future__ import annotations

_TEXT_PREFIX = "model.language_model."
_MODEL_PREFIX = "model."


def _strip_text_key(name: str) -> str:
    """Map a checkpoint key onto the text-only model.

    Args:
        name: A full checkpoint tensor name.

    Returns:
        The name without the `model.language_model.` prefix.
    """
    return name.removeprefix(_TEXT_PREFIX)


def _strip_model_key(name: str) -> str:
    """Map a checkpoint key onto the full vision-language model.

    Args:
        name: A full checkpoint tensor name.

    Returns:
        The name without the `model.` prefix.
    """
    return name.removeprefix(_MODEL_PREFIX)


__all__ = ["_strip_model_key", "_strip_text_key"]
