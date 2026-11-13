from __future__ import annotations

from pathlib import Path

SDXL_DIR = "sdxl"
DEFAULT_MODEL_NAME = "base"
MODEL_COMPONENTS = (
    "unet",
    "text_encoder",
    "text_encoder_2",
    "vae",
    "tokenizer",
    "tokenizer_2",
    "scheduler",
)


def resolve_model_root(root: str | Path, name: str = DEFAULT_MODEL_NAME) -> Path:
    """Return `<root>/sdxl/<name>`.

    Args:
        root: The data root, for example `/iml_root`.
        name: The model folder name. Defaults to `base`.

    Returns:
        The model folder. It is not created.

    Raises:
        ValueError: If `name` is empty, or is not a single path segment.
    """
    model_name = name.strip()
    if (
        not model_name
        or model_name in {".", ".."}
        or "/" in model_name
        or "\\" in model_name
    ):
        raise ValueError(f"Invalid model name: {name!r}. Expected one folder name, for example 'base'.")

    return Path(root).expanduser().resolve() / SDXL_DIR / model_name


def resolve_outputs_dir(root: str | Path) -> Path:
    """Return `<root>/sdxl/outputs`.

    Args:
        root: The data root.

    Returns:
        The output folder. It is not created.
    """
    return Path(root).expanduser().resolve() / SDXL_DIR / "outputs"


def validate_model_root(path: str | Path) -> Path:
    """Return a model folder after checking that every component exists.

    Args:
        path: The model folder.

    Returns:
        The absolute model folder.

    Raises:
        FileNotFoundError: If the folder or one of its components is missing.
    """
    model_root = Path(path).expanduser().resolve()
    if not model_root.is_dir():
        raise FileNotFoundError(f"Model folder not found: {model_root}")

    missing = [name for name in MODEL_COMPONENTS if not (model_root / name).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Model folder is incomplete: {model_root}. Missing: {', '.join(missing)}."
        )
    return model_root


__all__ = [
    "DEFAULT_MODEL_NAME",
    "MODEL_COMPONENTS",
    "SDXL_DIR",
    "resolve_model_root",
    "resolve_outputs_dir",
    "validate_model_root",
]
