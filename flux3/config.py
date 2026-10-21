from __future__ import annotations

from pathlib import Path

TEXT_ENCODER_OUT_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)
TEXT_ENCODER_LAYERS = 33
TEXT_PAD_MULTIPLE = 80
IMAGE_ROPE_SCALE = 0.5
SCHEDULER_SHIFT = 6.93
MODEL_DIR = "flux3/model"


def resolve_model_root(root: str | Path) -> Path:
    """Return the Flux 3 model folder for a repo root.

    Args:
        root: The root folder that holds the `flux3` folder.

    Returns:
        The absolute path `<root>/flux3/model`.
    """
    return Path(root).expanduser().resolve() / "flux3" / "model"
