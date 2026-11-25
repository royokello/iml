from __future__ import annotations

import math
from pathlib import Path

PACKAGE_NAME = "qwen_2"
MODEL_DIR = "qwen_2/model"
OUTPUTS_DIR = "qwen_2/outputs"
VARIANTS = ("turbo", "base")
DEFAULT_VARIANT = "turbo"

TURBO_SIGMAS = (
    1.0,
    0.978453,
    0.95418,
    0.926626,
    0.89508,
    0.845148,
    0.704534,
    0.414568,
)
BASE_STEPS = 40
BASE_IMAGE_SEQ_LEN = 256
MAX_IMAGE_SEQ_LEN = 8192
BASE_SHIFT = 0.5
MAX_SHIFT = 0.9
SHIFT_TERMINAL = 0.02

VAE_SCALE_FACTOR = 16
LATENT_CHANNELS = 64
PATCH_SIZE = 1
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
SIZE_MULTIPLE = 32

TEXT_HIDDEN_DIM = 4096
TEXT_SYSTEM_PROMPT = "Comprehend and analyze the provided prompt."
TEXT_TEMPLATE_T2I = (
    "<|im_start|>system\n"
    f"{TEXT_SYSTEM_PROMPT}<|im_end|>\n"
    "<|im_start|>user\n{prompt}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

DEFAULT_TEXT_QUANT_METHOD = "sym-high-mini"
DEFAULT_DENOISER_QUANT_METHOD = "sym-med-nano"
DEFAULT_SEED = 19930625


def resolve_model_root(root: str | Path) -> Path:
    """Return the qwen_2 model folder for a project root."""
    return Path(root).expanduser().resolve() / MODEL_DIR


def resolve_outputs_dir(root: str | Path) -> Path:
    """Return the qwen_2 output folder for a project root."""
    return Path(root).expanduser().resolve() / OUTPUTS_DIR


def calculate_shift(image_tokens: int) -> float:
    """Return the flow-matching shift for a token count.

    Args:
        image_tokens: The number of image tokens.

    Returns:
        The shift, clamped to the base and maximum shift.
    """
    ratio = (image_tokens - BASE_IMAGE_SEQ_LEN) / (MAX_IMAGE_SEQ_LEN - BASE_IMAGE_SEQ_LEN)
    shift = BASE_SHIFT + ratio * (MAX_SHIFT - BASE_SHIFT)
    return min(max(shift, BASE_SHIFT), MAX_SHIFT)


def round_to_multiple(value: int, multiple: int = SIZE_MULTIPLE) -> int:
    """Round a value up to the nearest multiple.

    Args:
        value: The value to round.
        multiple: The multiple to round to.

    Returns:
        The rounded value.
    """
    return math.ceil(value / multiple) * multiple


def image_token_count(width: int, height: int) -> int:
    """Return the number of image tokens for an output size.

    Args:
        width: The output width in pixels.
        height: The output height in pixels.

    Returns:
        The token count.
    """
    return (height // VAE_SCALE_FACTOR) * (width // VAE_SCALE_FACTOR)
