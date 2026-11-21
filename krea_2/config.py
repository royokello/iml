from __future__ import annotations

from pathlib import Path

MODEL_DIR = "krea_2/model"

TEXT_ENCODER_LAYERS = 36
TEXT_ENCODER_SELECT_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
TEXT_HIDDEN_DIM = 2560
TEXT_MAX_LENGTH = 512

PROMPT_TEMPLATE_PREFIX = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and background:"
    "<|im_end|>\n<|im_start|>user\n"
)
PROMPT_TEMPLATE_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
PROMPT_TEMPLATE_START_IDX = 34
PROMPT_TEMPLATE_SUFFIX_TOKENS = 5

VAE_SCALE_FACTOR = 8
PATCH_SIZE = 2
LATENT_CHANNELS = 16
LATENTS_MEAN = (
    -0.7571,
    -0.7089,
    -0.9113,
    0.1075,
    -0.1745,
    0.9653,
    -0.1517,
    1.5508,
    0.4134,
    -0.0715,
    0.5517,
    -0.3632,
    -0.1922,
    -0.9497,
    0.2503,
    -0.2921,
)
LATENTS_STD = (
    2.8184,
    1.4541,
    2.3275,
    2.6558,
    1.2196,
    1.7708,
    2.6052,
    2.0743,
    3.2687,
    2.1526,
    2.8652,
    1.5579,
    1.6382,
    1.1253,
    2.8251,
    1.9160,
)
VAE_CONFIG = {
    "_class_name": "AutoencoderKLQwenImage",
    "_diffusers_version": "0.34.0.dev0",
    "attn_scales": [],
    "base_dim": 96,
    "dim_mult": [1, 2, 4, 4],
    "dropout": 0.0,
    "latents_mean": list(LATENTS_MEAN),
    "latents_std": list(LATENTS_STD),
    "num_res_blocks": 2,
    "temperal_downsample": [False, True, True],
    "z_dim": 16,
}

DEFAULT_TURBO_STEPS = 8
DEFAULT_RAW_STEPS = 28
DEFAULT_RAW_GUIDANCE = 4.5
DEFAULT_TURBO_MU = 1.15
BASE_IMAGE_SEQ_LEN = 256
MAX_IMAGE_SEQ_LEN = 6400
BASE_SHIFT = 0.5
MAX_SHIFT = 1.15


def resolve_model_root(root: str | Path) -> Path:
    """Return the Krea 2 model folder for a repo root.

    Args:
        root: The root folder that holds the `krea_2` folder.

    Returns:
        The absolute path `<root>/krea_2/model`.
    """
    return Path(root).expanduser().resolve() / "krea_2" / "model"


def resolve_outputs_dir(root: str | Path) -> Path:
    """Return the Krea 2 output folder for a repo root.

    Args:
        root: The root folder that holds the `krea_2` folder.

    Returns:
        The absolute path `<root>/krea_2/outputs`.
    """
    return Path(root).expanduser().resolve() / "krea_2" / "outputs"


def image_token_count(width: int, height: int) -> int:
    """Return the number of image tokens in a latent grid.

    Args:
        width: The output width in pixels.
        height: The output height in pixels.

    Returns:
        `(height // 16) * (width // 16)`.
    """
    return (int(height) // 16) * (int(width) // 16)
