from __future__ import annotations

import math
from collections.abc import Sequence
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

DEFAULT_TEXT_QUANT_METHOD = "sym-med-nano"
DEFAULT_DENOISER_QUANT_METHOD = "aff-high-nano"
DEFAULT_SEED = 19930625
UNQUANTIZED_DTYPES = ("fp16", "bf16", "fp32")
DEFAULT_WEIGHT_DTYPE = "fp16"
EDIT_IMAGE_SLOT = "<image1><|vision_start|><|image_pad|><|vision_end|>"
DEFAULT_REF_SIZE = 512


def parse_quant_method(method: str | None) -> tuple[str | None, str]:
    """Split a method name into a quant method and an unquantized dtype.

    Args:
        method: A quantization method name, one of `UNQUANTIZED_DTYPES`, or
            None.

    Returns:
        A `(quant_method, weight_dtype)` pair. A dtype name selects the
        unquantized checkpoint, so the first item is None and the second is
        that dtype. A quant method returns the method and
        `DEFAULT_WEIGHT_DTYPE`. None returns `(None, DEFAULT_WEIGHT_DTYPE)`.
    """
    if method is None:
        return None, DEFAULT_WEIGHT_DTYPE
    lowered = method.lower()
    if lowered in UNQUANTIZED_DTYPES:
        return None, lowered
    return method, DEFAULT_WEIGHT_DTYPE


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


def build_edit_text(prompt: str, image_count: int) -> str:
    """Return the edit chat text with one image marker block per reference.

    Args:
        prompt: The user prompt. An empty prompt becomes a single space.
        image_count: The number of reference images. One or more.

    Returns:
        The chat text that the processor tokenizes.

    Raises:
        ValueError: If `image_count` is less than one.
    """
    if image_count < 1:
        raise ValueError(f"image_count={image_count} needs at least one reference image.")
    slots = " ".join(
        f"<image{index}><|vision_start|><|image_pad|><|vision_end|>"
        for index in range(1, image_count + 1)
    )
    return (
        "<|im_start|>system\n"
        f"{TEXT_SYSTEM_PROMPT}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{slots}{prompt or ' '}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def calculate_dimensions(area: int, aspect: float) -> tuple[int, int]:
    """Return a width and height with the given area and aspect ratio.

    Each side rounds to the nearest multiple of `SIZE_MULTIPLE`, as the
    diffusers reference pipeline does. Do not use `round_to_multiple` here: it
    rounds up, which widens every non-square reference.

    Args:
        area: The target pixel area.
        aspect: The width divided by the height.

    Returns:
        The `(width, height)` pair in pixels.
    """
    width = round(math.sqrt(area * aspect) / SIZE_MULTIPLE) * SIZE_MULTIPLE
    height = round((width / aspect) / SIZE_MULTIPLE) * SIZE_MULTIPLE
    return width, height


def resolve_sizes(
    ref_sizes: Sequence[tuple[int, int]] | None,
    width: int | None,
    height: int | None,
    ref_size: int,
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    """Return one size per reference image and the output size.

    Every reference is resized to `ref_size` squared at its own aspect ratio.
    Each output axis takes the explicit value when given, else the last
    reference's derived size, else `DEFAULT_WIDTH` / `DEFAULT_HEIGHT`.

    Args:
        ref_sizes: The original `(width, height)` of each reference, in order.
            None or an empty sequence means text-to-image.
        width: The requested output width, or None to derive it.
        height: The requested output height, or None to derive it.
        ref_size: The reference area side length, a positive multiple of
            `SIZE_MULTIPLE`.

    Returns:
        A `(reference_sizes, output_size)` pair.

    Raises:
        ValueError: If `ref_size` is not a positive multiple of
            `SIZE_MULTIPLE`.
    """
    if ref_size <= 0 or ref_size % SIZE_MULTIPLE != 0:
        raise ValueError(
            f"ref_size={ref_size} must be a positive multiple of {SIZE_MULTIPLE}."
        )
    area = ref_size * ref_size
    derived = [calculate_dimensions(area, ref_width / ref_height) for ref_width, ref_height in ref_sizes or ()]
    fallback = derived[-1] if derived else (DEFAULT_WIDTH, DEFAULT_HEIGHT)
    output = (
        width if width is not None else fallback[0],
        height if height is not None else fallback[1],
    )
    return derived, output
