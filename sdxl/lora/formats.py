from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from safetensors import safe_open

# SDXL sets `layers_per_block` to 2. The LDM block index counts the layers and
# the downsampler, so each group spans 3 indices.
_LAYERS_PER_BLOCK = 2

_KOHYA_PREFIXES = (
    ("lora_unet_", "unet"),
    ("lora_te1_", "text_encoder"),
    ("lora_te2_", "text_encoder_2"),
)

# The suffix of a kohya key and the suffix of the native name it becomes.
_KOHYA_FIELDS = (
    (".lora_down.weight", "lora_A.weight"),
    (".lora_up.weight", "lora_B.weight"),
    (".alpha", "alpha"),
)

_NATIVE_FIELDS = (
    (".lora_A.weight", "A"),
    (".lora_B.weight", "B"),
    (".alpha", "alpha"),
)

_ATTENTION_TAIL = re.compile(
    r"^transformer_blocks_(\d+)_(attn1|attn2)_(to_q|to_k|to_v|to_out_0)$"
)
_FEED_FORWARD_TAIL = re.compile(r"^transformer_blocks_(\d+)_(ff_net_0_proj|ff_net_2)$")
_INPUT_PREFIX = re.compile(r"^input_blocks_(\d+)_1_(.+)$")
_OUTPUT_PREFIX = re.compile(r"^output_blocks_(\d+)_1_(.+)$")
_MIDDLE_PREFIX = re.compile(r"^middle_block_1_(.+)$")
_TEXT_LAYER = re.compile(
    r"^text_model_encoder_layers_(\d+)_(self_attn|mlp)"
    r"_(q_proj|k_proj|v_proj|out_proj|fc1|fc2)$"
)


def normalize_module_name(name: str) -> str:
    """Remove the `text_model.` prefix that some transformers classes add.

    `CLIPTextModel` exposes `encoder.layers.*`. `CLIPTextModelWithProjection`
    exposes `text_model.encoder.layers.*`. Both must match the same key.

    Args:
        name: A module path.

    Returns:
        The path without a leading `text_model.`.
    """
    return name.removeprefix("text_model.")


def detect_format(keys: Iterable[str]) -> str:
    """Return the checkpoint format from its tensor names.

    Args:
        keys: The tensor names.

    Returns:
        `kohya` or `native`.

    Raises:
        RuntimeError: If no key matches either format.
    """
    for key in keys:
        if key.startswith("lora_unet_") or key.startswith("lora_te1_") or key.startswith("lora_te2_"):
            return "kohya"
        if key.endswith(".lora_A.weight") or key.endswith(".lora_B.weight"):
            return "native"
    raise RuntimeError("LoRA checkpoint is neither kohya nor native format.")


def format_of(path: str | Path) -> str:
    """Return the checkpoint format of a file, reading only its header.

    Args:
        path: The LoRA checkpoint.

    Returns:
        `kohya` or `native`.
    """
    with safe_open(str(path), framework="np") as handle:
        return detect_format(handle.keys())


def _unet_tail(rest: str) -> str | None:
    """Convert the part after an LDM block prefix to diffusers names.

    Args:
        rest: The key part after `input_blocks_{i}_1_` or a sibling prefix.

    Returns:
        The diffusers name part, or None for an unknown part.
    """
    if rest in ("proj_in", "proj_out"):
        return rest

    match = _ATTENTION_TAIL.match(rest)
    if match:
        index, attention, projection = match.groups()
        projection = "to_out.0" if projection == "to_out_0" else projection
        return f"transformer_blocks.{index}.{attention}.{projection}"

    match = _FEED_FORWARD_TAIL.match(rest)
    if match:
        index, projection = match.groups()
        projection = "ff.net.0.proj" if projection == "ff_net_0_proj" else "ff.net.2"
        return f"transformer_blocks.{index}.{projection}"

    return None


def convert_kohya_key(key: str) -> tuple[str, str] | None:
    """Convert one kohya tensor name to the model name that holds it.

    Args:
        key: A tensor name, for example
            `lora_unet_output_blocks_5_1_transformer_blocks_1_attn1_to_q.lora_down.weight`.

    Returns:
        A `(target, name)` pair, or None when the name is not supported. The
        target is `unet`, `text_encoder`, or `text_encoder_2`. The name is the
        native module path plus `.lora_A.weight`, `.lora_B.weight`, or `.alpha`.

    Example:
        >>> convert_kohya_key("lora_unet_output_blocks_5_1_transformer_blocks_1_attn1_to_q.lora_down.weight")
        ('unet', 'up_blocks.1.attentions.2.transformer_blocks.1.attn1.to_q.lora_A.weight')
    """
    body = None
    field = None
    for suffix, native_field in _KOHYA_FIELDS:
        if key.endswith(suffix):
            body = key[: -len(suffix)]
            field = native_field
            break
    if body is None or field is None:
        return None

    target = None
    for prefix, tag in _KOHYA_PREFIXES:
        if body.startswith(prefix):
            body = body[len(prefix) :]
            target = tag
            break
    if target is None:
        return None

    if target == "unet":
        match = _INPUT_PREFIX.match(body)
        if match:
            index = int(match.group(1))
            block = (index - 1) // (_LAYERS_PER_BLOCK + 1)
            layer = (index - 1) % (_LAYERS_PER_BLOCK + 1)
            rest = _unet_tail(match.group(2))
            module = f"down_blocks.{block}.attentions.{layer}" if rest else None
        else:
            match = _OUTPUT_PREFIX.match(body)
            if match:
                index = int(match.group(1))
                block = index // (_LAYERS_PER_BLOCK + 1)
                layer = index % (_LAYERS_PER_BLOCK + 1)
                rest = _unet_tail(match.group(2))
                module = f"up_blocks.{block}.attentions.{layer}" if rest else None
            else:
                match = _MIDDLE_PREFIX.match(body)
                rest = _unet_tail(match.group(1)) if match else None
                module = "mid_block.attentions.0" if rest else None
    else:
        match = _TEXT_LAYER.match(body)
        rest = f"{match.group(2)}.{match.group(3)}" if match else None
        module = f"text_model.encoder.layers.{match.group(1)}" if match else None

    if module is None or rest is None:
        return None
    return target, f"{module}.{rest}.{field}"


def parse_kohya_key(key: str) -> tuple[str, str, str] | None:
    """Split a kohya key into target, native module name, and field.

    Args:
        key: A tensor name.

    Returns:
        A `(target, module, field)` triple, or None. The field is `A`, `B`, or
        `alpha`.
    """
    converted = convert_kohya_key(key)
    if converted is None:
        return None
    target, name = converted
    for suffix, field in ((".lora_A.weight", "A"), (".lora_B.weight", "B"), (".alpha", "alpha")):
        if name.endswith(suffix):
            module = name[: -len(suffix)]
            return target, normalize_module_name(module), field
    return None


def parse_native_key(key: str) -> tuple[str, str, str] | None:
    """Split a native key into target, module name, and field.

    Args:
        key: A tensor name, for example `up_blocks.0.attentions.0....lora_A.weight`.

    Returns:
        A `(target, module, field)` triple, or None. Native files carry UNet
        modules only.
    """
    for suffix, field in _NATIVE_FIELDS:
        if key.endswith(suffix):
            module = normalize_module_name(key[: -len(suffix)])
            return "unet", module, field
    return None
