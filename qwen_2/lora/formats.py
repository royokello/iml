"""Parse external LoRA checkpoints into model module names.

Two key layouts are accepted:

- The ComfyUI and ai-toolkit layout, which prefixes every name with
  `diffusion_model.` and fuses the SwiGLU pair into `img_mlp.gate_up`.
- The repo layout, which stores bare `lora_A` and `lora_B` keys per module.

The fused layout holds `[gate_layer; proj]` rows on its `lora_B` tensor. This
module records which half a module owns. It does not read the row count,
because the model supplies that at injection time.

The header reader does not import torch, so the parser runs without a GPU.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from qwen_2.lora.config import QWEN2_LORA_FUSED, QWEN2_LORA_TARGETS
from utils.safetensors_io import TensorFile

_PREFIX = "diffusion_model."
_BLOCK_NAME = re.compile(r"^transformer_blocks\.(\d+)\.(.+)$")

_FACTORS = (
    (".lora_A.weight", "lora_A.weight"),
    (".lora_B.weight", "lora_B.weight"),
    (".alpha", "alpha"),
)

# The fused spelling and the two model modules that share its factors.
_FUSED_PARTS = (
    ("img_mlp.gate_layer", "gate"),
    ("img_mlp.proj", "up"),
)


@dataclass(frozen=True)
class LoraEntry:
    """One model module and the checkpoint tensors that feed it.

    Attributes:
        module: The model module path, such as
            `transformer_blocks.0.attn.to_q`.
        a_key: The checkpoint tensor for the down factor.
        b_key: The checkpoint tensor for the up factor.
        alpha_key: The per-module alpha tensor, or None.
        half: `gate` or `up` when the checkpoint fuses `img_mlp.gate_up`
            and this entry owns one half. None for a plain module.
    """

    module: str
    a_key: str
    b_key: str
    alpha_key: str | None = None
    half: str | None = None


@dataclass(frozen=True)
class LoraFile:
    """A parsed LoRA checkpoint.

    Attributes:
        path: The checkpoint path.
        format: `comfy` or `native`.
        rank: The largest module rank in the file.
        alpha: The file alpha. The rank when the file stores none.
        entries: Module path to entry.
    """

    path: Path
    format: str
    rank: int
    alpha: float
    entries: dict[str, LoraEntry]


def parse_lora_keys(keys: list[str] | tuple[str, ...]) -> dict[str, LoraEntry]:
    """Group a checkpoint's tensor names by model module.

    Args:
        keys: The tensor names.

    Returns:
        Module path to entry, in file order.

    Raises:
        ValueError: If a name is unsupported, a module lacks a factor, or a
            fused and a plain spelling name the same module.
    """
    grouped: dict[str, dict] = {}
    rejected: list[str] = []

    for key in keys:
        body = key.removeprefix(_PREFIX)
        field = None
        for suffix, field_name in _FACTORS:
            if body.endswith(suffix):
                body = body[: -len(suffix)]
                field = field_name
                break
        match = _BLOCK_NAME.match(body)
        if field is None or match is None:
            rejected.append(key)
            continue

        block, module_suffix = match.group(1), match.group(2)
        if module_suffix == QWEN2_LORA_FUSED:
            parts = [
                (f"transformer_blocks.{block}.{name}", half)
                for name, half in _FUSED_PARTS
            ]
        elif module_suffix in QWEN2_LORA_TARGETS:
            parts = [(f"transformer_blocks.{block}.{module_suffix}", None)]
        else:
            rejected.append(key)
            continue

        for module, half in parts:
            bucket = grouped.setdefault(module, {"fields": {}, "half": half})
            if bucket["half"] != half:
                raise ValueError(
                    f"LoRA module {module!r} is named both fused and plain."
                )
            bucket["fields"][field] = key

    if rejected:
        listed = ", ".join(sorted(rejected))
        raise ValueError(f"Unsupported LoRA tensors: {listed}")

    entries: dict[str, LoraEntry] = {}
    for module, bucket in grouped.items():
        fields = bucket["fields"]
        if "lora_A.weight" not in fields or "lora_B.weight" not in fields:
            raise ValueError(f"LoRA module {module!r} lacks a down or up tensor.")
        entries[module] = LoraEntry(
            module=module,
            a_key=fields["lora_A.weight"],
            b_key=fields["lora_B.weight"],
            alpha_key=fields.get("alpha"),
            half=bucket["half"],
        )
    return entries


def read_lora_file(path: str | Path) -> LoraFile:
    """Read a LoRA checkpoint's names, format, rank, and alpha.

    Args:
        path: The checkpoint path.

    Returns:
        The parsed file.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If a name is unsupported or the file holds no module.
    """
    tensor_file = TensorFile(path)
    keys = tensor_file.keys()
    entries = parse_lora_keys(keys)
    if not entries:
        raise ValueError(f"LoRA checkpoint holds no modules: {path}")

    rank = max(int(tensor_file.info(entry.a_key)["shape"][0]) for entry in entries.values())

    metadata = tensor_file.metadata()
    raw_alpha = metadata.get("alpha") or metadata.get("ss_network_alpha")
    alpha = float(rank) if raw_alpha is None else float(raw_alpha)

    checkpoint_format = "comfy" if any(key.startswith(_PREFIX) for key in keys) else "native"
    return LoraFile(
        path=Path(path),
        format=checkpoint_format,
        rank=rank,
        alpha=alpha,
        entries=entries,
    )


__all__ = ["LoraEntry", "LoraFile", "parse_lora_keys", "read_lora_file"]
