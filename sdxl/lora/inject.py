from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open

from flux2.lora.model import TrainableLoraLinear, inject_trainable_lora_modules
from sdxl.lora.config import SDXL_LORA_TARGETS, SDXL_TEXT_TARGETS
from sdxl.lora.formats import detect_format, normalize_module_name, parse_kohya_key, parse_native_key


def _wrappers(model: torch.nn.Module) -> dict[str, TrainableLoraLinear]:
    """Return the injected LoRA modules of a model, keyed by module name.

    Args:
        model: A model that holds `TrainableLoraLinear` children.

    Returns:
        The module name without a leading `text_model.` mapped to the wrapper.
    """
    return {
        normalize_module_name(name): module
        for name, module in model.named_modules()
        if isinstance(module, TrainableLoraLinear)
    }


def _parse_entries(
    keys: list[str],
    path: Path,
) -> tuple[str, dict[tuple[str, str], dict[str, str]]]:
    """Group a checkpoint's tensor names by target and module.

    Args:
        keys: The tensor names.
        path: The checkpoint path, used in error messages.

    Returns:
        The format and a map of `(target, module)` to `{field: tensor name}`.

    Raises:
        RuntimeError: If a name does not match the format, or a module lacks a
            factor.
    """
    checkpoint_format = detect_format(keys)
    parse = parse_kohya_key if checkpoint_format == "kohya" else parse_native_key

    entries: dict[tuple[str, str], dict[str, str]] = {}
    for key in keys:
        parsed = parse(key)
        if parsed is None:
            raise RuntimeError(f"Unsupported LoRA tensor {key!r} in {path}")
        target, module, field = parsed
        entries.setdefault((target, module), {})[field] = key

    for (target, module), fields in entries.items():
        if "A" not in fields or "B" not in fields:
            raise RuntimeError(
                f"LoRA module {module!r} in {target} lacks a down or up tensor in {path}"
            )
    return checkpoint_format, entries


def _read_rank(meta: dict[str, str], entries: dict[tuple[str, str], dict[str, str]], handle) -> int:
    """Return the LoRA rank of a checkpoint.

    Args:
        meta: The checkpoint metadata.
        entries: The parsed entries.
        handle: An open safetensors handle.

    Returns:
        The rank.

    Raises:
        RuntimeError: If the rank is absent from the metadata and no factor can
            be read.
    """
    raw = meta.get("rank") or meta.get("ss_network_dim")
    if raw is not None:
        return int(raw)
    for fields in entries.values():
        if "A" in fields:
            return int(handle.get_slice(fields["A"]).get_shape()[0])
    raise RuntimeError("Cannot read the LoRA rank from the checkpoint.")


def _read_alpha(meta: dict[str, str], rank: int) -> float:
    """Return the LoRA alpha of a checkpoint, defaulting to the rank.

    Args:
        meta: The checkpoint metadata.
        rank: The LoRA rank.

    Returns:
        The alpha.
    """
    raw = meta.get("alpha") or meta.get("ss_network_alpha")
    if raw is None:
        return float(rank)
    return float(raw)


def inject_sdxl_loras(
    unet: torch.nn.Module,
    text_encoder: torch.nn.Module,
    text_encoder_2: torch.nn.Module,
    loras: dict[str, float],
) -> dict[str, int]:
    """Inject SDXL LoRA modules and load one or more checkpoints into them.

    A checkpoint may use kohya names or native names. Text-encoder tensors are
    loaded only when the checkpoint has them.

    Args:
        unet: The SDXL UNet.
        text_encoder: CLIP-L.
        text_encoder_2: CLIP-G.
        loras: A map of checkpoint path to strength.

    Returns:
        The number of injected modules per model, keyed `unet`, `text_encoder`,
        and `text_encoder_2`.

    Raises:
        FileNotFoundError: If a checkpoint is missing.
        RuntimeError: If a key does not map, a module is absent from the model,
            or two checkpoints use different ranks.
        ValueError: If a strength is negative.
    """
    if not loras:
        return {"unet": 0, "text_encoder": 0, "text_encoder_2": 0}

    rank: int | None = None
    models = {"unet": unet, "text_encoder": text_encoder, "text_encoder_2": text_encoder_2}
    maps: dict[str, dict[str, TrainableLoraLinear]] = {name: {} for name in models}
    injected_targets: set[str] = set()

    for raw_path, strength in loras.items():
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"LoRA checkpoint not found: {path}")
        if strength < 0:
            raise ValueError(f"Negative LoRA strength is not supported: {strength}")

        with safe_open(str(path), framework="pt") as handle:
            keys = list(handle.keys())
            meta = dict(handle.metadata() or {})
            checkpoint_format, entries = _parse_entries(keys, path)

            file_rank = _read_rank(meta, entries, handle)
            if rank is None:
                rank = file_rank
            elif file_rank != rank:
                raise RuntimeError(
                    f"LoRA rank {file_rank} in {path} does not match rank {rank} from the first checkpoint."
                )
            file_alpha = _read_alpha(meta, rank)

            for target in {target for target, _ in entries}:
                if target in injected_targets:
                    continue
                inject_trainable_lora_modules(
                    models[target],
                    target_linear_names=SDXL_LORA_TARGETS if target == "unet" else SDXL_TEXT_TARGETS,
                    rank=rank,
                    alpha=file_alpha,
                )
                maps[target] = _wrappers(models[target])
                injected_targets.add(target)

            scale = strength**0.5 if strength != 1.0 else 1.0
            print(
                f"    loading {path.name} [{strength:.2f}] format={checkpoint_format} "
                f"rank={rank} alpha={file_alpha:g}",
                flush=True,
            )

            for (target, module), fields in entries.items():
                wrapper = maps[target].get(module)
                if wrapper is None:
                    raise RuntimeError(f"LoRA module {module!r} is not in the {target} model: {path}")

                down = handle.get_tensor(fields["A"])
                up = handle.get_tensor(fields["B"])
                wrapper.lora_A.data.copy_(
                    down.to(device=wrapper.lora_A.device, dtype=wrapper.lora_A.dtype)
                )
                wrapper.lora_B.data.copy_(up.to(device=wrapper.lora_B.device, dtype=wrapper.lora_B.dtype))

                if "alpha" in fields:
                    alpha_value = float(handle.get_tensor(fields["alpha"]).float().reshape(-1)[0])
                    wrapper.alpha = int(alpha_value) if alpha_value.is_integer() else alpha_value
                else:
                    alpha_value = file_alpha
                    wrapper.alpha = int(alpha_value) if float(alpha_value).is_integer() else alpha_value
                wrapper.scaling = alpha_value / rank

                if scale != 1.0:
                    wrapper.lora_A.data.mul_(scale)
                    wrapper.lora_B.data.mul_(scale)

    return {name: len(module_map) for name, module_map in maps.items()}


__all__ = ["inject_sdxl_loras"]
