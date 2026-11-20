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


def _checkpoint_rank(
    handle,
    entries: dict[tuple[str, str], dict[str, str]],
    path: Path,
) -> int:
    """Return the largest module rank of a checkpoint.

    The factor shapes are authoritative. Metadata rank is not trusted, because
    a checkpoint may store a different rank per module.

    Args:
        handle: An open safetensors handle.
        entries: The parsed entries.
        path: The checkpoint path, used in error messages.

    Returns:
        The largest rank over every module in the checkpoint.

    Raises:
        RuntimeError: If the checkpoint holds no module.
    """
    ranks = [int(handle.get_slice(fields["A"]).get_shape()[0]) for fields in entries.values()]
    if not ranks:
        raise RuntimeError(f"LoRA checkpoint holds no modules: {path}")
    return max(ranks)


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
    """Inject SDXL LoRA modules and add one or more checkpoints into them.

    A checkpoint may use kohya names or native names. Text-encoder tensors are
    loaded only when the checkpoint has them.

    The checkpoints may use different ranks. Every module is injected at the
    largest rank of the list. A checkpoint with a smaller rank writes its
    factors into the leading block of that rank and leaves the rest at zero, so
    the product of its two factors is unchanged and its result is exact.

    The checkpoints add together. Each factor is multiplied by the square root
    of `alpha / rank * strength`, and the module scaling is 1, so every
    checkpoint contributes its own weight delta to the same module and the
    strengths do not interact.

    Args:
        unet: The SDXL UNet.
        text_encoder: CLIP-L.
        text_encoder_2: CLIP-G.
        loras: A map of checkpoint path to strength. The strength multiplies
            that checkpoint's contribution.

    Returns:
        The number of injected modules per model, keyed `unet`, `text_encoder`,
        and `text_encoder_2`.

    Raises:
        FileNotFoundError: If a checkpoint is missing.
        RuntimeError: If a key does not map, a module is absent from the model,
            or a factor shape does not fit the model.
        ValueError: If a strength is negative.
    """
    if not loras:
        return {"unet": 0, "text_encoder": 0, "text_encoder_2": 0}

    models = {"unet": unet, "text_encoder": text_encoder, "text_encoder_2": text_encoder_2}
    maps: dict[str, dict[str, TrainableLoraLinear]] = {name: {} for name in models}

    # Pass 1: read the headers, so the injection rank is known before any
    # module is built.
    checkpoints: list[
        tuple[Path, float, str, dict[tuple[str, str], dict[str, str]], float, int]
    ] = []
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
            file_rank = _checkpoint_rank(handle, entries, path)
            file_alpha = _read_alpha(meta, file_rank)

        checkpoints.append((path, float(strength), checkpoint_format, entries, file_alpha, file_rank))

    rank = max(checkpoint[5] for checkpoint in checkpoints)

    for target in sorted({target for checkpoint in checkpoints for target, _ in checkpoint[3]}):
        inject_trainable_lora_modules(
            models[target],
            target_linear_names=SDXL_LORA_TARGETS if target == "unet" else SDXL_TEXT_TARGETS,
            rank=rank,
            alpha=rank,
        )
        maps[target] = _wrappers(models[target])

    # Start every module at zero, so each checkpoint adds into empty factors.
    for module_map in maps.values():
        for wrapper in module_map.values():
            wrapper.lora_A.data.zero_()
            wrapper.lora_B.data.zero_()
            wrapper.alpha = rank
            wrapper.scaling = 1.0

    # Pass 2: add each checkpoint into the shared factors.
    for path, strength, checkpoint_format, entries, file_alpha, file_rank in checkpoints:
        print(
            f"    adding {path.name} [{strength:.2f}] format={checkpoint_format} "
            f"rank={file_rank} alpha={file_alpha:g}",
            flush=True,
        )

        with safe_open(str(path), framework="pt") as handle:
            for (target, module), fields in entries.items():
                wrapper = maps[target].get(module)
                if wrapper is None:
                    raise RuntimeError(f"LoRA module {module!r} is not in the {target} model: {path}")

                down = handle.get_tensor(fields["A"])
                up = handle.get_tensor(fields["B"])
                module_rank = int(down.shape[0])
                if module_rank > rank or int(up.shape[1]) != module_rank:
                    raise RuntimeError(
                        f"LoRA factors for {module!r} in {path} have shapes "
                        f"{tuple(down.shape)} and {tuple(up.shape)}, which do not fit rank {rank}."
                    )
                if int(down.shape[1]) != wrapper.in_features or int(up.shape[0]) != wrapper.out_features:
                    raise RuntimeError(
                        f"LoRA module {module!r} in {path} is shaped for "
                        f"{up.shape[0]}x{down.shape[1]}, but the model reports "
                        f"{wrapper.out_features}x{wrapper.in_features}."
                    )

                if "alpha" in fields:
                    alpha_value = float(handle.get_tensor(fields["alpha"]).float().reshape(-1)[0])
                else:
                    alpha_value = file_alpha

                factor = (alpha_value / module_rank) * strength
                if factor == 0.0:
                    continue
                root = factor**0.5

                device = wrapper.lora_A.device
                dtype = wrapper.lora_A.dtype
                padded_a = torch.zeros((rank, wrapper.in_features), device=device, dtype=dtype)
                padded_a[:module_rank] = down.to(device=device, dtype=dtype) * root
                padded_b = torch.zeros((wrapper.out_features, rank), device=device, dtype=dtype)
                padded_b[:, :module_rank] = up.to(device=device, dtype=dtype) * root

                wrapper.lora_A.data.add_(padded_a)
                wrapper.lora_B.data.add_(padded_b)

    return {name: len(module_map) for name, module_map in maps.items()}


__all__ = ["inject_sdxl_loras"]
