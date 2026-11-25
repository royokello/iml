"""Inject external LoRA checkpoints into the Qwen 2 denoiser.

The base linears stay as they are. A `TrainableLoraLinear` wrapper adds a
rank-limited branch beside each one, which works for `nn.Linear` and for
`QuantizedLinear` alike.

Every checkpoint adds into one shared set of factors. The factors are built at
the largest rank in the list. A smaller checkpoint fills the leading block and
leaves zeros, so its product is exact and the strengths do not interact.

Strength and alpha fold into the square root of both factors, and the wrapper
scaling stays 1. That keeps the arithmetic linear in `alpha / rank * strength`.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch

from flux2.lora.model import TrainableLoraLinear, inject_trainable_lora_modules
from qwen_2.lora.config import QWEN2_LORA_TARGETS
from qwen_2.lora.formats import LoraFile, read_lora_file
from utils.safetensors_io import TensorFile

__all__ = ["apply_loras"]


def _wrappers(model: torch.nn.Module) -> dict[str, TrainableLoraLinear]:
    """Return the LoRA modules of a model, keyed by module path.

    Args:
        model: The injected model.

    Returns:
        Module path to wrapper.
    """
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, TrainableLoraLinear)
    }


def _down_and_up(
    tensor_file: TensorFile,
    entry,
    wrapper: TrainableLoraLinear,
    path: Path,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read one module's factors and place the fused half.

    Args:
        tensor_file: The open checkpoint.
        entry: The parsed module entry.
        wrapper: The injected module.
        path: The checkpoint path, used in error messages.

    Returns:
        The down and up factors, shaped for the base module.

    Raises:
        RuntimeError: If a factor shape does not fit the base module.
    """
    down = tensor_file.read(entry.a_key).to(torch.float32)
    up = tensor_file.read(entry.b_key).to(torch.float32)
    rank = int(down.shape[0])
    if int(up.shape[1]) != rank:
        raise RuntimeError(
            f"LoRA factors for {entry.module!r} in {path} have shapes "
            f"{tuple(down.shape)} and {tuple(up.shape)}, which disagree on the rank."
        )
    if int(down.shape[1]) != wrapper.in_features:
        raise RuntimeError(
            f"LoRA module {entry.module!r} in {path} is shaped for "
            f"{down.shape[1]} inputs, but the model reports {wrapper.in_features}."
        )

    if entry.half is not None:
        fused_rows = 2 * wrapper.out_features
        if int(up.shape[0]) != fused_rows:
            raise RuntimeError(
                f"Fused LoRA module {entry.module!r} in {path} has {up.shape[0]} rows, "
                f"but the model needs {fused_rows}."
            )
        up = up[: wrapper.out_features] if entry.half == "gate" else up[wrapper.out_features :]

    if int(up.shape[0]) != wrapper.out_features:
        raise RuntimeError(
            f"LoRA module {entry.module!r} in {path} writes {up.shape[0]} outputs, "
            f"but the model reports {wrapper.out_features}."
        )
    return down, up


def apply_loras(
    model: torch.nn.Module,
    loras: Mapping[str, float],
    *,
    training: bool = False,
) -> int:
    """Inject LoRA modules and add one or more checkpoints into them.

    Args:
        model: The denoiser, quantized or not.
        loras: Checkpoint path to strength. The strength multiplies that
            checkpoint's contribution.
        training: Keep the factors in float32. False stores them in the model
            dtype, which halves their footprint and removes a cast from every
            forward call. The branch computes in the model dtype either way.

    Returns:
        The number of injected modules.

    Raises:
        FileNotFoundError: If a checkpoint is missing.
        ValueError: If a strength is negative.
        RuntimeError: If a module is absent from the model, or a factor shape
            does not fit it.
    """
    if not loras:
        return 0

    model_dtype = next(model.parameters()).dtype

    files: list[tuple[LoraFile, float]] = []
    for raw_path, strength in loras.items():
        path = Path(raw_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"LoRA checkpoint not found: {path}")
        if strength < 0:
            raise ValueError(f"Negative LoRA strength is not supported: {strength}")
        files.append((read_lora_file(path), float(strength)))

    rank = max(lora.rank for lora, _ in files)
    injected = inject_trainable_lora_modules(
        model,
        target_linear_names=QWEN2_LORA_TARGETS,
        rank=rank,
        alpha=rank,
    )
    wrappers = _wrappers(model)

    for wrapper in wrappers.values():
        wrapper.lora_A.data.zero_()
        wrapper.lora_B.data.zero_()
        wrapper.alpha = rank
        wrapper.scaling = 1.0

    for lora, strength in files:
        print(
            f"    {lora.path.name} [{strength:.2f}] format={lora.format} "
            f"rank={lora.rank} alpha={lora.alpha:g}",
            flush=True,
        )
        tensor_file = TensorFile(lora.path)
        for module, entry in lora.entries.items():
            wrapper = wrappers.get(module)
            if wrapper is None:
                raise RuntimeError(
                    f"LoRA module {module!r} is not in the model: {lora.path}"
                )
            down, up = _down_and_up(tensor_file, entry, wrapper, lora.path)
            module_rank = int(down.shape[0])

            if entry.alpha_key is None:
                alpha = lora.alpha
            else:
                alpha = float(tensor_file.read(entry.alpha_key).reshape(-1)[0])
            factor = (alpha / module_rank) * strength
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

    if not training:
        for wrapper in wrappers.values():
            wrapper.lora_A.data = wrapper.lora_A.data.to(model_dtype)
            wrapper.lora_B.data = wrapper.lora_B.data.to(model_dtype)

    return len(injected)
