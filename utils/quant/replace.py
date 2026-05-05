from __future__ import annotations

from collections.abc import Callable, Iterable

import torch.nn as nn

from utils.quant.linear import QuantizedLinear


def target_tensors_to_linear_names(
    target_tensors: Iterable[str],
    *,
    key_transform: Callable[[str], str] | None = None,
) -> tuple[str, ...]:
    names: list[str] = []
    for tensor in target_tensors:
        name = tensor if key_transform is None else key_transform(tensor)
        names.append(name.removesuffix(".weight"))
    return tuple(names)


def replace_targeted_linear_modules(
    module: nn.Module,
    *,
    method: str,
    target_linear_names: tuple[str, ...],
    quantize_weights: bool = True,
    path: tuple[str, ...] = (),
    checkpoint_keys: set[str] | None = None,
) -> None:
    for name, child in list(module.named_children()):
        child_path = (*path, name)
        if isinstance(child, nn.Linear) and is_targeted_linear(
            child_path,
            target_linear_names,
            checkpoint_keys=checkpoint_keys,
        ):
            setattr(
                module,
                name,
                QuantizedLinear(child, method=method)
                if quantize_weights
                else QuantizedLinear.from_prequantized(child, method=method),
            )
            continue
        replace_targeted_linear_modules(
            child,
            method=method,
            target_linear_names=target_linear_names,
            quantize_weights=quantize_weights,
            path=child_path,
            checkpoint_keys=checkpoint_keys,
        )


def is_targeted_linear(
    module_path: tuple[str, ...],
    target_linear_names: tuple[str, ...],
    *,
    checkpoint_keys: set[str] | None = None,
) -> bool:
    if not module_path:
        return False

    full_name = ".".join(module_path)
    if full_name not in target_linear_names:
        return False

    if checkpoint_keys is None:
        return True

    return f"{full_name}.sub_scales" in checkpoint_keys


__all__ = [
    "is_targeted_linear",
    "replace_targeted_linear_modules",
    "target_tensors_to_linear_names",
]
