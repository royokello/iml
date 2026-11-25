from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from utils.quant.name import convert_quant_ranks, mixed_quant_methods

_RANK_KEY = re.compile(r"^rank_([1-9][0-9]*)$")
_PASS_THROUGH_KEYS = ("fp32", "fp16", "bf16")


def _rank_keys(target_tensors: Mapping[str, Sequence[str]]) -> list[str]:
    """Return the rank keys of a target set in rank order.

    Raises:
        ValueError: If the rank keys do not start at `rank_1` or have a gap.
    """
    numbered = sorted(
        (int(match.group(1)), key)
        for key in target_tensors
        if (match := _RANK_KEY.match(key))
    )
    numbers = [number for number, _ in numbered]
    if numbers and numbers != list(range(1, len(numbers) + 1)):
        raise ValueError(
            "Rank keys must start at 'rank_1' and have no gaps, "
            f"got {sorted(target_tensors)}."
        )
    return [key for _, key in numbered]


def _validated_rank_keys(
    target_tensors: Mapping[str, Sequence[str]],
) -> list[str]:
    """Return the rank keys, and reject any group name that is not a rank or a method.

    Raises:
        ValueError: If a group name is neither a rank nor a known method.
    """
    allowed = set(mixed_quant_methods()) | set(_PASS_THROUGH_KEYS)
    unknown = [
        key for key in target_tensors if not _RANK_KEY.match(key) and key not in allowed
    ]
    if unknown:
        raise ValueError(
            f"Unknown target group(s) {sorted(unknown)}. Use 'rank_1', 'rank_2', ... "
            "for the precision ranks, or a method name to pin a group."
        )
    return _rank_keys(target_tensors)


def build_mixed_target_config(
    method: str,
    target_tensors: Mapping[str, Sequence[str]],
) -> dict[str, list[str]]:
    """Map tensor groups onto quantization methods for one method name.

    A target set names its groups `rank_1`, `rank_2`, ... in order of precision,
    most precise first. Rank 1 takes the base method. The bottom rank drops by
    the depth of the variant, and the ranks between drop by one rung. A group
    named after a method, such as `sym-high`, keeps that method whatever the
    ladder says.

    Args:
        method: A method name such as `sym-med-nano`.
        target_tensors: The tensor groups.

    Returns:
        A mapping of method name to full tensor names.

    Raises:
        ValueError: If a group name is unknown, the ranks are not contiguous, or
            the method cannot reach a requested rank.
    """
    rank_keys = _validated_rank_keys(target_tensors)
    if not rank_keys:
        raise ValueError("A target set must have at least a 'rank_1' group.")

    config: dict[str, list[str]] = {}

    def add(target_method: str, names: Sequence[str]) -> None:
        if names:
            config.setdefault(target_method, []).extend(names)

    for rank_key, rank_method in zip(
        rank_keys,
        convert_quant_ranks(method, len(rank_keys)),
    ):
        add(rank_method, list(target_tensors[rank_key]))

    for key in _PASS_THROUGH_KEYS:
        add(key, list(target_tensors.get(key, [])))

    pinned_keys = [
        key
        for key in target_tensors
        if key not in _PASS_THROUGH_KEYS and key not in rank_keys
    ]
    for key in sorted(pinned_keys):
        add(key, list(target_tensors[key]))

    return config


__all__ = ["build_mixed_target_config"]
