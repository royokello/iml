from __future__ import annotations

from collections.abc import Mapping, Sequence

from utils.quant.name import convert_quant_name


def build_mixed_target_config(
    method: str,
    target_tensors: Mapping[str, Sequence[str]],
) -> dict[str, list[str]]:
    high_method, low_method = convert_quant_name(method)
    high_targets = list(target_tensors.get("high", []))
    low_targets = list(target_tensors.get("low", []))
    fp32_targets = list(target_tensors.get("fp32", []))
    fp16_targets = list(target_tensors.get("fp16", []))

    config: dict[str, list[str]] = {}
    if high_method == low_method:
        config[high_method] = high_targets + low_targets
    else:
        config[high_method] = high_targets
        config[low_method] = low_targets
    if fp32_targets:
        config["fp32"] = fp32_targets
    if fp16_targets:
        config["fp16"] = fp16_targets
    return config


__all__ = ["build_mixed_target_config"]
