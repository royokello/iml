from __future__ import annotations

from typing import Dict

import torch

from ..blocks.conv_2d import QuantConv1x1
from ..blocks.linear import QuantLinear
from .quantize import quantize_to_block32

OPTIONAL_TENSORS = {
    # Fourier time projection weights are procedurally re-created from config, so
    # they are not serialized in the original SDXL checkpoints.
    "time_proj.weight",
}


def load_weights(unet, model: Dict[str, torch.Tensor]) -> None:
    """
    Filter-load only keys that exist in this UNet and match in shape.
    `model` is expected to be a dict like from safetensors or state_dict().
    """
    state = unet.state_dict()
    quant_modules = {
        name: module
        for name, module in unet.named_modules()
        if isinstance(module, (QuantLinear, QuantConv1x1))
    }
    updated: dict[str, torch.Tensor] = {}
    matched = 0
    missing: list[str] = []
    missing_details: list[tuple[str, tuple[int, ...]]] = []

    for name, target in state.items():
        source_name = name
        module_name, _, param_name = name.rpartition(".")
        quant_module = quant_modules.get(module_name)
        if quant_module is not None:
            if param_name == "weight":
                tensor = model.get(source_name)
                if tensor is None:
                    if name in OPTIONAL_TENSORS:
                        continue
                    missing.append(f"{name} (looked for {source_name})")
                    missing_details.append((name, tuple(int(dim) for dim in target.shape)))
                    continue

                quantized, scales = quantize_to_block32(tensor)
                quant_module.load_quantized_weights(quantized, scales)
                updated[name] = quant_module.weight
                updated[f"{module_name}.weight_scale"] = quant_module.weight_scale
                matched += 1
                continue

            if param_name == "weight_scale":
                matched += 1
                continue

        tensor = model.get(source_name)
        if tensor is None:
            if name in OPTIONAL_TENSORS:
                # Skip optional tensors that Diffusers rebuilds on init.
                continue
            missing.append(f"{name} (looked for {source_name})")
            missing_details.append((name, tuple(int(dim) for dim in target.shape)))
            continue

        updated[name] = tensor.to(dtype=target.dtype)
        matched += 1

    missing_count = len(missing)
    total_expected = len(state)

    if missing_count:
        for tensor_name, tensor_shape in missing_details:
            print(f"[unet-load-missing] {tensor_name}: shape={tensor_shape}")
        print(f"[unet-load] matched={matched}, missing={missing_count}")
        print(f"[unet-load] expected={total_expected}, loaded={matched}")
        raise SystemExit("[unet-load] aborting because tensors are missing (see above)")
    else:
        print(f"[unet-load] matched={matched}, missing={missing_count}")
        print(f"[unet-load] expected={total_expected}, loaded={matched}")

    state.update(updated)
    unet.load_state_dict(state, strict=False)
