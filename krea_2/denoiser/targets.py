from __future__ import annotations

_NUM_TRANSFORMER_BLOCKS = 28
_NUM_LAYERWISE_BLOCKS = 2
_NUM_REFINER_BLOCKS = 2

_HIGH_SUFFIXES = (
    "attn.to_v.weight",
    "attn.to_out.0.weight",
    "ff.down.weight",
)

_LOW_SUFFIXES = (
    "attn.to_q.weight",
    "attn.to_k.weight",
    "attn.to_gate.weight",
    "ff.gate.weight",
    "ff.up.weight",
)

_FP32_SUFFIXES = (
    "attn.norm_k.weight",
    "attn.norm_q.weight",
    "norm1.weight",
    "norm2.weight",
)

_GLOBAL_HIGH = (
)

_GLOBAL_FP32 = (
    "img_in.bias",
    "img_in.weight",
    "txt_in.norm.weight",
    "txt_in.linear_1.weight",
    "txt_in.linear_1.bias",
    "txt_in.linear_2.weight",
    "txt_in.linear_2.bias",
    "final_layer.norm.weight",
    "text_fusion.projector.weight",
    "time_embed.linear_1.weight",
    "time_embed.linear_1.bias",
    "time_embed.linear_2.weight",
    "time_embed.linear_2.bias",
    "final_layer.linear.weight",
    "final_layer.linear.bias",
    "final_layer.scale_shift_table",
)

_GLOBAL_FP16 = (
    "time_mod_proj.weight",
    "time_mod_proj.bias",
)


def _build_krea2_denoiser_target_tensors() -> dict[str, list[str]]:
    high: list[str] = []
    low: list[str] = []
    fp32: list[str] = []
    fp16: list[str] = []

    for i in range(_NUM_TRANSFORMER_BLOCKS):
        prefix = f"transformer_blocks.{i}."
        for s in _HIGH_SUFFIXES:
            high.append(prefix + s)
        for s in _LOW_SUFFIXES:
            low.append(prefix + s)
        for s in _FP32_SUFFIXES:
            fp32.append(prefix + s)
        fp32.append(f"{prefix}scale_shift_table")

    for i in range(_NUM_LAYERWISE_BLOCKS):
        prefix = f"text_fusion.layerwise_blocks.{i}."
        for s in _HIGH_SUFFIXES:
            high.append(prefix + s)
        for s in _LOW_SUFFIXES:
            low.append(prefix + s)
        for s in _FP32_SUFFIXES:
            fp32.append(prefix + s)

    for i in range(_NUM_REFINER_BLOCKS):
        prefix = f"text_fusion.refiner_blocks.{i}."
        for s in _HIGH_SUFFIXES:
            high.append(prefix + s)
        for s in _LOW_SUFFIXES:
            low.append(prefix + s)
        for s in _FP32_SUFFIXES:
            fp32.append(prefix + s)

    for name in _GLOBAL_HIGH:
        high.append(name)
    for name in _GLOBAL_FP32:
        fp32.append(name)
    for name in _GLOBAL_FP16:
        fp16.append(name)

    return {"high": high, "low": low, "fp32": fp32, "fp16": fp16}
