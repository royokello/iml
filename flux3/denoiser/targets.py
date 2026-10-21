from __future__ import annotations

_BLOCK_COUNTS = {
    "mode_blocks.image": 5,
    "mode_blocks.image_cond": 5,
    "txt_mode_blocks": 5,
    "single_transformer_blocks": 28,
}

_HIGH_SUFFIXES = (
    "attn.to_v.weight",
    "ff.linear_out.weight",
)

_LOW_SUFFIXES = (
    "attn.to_q.weight",
    "attn.to_k.weight",
    "attn.to_out.0.weight",
    "ff.linear_in.weight",
)

_FP32_SUFFIXES = (
    "attn.norm_q.weight",
    "attn.norm_k.weight",
)

_GLOBAL_LOW = (
    "context_embedder.weight",
)

_GLOBAL_FP32 = (
    "x_embedders.image.weight",
    "x_embedders.image_cond.weight",
    "timestep_embedder.linear_1.weight",
    "timestep_embedder.linear_2.weight",
    "target_timestep_embedder.linear_1.weight",
    "target_timestep_embedder.linear_2.weight",
    "norm_out.linear.weight",
    "proj_out.weight",
)

_GLOBAL_FP16 = (
    "early_stream_modulations.image.linear.weight",
    "early_stream_modulations.image_cond.linear.weight",
    "early_stream_modulations.txt.linear.weight",
    "single_stream_modulations.image.linear.weight",
    "single_stream_modulations.image_cond.linear.weight",
    "single_stream_modulations.txt.linear.weight",
)


def _build_flux3_denoiser_target_tensors() -> dict[str, list[str]]:
    high: list[str] = []
    low: list[str] = []
    fp32: list[str] = []
    fp16: list[str] = []

    for group, count in _BLOCK_COUNTS.items():
        for block_idx in range(count):
            prefix = f"{group}.{block_idx}."
            for suffix in _HIGH_SUFFIXES:
                high.append(prefix + suffix)
            for suffix in _LOW_SUFFIXES:
                low.append(prefix + suffix)
            for suffix in _FP32_SUFFIXES:
                fp32.append(prefix + suffix)

    low.extend(_GLOBAL_LOW)
    fp32.extend(_GLOBAL_FP32)
    fp16.extend(_GLOBAL_FP16)

    return {"high": high, "low": low, "fp32": fp32, "fp16": fp16}
