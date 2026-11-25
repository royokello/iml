from __future__ import annotations

_BLOCK_COUNTS = {
    "mode_blocks.image": 5,
    "mode_blocks.image_cond": 5,
    "txt_mode_blocks": 5,
    "single_transformer_blocks": 28,
}

_HIGH_SUFFIXES = (
    "attn.to_v.weight",
    "attn.to_out.0.weight",
    "ff.linear_out.weight",
)

_LOW_SUFFIXES = (
    "attn.to_q.weight",
    "attn.to_k.weight",
    "ff.linear_in.weight",
)

_BF16_SUFFIXES = (
    "attn.norm_q.weight",
    "attn.norm_k.weight",
)

_GLOBAL_LOW = (
    "context_embedder.weight",
)

_GLOBAL_BF16 = (
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
    rank_1: list[str] = []
    rank_2: list[str] = []
    bf16: list[str] = []
    fp16: list[str] = []

    for group, count in _BLOCK_COUNTS.items():
        for block_idx in range(count):
            prefix = f"{group}.{block_idx}."
            for suffix in _HIGH_SUFFIXES:
                rank_1.append(prefix + suffix)
            for suffix in _LOW_SUFFIXES:
                rank_2.append(prefix + suffix)
            for suffix in _BF16_SUFFIXES:
                bf16.append(prefix + suffix)

    rank_2.extend(_GLOBAL_LOW)
    bf16.extend(_GLOBAL_BF16)
    fp16.extend(_GLOBAL_FP16)

    return {"rank_1": rank_1, "rank_2": rank_2, "bf16": bf16, "fp16": fp16}
