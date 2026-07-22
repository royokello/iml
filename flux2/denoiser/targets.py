from __future__ import annotations

_DOUBLE_BLOCKS = {
    "4b": 5,
    "9b": 8,
}
_SINGLE_BLOCKS = {
    "4b": 20,
    "9b": 24,
}

_DOUBLE_HIGH_SUFFIXES = (
    "attn.to_v.weight",
    "attn.to_out.0.weight",
    "attn.add_v_proj.weight",
    "attn.to_add_out.weight",
    "ff.linear_out.weight",
    "ff_context.linear_out.weight",
)

_DOUBLE_LOW_SUFFIXES = (
    "attn.to_q.weight",
    "attn.to_k.weight",
    "attn.add_q_proj.weight",
    "attn.add_k_proj.weight",
    "ff.linear_in.weight",
    "ff_context.linear_in.weight",
)

_DOUBLE_FP32_SUFFIXES = (
    "attn.norm_q.weight",
    "attn.norm_k.weight",
    "attn.norm_added_q.weight",
    "attn.norm_added_k.weight",
)

_SINGLE_SUFFIXES = (
    "attn.to_qkv_mlp_proj.weight",
    "attn.to_out.weight",
)

_SINGLE_FP32_SUFFIXES = (
    "attn.norm_q.weight",
    "attn.norm_k.weight",
)

_GLOBAL_FP32 = (
    "x_embedder.weight",
    "context_embedder.weight",
    "time_guidance_embed.timestep_embedder.linear_1.weight",
    "time_guidance_embed.timestep_embedder.linear_2.weight",
    "norm_out.linear.weight",
    "proj_out.weight",
)

_GLOBAL_FP16 = (
    "double_stream_modulation_img.linear.weight",
    "double_stream_modulation_txt.linear.weight",
    "single_stream_modulation.linear.weight",
)


def _build_flux2_denoiser_target_tensors(version: str) -> dict[str, list[str]]:
    high: list[str] = []
    low: list[str] = []
    fp32: list[str] = []
    fp16: list[str] = []

    num_double = _DOUBLE_BLOCKS[version]
    num_single = _SINGLE_BLOCKS[version]

    # Double (dual-stream) blocks
    double_edge = {0, num_double - 1}
    for block_idx in range(num_double):
        prefix = f"transformer_blocks.{block_idx}."
        all_high = block_idx in double_edge
        for s in _DOUBLE_HIGH_SUFFIXES:
            high.append(prefix + s)
        if all_high:
            for s in _DOUBLE_LOW_SUFFIXES:
                high.append(prefix + s)
        else:
            for s in _DOUBLE_LOW_SUFFIXES:
                low.append(prefix + s)
        for s in _DOUBLE_FP32_SUFFIXES:
            fp32.append(prefix + s)

    # Single (self-attention) blocks
    single_edge = {0, 1, num_single - 2, num_single - 1}
    for block_idx in range(num_single):
        prefix = f"single_transformer_blocks.{block_idx}."
        if block_idx in single_edge:
            for s in _SINGLE_SUFFIXES:
                high.append(prefix + s)
        else:
            for s in _SINGLE_SUFFIXES:
                low.append(prefix + s)
        for s in _SINGLE_FP32_SUFFIXES:
            fp32.append(prefix + s)

    for name in _GLOBAL_FP32:
        fp32.append(name)
    for name in _GLOBAL_FP16:
        fp16.append(name)

    return {"high": high, "low": low, "fp32": fp32, "fp16": fp16}
