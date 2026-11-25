from __future__ import annotations

_NUM_BLOCKS = 32

_BLOCK_HIGH_SUFFIXES = (
    "attn.to_v.weight",
    "attn.to_out.0.weight",
    "img_mlp.out.weight",
)

_BLOCK_LOW_SUFFIXES = (
    "attn.to_q.weight",
    "attn.to_k.weight",
    "img_mlp.proj.weight",
    "img_mlp.gate_layer.weight",
)

_BLOCK_BF16_SUFFIXES = (
    "attn.norm_q.weight",
    "attn.norm_k.weight",
)

_GLOBAL_BF16 = (
    "modulation.1.weight",
    "time_text_embed.timestep_embedder.linear_1.weight",
    "time_text_embed.timestep_embedder.linear_2.weight",
    "txt_in.text_norm.weight",
    "txt_in.in_layer.weight",
    "txt_in.out_layer.weight",
    "img_in.weight",
    "norm_out.linear.weight",
    "proj_out.weight",
)


def _build_qwen2_denoiser_target_tensors() -> dict[str, list[str]]:
    """Build the quantized and BF16 tensor groups for the denoiser.

    Returns:
        A mapping of group name to checkpoint tensor names.
    """
    high: list[str] = []
    low: list[str] = []
    bf16: list[str] = []

    for block_idx in range(_NUM_BLOCKS):
        prefix = f"transformer_blocks.{block_idx}."
        for suffix in _BLOCK_HIGH_SUFFIXES:
            high.append(prefix + suffix)
        for suffix in _BLOCK_LOW_SUFFIXES:
            low.append(prefix + suffix)
        for suffix in _BLOCK_BF16_SUFFIXES:
            bf16.append(prefix + suffix)

    bf16.extend(_GLOBAL_BF16)

    return {"high": high, "low": low, "bf16": bf16}


__all__ = ["_build_qwen2_denoiser_target_tensors"]
