from __future__ import annotations

_NUM_BLOCKS = 32

# The rank order follows the published dynamic GGUF recipes for this model:
# `to_q` and `to_out` are never below the block median, `to_v` sits under them,
# the image MLP moves as one unit, and `to_k` takes the fewest bits. The shared
# ladder turns these three ranks into one method each.
_BLOCK_HIGH_SUFFIXES = (
    "attn.to_q.weight",
    "attn.to_out.0.weight",
)

_BLOCK_MEDIUM_SUFFIXES = (
    "attn.to_v.weight",
    "img_mlp.proj.weight",
    "img_mlp.gate_layer.weight",
    "img_mlp.out.weight",
)

_BLOCK_LOW_SUFFIXES = ("attn.to_k.weight",)

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
    rank_1: list[str] = []
    rank_2: list[str] = []
    rank_3: list[str] = []
    bf16: list[str] = []

    for block_idx in range(_NUM_BLOCKS):
        prefix = f"transformer_blocks.{block_idx}."
        for suffix in _BLOCK_HIGH_SUFFIXES:
            rank_1.append(prefix + suffix)
        for suffix in _BLOCK_MEDIUM_SUFFIXES:
            rank_2.append(prefix + suffix)
        for suffix in _BLOCK_LOW_SUFFIXES:
            rank_3.append(prefix + suffix)
        for suffix in _BLOCK_BF16_SUFFIXES:
            bf16.append(prefix + suffix)

    bf16.extend(_GLOBAL_BF16)

    return {"rank_1": rank_1, "rank_2": rank_2, "rank_3": rank_3, "bf16": bf16}


__all__ = ["_build_qwen2_denoiser_target_tensors"]
