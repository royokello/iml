from __future__ import annotations

_WAN22_TEXT_ENCODER_BLOCKS = 24
_WAN22_TEXT_ENCODER_HIGH_SUFFIXES = (
    "attn.v.weight",
    "attn.o.weight",
    "ffn.fc2.weight",
)
_WAN22_TEXT_ENCODER_LOW_SUFFIXES = (
    "attn.q.weight",
    "attn.k.weight",
    "ffn.gate.0.weight",
    "ffn.fc1.weight",
)

_WAN22_DENOISER_BLOCKS = 30
_WAN22_DENOISER_HIGH_SUFFIXES = (
    "self_attn.v.weight",
    "self_attn.o.weight",
    "cross_attn.v.weight",
    "cross_attn.o.weight",
    "ffn.2.weight",
)
_WAN22_DENOISER_LOW_SUFFIXES = (
    "self_attn.q.weight",
    "self_attn.k.weight",
    "cross_attn.q.weight",
    "cross_attn.k.weight",
    "ffn.0.weight",
)


def _build_block_targets(
    *,
    block_count: int,
    suffixes: tuple[str, ...],
) -> list[str]:
    tensors: list[str] = []
    for block_idx in range(block_count):
        block_prefix = f"blocks.{block_idx}."
        for suffix in suffixes:
            tensors.append(block_prefix + suffix)
    return tensors


def _build_split_block_targets(
    *,
    block_count: int,
    high_suffixes: tuple[str, ...],
    low_suffixes: tuple[str, ...],
) -> dict[str, list[str]]:
    return {
        "high": _build_block_targets(block_count=block_count, suffixes=high_suffixes),
        "low": _build_block_targets(block_count=block_count, suffixes=low_suffixes),
    }


def _build_wan22_text_encoder_mixed_target_tensors() -> dict[str, list[str]]:
    return _build_split_block_targets(
        block_count=_WAN22_TEXT_ENCODER_BLOCKS,
        high_suffixes=_WAN22_TEXT_ENCODER_HIGH_SUFFIXES,
        low_suffixes=_WAN22_TEXT_ENCODER_LOW_SUFFIXES,
    )


def _build_wan22_denoiser_mixed_target_tensors() -> dict[str, list[str]]:
    return _build_split_block_targets(
        block_count=_WAN22_DENOISER_BLOCKS,
        high_suffixes=_WAN22_DENOISER_HIGH_SUFFIXES,
        low_suffixes=_WAN22_DENOISER_LOW_SUFFIXES,
    )


__all__ = [
    "_build_wan22_denoiser_mixed_target_tensors",
    "_build_wan22_text_encoder_mixed_target_tensors",
]
