from __future__ import annotations

# The SDXL UNet has 743 rank-2 linear weights. Each one falls into exactly one
# group. A name that matches no suffix goes to `rank_1`, which is the safe default.
#
# `rank_1`  attention projections and the spatial transformer input/output
#           projections.
# `rank_2`  feed-forward layers inside the transformer blocks. These carry most of
#           the parameters, so a `-mini` or `-max` variant puts them at the lower
#           method.
# `fp16`  the time and additive embeddings. They are small and sensitive.

_FP16_MODULE_NAMES = (
    "time_embedding.linear_1",
    "time_embedding.linear_2",
    "add_embedding.linear_1",
    "add_embedding.linear_2",
)

_FP16_MODULE_SUFFIXES = ("time_emb_proj",)

_LOW_MODULE_SUFFIXES = (
    "ff.net.0.proj",
    "ff.net.2",
)

_HIGH_MODULE_SUFFIXES = (
    "attn1.to_q",
    "attn1.to_k",
    "attn1.to_v",
    "attn1.to_out.0",
    "attn2.to_q",
    "attn2.to_k",
    "attn2.to_v",
    "attn2.to_out.0",
    "proj_in",
    "proj_out",
)


def classify_linear_module(module_name: str) -> str:
    """Return the target group for one linear module.

    Args:
        module_name: The module path without the `.weight` suffix, for example
            `down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q`.

    Returns:
        One of `rank_1`, `rank_2`, or `fp16`.
    """
    if module_name in _FP16_MODULE_NAMES:
        return "fp16"
    if module_name.endswith(_FP16_MODULE_SUFFIXES):
        return "fp16"
    if module_name.endswith(_LOW_MODULE_SUFFIXES):
        return "rank_2"
    if module_name.endswith(_HIGH_MODULE_SUFFIXES):
        return "rank_1"
    return "rank_1"


def build_sdxl_denoiser_target_tensors(
    linear_weight_names: list[str] | tuple[str, ...],
) -> dict[str, list[str]]:
    """Group the UNet linear weights by quantization target.

    Args:
        linear_weight_names: The rank-2 `.weight` names of the UNet, for example
            `down_blocks.0.resnets.0.time_emb_proj.weight`.

    Returns:
        A mapping with the `rank_1`, `rank_2`, and `fp16` groups. Every input name
        appears in exactly one group.
    """
    groups: dict[str, list[str]] = {"rank_1": [], "rank_2": [], "fp16": []}
    for name in linear_weight_names:
        module_name = name.removesuffix(".weight")
        groups[classify_linear_module(module_name)].append(name)
    return groups


__all__ = ["build_sdxl_denoiser_target_tensors", "classify_linear_module"]
