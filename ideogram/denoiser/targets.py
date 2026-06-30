from __future__ import annotations

_NUM_LAYERS = 34

_EDGE_LAYERS = {0, 1, _NUM_LAYERS - 2, _NUM_LAYERS - 1}

_HIGH_SUFFIXES = (
    "attention.o.weight",
    "feed_forward.w2.weight",
    "adaln_modulation.weight",
)

_LOW_SUFFIXES = (
    "attention.qkv.weight",
    "feed_forward.w1.weight",
    "feed_forward.w3.weight",
)

_FP32_SUFFIXES = (
    "attention.norm_q.weight",
    "attention.norm_k.weight",
    "attention_norm1.weight",
    "attention_norm2.weight",
    "ffn_norm1.weight",
    "ffn_norm2.weight",
)


def _build_ideogram_denoiser_target_tensors() -> dict[str, list[str]]:
    high: list[str] = []
    low: list[str] = []
    fp32: list[str] = []

    for layer_idx in range(_NUM_LAYERS):
        prefix = f"layers.{layer_idx}."

        for suffix in _FP32_SUFFIXES:
            fp32.append(prefix + suffix)

        if layer_idx in _EDGE_LAYERS:
            for suffix in _HIGH_SUFFIXES:
                high.append(prefix + suffix)
            for suffix in _LOW_SUFFIXES:
                high.append(prefix + suffix)
        else:
            for suffix in _HIGH_SUFFIXES:
                high.append(prefix + suffix)
            for suffix in _LOW_SUFFIXES:
                low.append(prefix + suffix)

    high.append("adaln_proj.weight")
    high.append("final_layer.linear.weight")
    high.append("final_layer.adaln_modulation.weight")
    high.append("input_proj.weight")
    high.append("llm_cond_proj.weight")
    high.append("t_embedding.mlp_in.weight")
    high.append("t_embedding.mlp_out.weight")

    fp32.append("llm_cond_norm.weight")

    fp16: list[str] = []
    for layer_idx in range(_NUM_LAYERS):
        fp16.append(f"layers.{layer_idx}.adaln_modulation.bias")
    fp16.append("adaln_proj.bias")
    fp16.append("final_layer.linear.bias")
    fp16.append("final_layer.adaln_modulation.bias")
    fp16.append("input_proj.bias")
    fp16.append("llm_cond_proj.bias")
    fp16.append("t_embedding.mlp_in.bias")
    fp16.append("t_embedding.mlp_out.bias")
    fp16.append("embed_image_indicator.weight")

    return {"high": high, "low": low, "fp32": fp32, "fp16": fp16}
