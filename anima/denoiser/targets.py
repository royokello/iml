from __future__ import annotations

_NUM_MAIN_BLOCKS = 28
_NUM_LLM_ADAPTER_BLOCKS = 6

_HIGH_SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.output_proj.weight",
    "cross_attn.q_proj.weight",
    "cross_attn.v_proj.weight",
    "cross_attn.output_proj.weight",
    "mlp.layer1.weight",
    "mlp.layer2.weight",
)

_LOW_SUFFIXES = (
    "self_attn.k_proj.weight",
    "cross_attn.k_proj.weight",
)

_FP32_SUFFIXES = (
    "self_attn.k_norm.weight",
    "self_attn.q_norm.weight",
    "cross_attn.k_norm.weight",
    "cross_attn.q_norm.weight",
)

_ADALN_SUFFIXES = (
    "adaln_modulation_cross_attn.1.weight",
    "adaln_modulation_cross_attn.2.weight",
    "adaln_modulation_mlp.1.weight",
    "adaln_modulation_mlp.2.weight",
    "adaln_modulation_self_attn.1.weight",
    "adaln_modulation_self_attn.2.weight",
)

_LLM_ADAPTER_HIGH_SUFFIXES = (
    "mlp.0.weight",
    "mlp.2.weight",
)

_LLM_ADAPTER_LOW_SUFFIXES = (
    "cross_attn.k_proj.weight",
    "cross_attn.q_proj.weight",
    "cross_attn.v_proj.weight",
    "cross_attn.o_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.q_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
)

_LLM_ADAPTER_FP32_SUFFIXES = (
    "cross_attn.k_norm.weight",
    "cross_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "self_attn.q_norm.weight",
    "norm_cross_attn.weight",
    "norm_mlp.weight",
    "norm_self_attn.weight",
)

_LLM_ADAPTER_FP16_SUFFIXES = (
    "mlp.0.bias",
    "mlp.2.bias",
)

_GLOBAL_HIGH = (
    "net.t_embedder.1.linear_1.weight",
    "net.t_embedder.1.linear_2.weight",
)

_GLOBAL_LOW = (
)

_GLOBAL_FP32 = (
    "net.final_layer.adaln_modulation.1.weight",
    "net.final_layer.adaln_modulation.2.weight",
    "net.final_layer.linear.weight",
    "net.llm_adapter.norm.weight",
    "net.llm_adapter.out_proj.weight",
    "net.llm_adapter.out_proj.bias",
    "net.t_embedding_norm.weight",
    "net.x_embedder.proj.1.weight",
)


_GLOBAL_FP16 = (
    "net.llm_adapter.embed.weight",
)

def _build_anima_denoiser_target_tensors() -> dict[str, list[str]]:
    high: list[str] = []
    low: list[str] = []
    fp32: list[str] = []
    fp16: list[str] = []

    for i in range(_NUM_MAIN_BLOCKS):
        prefix = f"net.blocks.{i}."
        for s in _HIGH_SUFFIXES:
            high.append(prefix + s)
        for s in _LOW_SUFFIXES:
            low.append(prefix + s)
        for s in _FP32_SUFFIXES:
            fp32.append(prefix + s)
        for s in _ADALN_SUFFIXES:
            fp32.append(prefix + s)

    for i in range(_NUM_LLM_ADAPTER_BLOCKS):
        prefix = f"net.llm_adapter.blocks.{i}."
        for s in _LLM_ADAPTER_HIGH_SUFFIXES:
            high.append(prefix + s)
        for s in _LLM_ADAPTER_LOW_SUFFIXES:
            low.append(prefix + s)
        for s in _LLM_ADAPTER_FP32_SUFFIXES:
            fp32.append(prefix + s)
        for s in _LLM_ADAPTER_FP16_SUFFIXES:
            fp16.append(prefix + s)

    for name in _GLOBAL_HIGH:
        high.append(name)
    for name in _GLOBAL_LOW:
        low.append(name)
    for name in _GLOBAL_FP32:
        fp32.append(name)
    for name in _GLOBAL_FP16:
        fp16.append(name)

    return {"high": high, "low": low, "fp32": fp32, "fp16": fp16}
