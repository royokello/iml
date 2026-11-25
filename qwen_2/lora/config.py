"""LoRA target names for the Qwen 2 denoiser.

The seven names cover every linear in the 32 transformer blocks that the
external checkpoints touch. `img_mlp.gate_up` is the fused spelling that
ComfyUI and ai-toolkit use. It expands to `gate_layer` and `proj`.
"""

QWEN2_LORA_TARGETS = (
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "img_mlp.gate_layer",
    "img_mlp.proj",
    "img_mlp.out",
)

QWEN2_LORA_FUSED = "img_mlp.gate_up"

DEFAULT_LORA_STRENGTH = 1.0

__all__ = [
    "DEFAULT_LORA_STRENGTH",
    "QWEN2_LORA_FUSED",
    "QWEN2_LORA_TARGETS",
]
