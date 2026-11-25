"""LoRA targets and defaults for Krea 2.

The eight names below cover every quantized linear in the 28 transformer blocks
and the 4 text-fusion blocks, 256 modules in total. `attn.to_out` matches
`attn.to_out.0` because the injector ignores numeric path segments.
"""

KREA2_LORA_TARGETS = (
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_gate",
    "attn.to_out",
    "ff.gate",
    "ff.up",
    "ff.down",
)
KREA2_LORA_RANK = 16
KREA2_LORA_ALPHA = 16

__all__ = ["KREA2_LORA_ALPHA", "KREA2_LORA_RANK", "KREA2_LORA_TARGETS"]
