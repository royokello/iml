FLUX2_LORA_TARGETS = (
    "transformer_blocks.attn.to_q",
    "transformer_blocks.attn.to_k",
    "transformer_blocks.attn.to_v",
    "transformer_blocks.attn.to_out",
    "transformer_blocks.attn.add_q_proj",
    "transformer_blocks.attn.add_k_proj",
    "transformer_blocks.attn.add_v_proj",
    "transformer_blocks.attn.to_add_out",
    "transformer_blocks.ff.linear_in",
    "transformer_blocks.ff.linear_out",
    "transformer_blocks.ff_context.linear_in",
    "transformer_blocks.ff_context.linear_out",
    "single_transformer_blocks.attn.to_qkv_mlp_proj",
    "single_transformer_blocks.attn.to_out",
)
FLUX2_LORA_RANK = 32
FLUX2_LORA_ALPHA = 32
