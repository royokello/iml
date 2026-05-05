from typing import Dict


_DOUBLE_BLOCKS = {
    "4b": 5,
    "9b": 8,
}
_SINGLE_BLOCKS = {
    "4b": 20,
    "9b": 24,
}

def _build_flux2_denoiser_target_tensors(version: str) -> Dict[str, list[str]]:
    high: list[str] = []
    low: list[str] = []

    for block_idx in range(_DOUBLE_BLOCKS[version]):
        prefix = f"transformer_blocks.{block_idx}."

        # HIGH priority: Value, Output, and FFN-out projections
        high.extend([
            f"{prefix}attn.to_v.weight",
            f"{prefix}attn.to_out.0.weight",
            f"{prefix}attn.add_v_proj.weight",
            f"{prefix}attn.to_add_out.weight",
            f"{prefix}ff.linear_out.weight",
            f"{prefix}ff_context.linear_out.weight",
        ])

        # LOW priority: Queries, Keys, and FFN-in (up) projections
        low.extend([
            f"{prefix}attn.to_q.weight",
            f"{prefix}attn.to_k.weight",
            f"{prefix}attn.add_q_proj.weight",
            f"{prefix}attn.add_k_proj.weight",
            f"{prefix}ff.linear_in.weight",
            f"{prefix}ff_context.linear_in.weight",
        ])

    for block_idx in range(_SINGLE_BLOCKS[version]):
        prefix = f"single_transformer_blocks.{block_idx}."

        # HIGH priority: the fused QKV+MLP and the output projection
        high.append(f"{prefix}attn.to_qkv_mlp_proj.weight")
        high.append(f"{prefix}attn.to_out.weight")

    return {"high": high, "low": low}