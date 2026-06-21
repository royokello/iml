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
    fp32: list[str] = []

    for block_idx in range(_DOUBLE_BLOCKS[version]):
        prefix = f"transformer_blocks.{block_idx}."

        high_tensors = [
            f"{prefix}attn.to_v.weight",
            f"{prefix}attn.to_out.0.weight",
            f"{prefix}attn.add_v_proj.weight",
            f"{prefix}attn.to_add_out.weight",
            f"{prefix}ff.linear_out.weight",
            f"{prefix}ff_context.linear_out.weight",
        ]

        low_tensors = [
            f"{prefix}attn.to_q.weight",
            f"{prefix}attn.to_k.weight",
            f"{prefix}attn.add_q_proj.weight",
            f"{prefix}attn.add_k_proj.weight",
            f"{prefix}ff.linear_in.weight",
            f"{prefix}ff_context.linear_in.weight",
        ]

        if (version == "9b" and block_idx in (0, 7)) or (version == "4b" and block_idx in (0, 4)):
            high.extend(high_tensors + low_tensors)
        else:
            high.extend(high_tensors)
            low.extend(low_tensors)

        fp32.extend([
            f"{prefix}attn.norm_q.weight",
            f"{prefix}attn.norm_k.weight",
            f"{prefix}attn.norm_added_q.weight",
            f"{prefix}attn.norm_added_k.weight",
        ])

    for block_idx in range(_SINGLE_BLOCKS[version]):
        prefix = f"single_transformer_blocks.{block_idx}."

        if (version == "9b" and block_idx in (0, 1, 22, 23)) or (version == "4b" and block_idx in (0, 1, 18, 19)):
            high.append(f"{prefix}attn.to_qkv_mlp_proj.weight")
            high.append(f"{prefix}attn.to_out.weight")
        elif version in ("9b", "4b"):
            low.append(f"{prefix}attn.to_qkv_mlp_proj.weight")
            low.append(f"{prefix}attn.to_out.weight")
        else:
            high.append(f"{prefix}attn.to_qkv_mlp_proj.weight")
            high.append(f"{prefix}attn.to_out.weight")

        fp32.append(f"{prefix}attn.norm_q.weight")
        fp32.append(f"{prefix}attn.norm_k.weight")

    return {"high": high, "low": low, "fp32": fp32}