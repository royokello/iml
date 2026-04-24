import torch

from diffusers.models.attention_dispatch import dispatch_attention_fn

from ..kv.layer_cache import Flux2KVLayerCache


def _flux2_kv_causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_txt_tokens: int,
    num_ref_tokens: int,
    kv_cache: Flux2KVLayerCache | None = None,
    backend=None,
) -> torch.Tensor:
    """Causal attention for KV caching where reference tokens only self-attend.

    All tensors use the diffusers convention: (batch_size, seq_len, num_heads, head_dim).

    Without cache (extract mode): sequence layout is [txt, ref, img]. txt+img tokens attend to all tokens, ref tokens
    only attend to themselves. With cache (cached mode): sequence layout is [txt, img]. Cached ref K/V are injected
    between txt and img.
    """
    # No ref tokens and no cache — standard full attention
    if num_ref_tokens == 0 and kv_cache is None:
        return dispatch_attention_fn(query, key, value, backend=backend)

    if kv_cache is not None:
        # Cached mode: inject ref K/V between txt and img
        k_ref, v_ref = kv_cache.get()

        k_all = torch.cat([key[:, :num_txt_tokens], k_ref, key[:, num_txt_tokens:]], dim=1)
        v_all = torch.cat([value[:, :num_txt_tokens], v_ref, value[:, num_txt_tokens:]], dim=1)

        return dispatch_attention_fn(query, k_all, v_all, backend=backend)

    # Extract mode: ref tokens self-attend, txt+img attend to all
    ref_start = num_txt_tokens
    ref_end = num_txt_tokens + num_ref_tokens

    q_txt = query[:, :ref_start]
    q_ref = query[:, ref_start:ref_end]
    q_img = query[:, ref_end:]

    k_txt = key[:, :ref_start]
    k_ref = key[:, ref_start:ref_end]
    k_img = key[:, ref_end:]

    v_txt = value[:, :ref_start]
    v_ref = value[:, ref_start:ref_end]
    v_img = value[:, ref_end:]

    # txt+img attend to all tokens
    q_txt_img = torch.cat([q_txt, q_img], dim=1)
    k_all = torch.cat([k_txt, k_ref, k_img], dim=1)
    v_all = torch.cat([v_txt, v_ref, v_img], dim=1)
    attn_txt_img = dispatch_attention_fn(q_txt_img, k_all, v_all, backend=backend)
    attn_txt = attn_txt_img[:, :ref_start]
    attn_img = attn_txt_img[:, ref_start:]

    # ref tokens self-attend only
    attn_ref = dispatch_attention_fn(q_ref, k_ref, v_ref, backend=backend)

    return torch.cat([attn_txt, attn_ref, attn_img], dim=1)
