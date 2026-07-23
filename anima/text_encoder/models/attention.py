import torch
import torch.nn as nn
import torch.nn.functional as F

from anima.denoiser.models.attention import RMSNorm


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states, n_rep):
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Qwen3Attention(nn.Module):
    def __init__(self, hidden_size=1024, num_heads=16, num_kv_heads=8, head_dim=128, device=None, dtype=None):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_heads // num_kv_heads

        inner_dim = num_heads * head_dim
        kv_dim = num_kv_heads * head_dim

        self.q_proj = nn.Linear(hidden_size, inner_dim, bias=False, device=device, dtype=dtype)
        self.q_norm = RMSNorm(head_dim, eps=1e-6)
        self.k_proj = nn.Linear(hidden_size, kv_dim, bias=False, device=device, dtype=dtype)
        self.k_norm = RMSNorm(head_dim, eps=1e-6)
        self.v_proj = nn.Linear(hidden_size, kv_dim, bias=False, device=device, dtype=dtype)
        self.o_proj = nn.Linear(inner_dim, hidden_size, bias=False, device=device, dtype=dtype)

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, self.num_heads, self.head_dim)
        kv_shape = (*input_shape, self.num_kv_heads, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(kv_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(kv_shape).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=query_states.dtype, device=query_states.device)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * torch.finfo(query_states.dtype).min

        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states,
            attn_mask=attention_mask,
            is_causal=attention_mask is None,
        )

        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output
