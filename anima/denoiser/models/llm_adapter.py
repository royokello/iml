import torch
import torch.nn as nn
import torch.nn.functional as F

from .rotary import RotaryEmbedding, apply_rotary_pos_emb
from .attention import RMSNorm


class LLMAdapterAttention(nn.Module):
    def __init__(self, query_dim, context_dim, n_heads, head_dim, device=None, dtype=None):
        super().__init__()
        inner_dim = head_dim * n_heads
        self.n_heads = n_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False, device=device, dtype=dtype)
        self.q_norm = RMSNorm(head_dim, eps=1e-6)
        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False, device=device, dtype=dtype)
        self.k_norm = RMSNorm(head_dim, eps=1e-6)
        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False, device=device, dtype=dtype)
        self.o_proj = nn.Linear(inner_dim, query_dim, bias=False, device=device, dtype=dtype)

    def forward(self, x, context=None, mask=None, position_embeddings=None, position_embeddings_context=None):
        context = x if context is None else context
        input_shape = x.shape[:-1]
        q_shape = (*input_shape, self.n_heads, self.head_dim)
        context_shape = context.shape[:-1]
        kv_shape = (*context_shape, self.n_heads, self.head_dim)

        query_states = self.q_norm(self.q_proj(x).view(q_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(context).view(kv_shape)).transpose(1, 2)
        value_states = self.v_proj(context).view(kv_shape).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states = apply_rotary_pos_emb(query_states, cos, sin)
        if position_embeddings_context is not None:
            cos, sin = position_embeddings_context
            key_states = apply_rotary_pos_emb(key_states, cos, sin)

        attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=mask)
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output


class LLMAdapterBlock(nn.Module):
    def __init__(self, source_dim, model_dim, num_heads, mlp_ratio=4.0, device=None, dtype=None):
        super().__init__()
        head_dim = model_dim // num_heads

        self.norm_self_attn = RMSNorm(model_dim, eps=1e-6)
        self.self_attn = LLMAdapterAttention(model_dim, model_dim, num_heads, head_dim, device=device, dtype=dtype)

        self.norm_cross_attn = RMSNorm(model_dim, eps=1e-6)
        self.cross_attn = LLMAdapterAttention(model_dim, source_dim, num_heads, head_dim, device=device, dtype=dtype)

        self.norm_mlp = RMSNorm(model_dim, eps=1e-6)
        mlp_hidden = int(model_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, mlp_hidden, bias=True, device=device, dtype=dtype),
            nn.GELU(),
            nn.Linear(mlp_hidden, model_dim, bias=True, device=device, dtype=dtype),
        )

    def forward(self, x, context, target_attention_mask=None, source_attention_mask=None,
                position_embeddings=None, position_embeddings_context=None):
        attn_out = self.self_attn(
            self.norm_self_attn(x), mask=target_attention_mask,
            position_embeddings=position_embeddings,
            position_embeddings_context=position_embeddings,
        )
        x = x + attn_out

        attn_out = self.cross_attn(
            self.norm_cross_attn(x), context=context, mask=source_attention_mask,
            position_embeddings=position_embeddings,
            position_embeddings_context=position_embeddings_context,
        )
        x = x + attn_out

        x = x + self.mlp(self.norm_mlp(x))
        return x


class LLMAdapter(nn.Module):
    def __init__(self, source_dim=1024, target_dim=1024, model_dim=1024, num_layers=6, num_heads=16,
                 mlp_ratio=4.0, device=None, dtype=None):
        super().__init__()
        self.embed = nn.Embedding(32128, target_dim, device=device, dtype=dtype)
        self.in_proj = nn.Identity() if model_dim == target_dim else nn.Linear(target_dim, model_dim, device=device, dtype=dtype)
        self.rotary_emb = RotaryEmbedding(model_dim // num_heads)
        self.blocks = nn.ModuleList([
            LLMAdapterBlock(source_dim, model_dim, num_heads, mlp_ratio=mlp_ratio, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(model_dim, target_dim, bias=True, device=device, dtype=dtype)
        self.norm = RMSNorm(target_dim, eps=1e-6)

    def forward(self, source_hidden_states, target_input_ids, target_attention_mask=None, source_attention_mask=None):
        if target_attention_mask is not None:
            target_attention_mask = target_attention_mask.to(torch.bool)
            if target_attention_mask.ndim == 2:
                target_attention_mask = target_attention_mask.unsqueeze(1).unsqueeze(1)
        if source_attention_mask is not None:
            source_attention_mask = source_attention_mask.to(torch.bool)
            if source_attention_mask.ndim == 2:
                source_attention_mask = source_attention_mask.unsqueeze(1).unsqueeze(1)

        context = source_hidden_states
        x = self.in_proj(self.embed(target_input_ids).to(dtype=context.dtype))
        position_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        position_ids_context = torch.arange(context.shape[1], device=x.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(x, position_ids)
        position_embeddings_context = self.rotary_emb(context, position_ids_context)

        for block in self.blocks:
            x = block(x, context,
                      target_attention_mask=target_attention_mask,
                      source_attention_mask=source_attention_mask,
                      position_embeddings=position_embeddings,
                      position_embeddings_context=position_embeddings_context)
        return self.norm(self.out_proj(x))
