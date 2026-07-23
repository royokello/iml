import torch.nn as nn

from anima.denoiser.models.attention import RMSNorm
from .attention import Qwen3Attention
from .mlp import Qwen3MLP


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, hidden_size=1024, num_heads=16, num_kv_heads=8, head_dim=128,
                 intermediate_size=3072, device=None, dtype=None):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, eps=1e-6)
        self.self_attn = Qwen3Attention(hidden_size, num_heads, num_kv_heads, head_dim, device=device, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = Qwen3MLP(hidden_size, intermediate_size, device=device, dtype=dtype)

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings=position_embeddings, attention_mask=attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
