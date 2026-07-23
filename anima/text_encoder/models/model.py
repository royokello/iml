import torch
import torch.nn as nn

from anima.denoiser.models.attention import RMSNorm
from anima.denoiser.models.rotary import RotaryEmbedding
from .decoder_layer import Qwen3DecoderLayer


class Qwen3Model(nn.Module):
    def __init__(self, vocab_size=151936, hidden_size=1024, num_hidden_layers=28,
                 num_heads=16, num_kv_heads=8, head_dim=128, intermediate_size=3072,
                 device=None, dtype=None):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, device=device, dtype=dtype)
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(hidden_size, num_heads, num_kv_heads, head_dim,
                              intermediate_size, device=device, dtype=dtype)
            for _ in range(num_hidden_layers)
        ])
        self.norm = RMSNorm(hidden_size, eps=1e-6)
        self.rotary_emb = RotaryEmbedding(head_dim, theta=1000000.0)

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.embed_tokens(input_ids)
        position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings=position_embeddings, attention_mask=attention_mask)

        hidden_states = self.norm(hidden_states)
        return hidden_states
