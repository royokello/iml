import torch
import torch.nn as nn

from .attention import Attention


class GPT2FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.layer1 = nn.Linear(d_model, d_ff, bias=False, device=device, dtype=dtype)
        self.act = nn.GELU()
        self.layer2 = nn.Linear(d_ff, d_model, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        return self.layer2(self.act(self.layer1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, x_dim, context_dim, num_heads, head_dim, mlp_ratio=4.0, adaln_lora_dim=256, device=None, dtype=None):
        super().__init__()
        self.norm_self_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6, device=device, dtype=dtype)
        self.self_attn = Attention(x_dim, x_dim, num_heads, head_dim, device=device, dtype=dtype)

        self.norm_cross_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6, device=device, dtype=dtype)
        self.cross_attn = Attention(x_dim, context_dim, num_heads, head_dim, device=device, dtype=dtype)

        self.norm_mlp = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6, device=device, dtype=dtype)
        mlp_hidden = int(x_dim * mlp_ratio)
        self.mlp = GPT2FeedForward(x_dim, mlp_hidden, device=device, dtype=dtype)

        self.adaln_modulation_self_attn = nn.Sequential(
            nn.SiLU(),
            nn.Linear(x_dim, adaln_lora_dim, bias=False, device=device, dtype=dtype),
            nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False, device=device, dtype=dtype),
        )
        self.adaln_modulation_cross_attn = nn.Sequential(
            nn.SiLU(),
            nn.Linear(x_dim, adaln_lora_dim, bias=False, device=device, dtype=dtype),
            nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False, device=device, dtype=dtype),
        )
        self.adaln_modulation_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(x_dim, adaln_lora_dim, bias=False, device=device, dtype=dtype),
            nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False, device=device, dtype=dtype),
        )

    def forward(self, x, emb, context, position_embeddings=None, adaln_lora=None, mask=None):
        mod_out = self.adaln_modulation_self_attn(emb)
        if adaln_lora is not None:
            mod_out = mod_out + adaln_lora
        shift, scale, gate = mod_out.chunk(3, dim=-1)
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
        gate = gate.unsqueeze(1)

        normed = self.norm_self_attn(x)
        normed = normed * (1 + scale) + shift
        attn_out = self.self_attn(normed, position_embeddings=position_embeddings, mask=mask)
        x = x + gate * attn_out

        mod_out = self.adaln_modulation_cross_attn(emb)
        if adaln_lora is not None:
            mod_out = mod_out + adaln_lora
        shift, scale, gate = mod_out.chunk(3, dim=-1)
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
        gate = gate.unsqueeze(1)

        normed = self.norm_cross_attn(x)
        normed = normed * (1 + scale) + shift
        attn_out = self.cross_attn(normed, context=context, mask=mask)
        x = x + gate * attn_out

        mod_out = self.adaln_modulation_mlp(emb)
        if adaln_lora is not None:
            mod_out = mod_out + adaln_lora
        shift, scale, gate = mod_out.chunk(3, dim=-1)
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
        gate = gate.unsqueeze(1)

        normed = self.norm_mlp(x)
        normed = normed * (1 + scale) + shift
        mlp_out = self.mlp(normed)
        x = x + gate * mlp_out

        return x
