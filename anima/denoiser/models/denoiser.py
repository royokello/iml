import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from .adaln import AdaLN
from .attention import RMSNorm
from .llm_adapter import LLMAdapter
from .transformer import TransformerBlock


class Timesteps(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps):
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / (half_dim - 0.0)
        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]
        sin_emb = torch.sin(emb)
        cos_emb = torch.cos(emb)
        emb = torch.cat([cos_emb, sin_emb], dim=-1)
        return emb


class TimestepEmbedding(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.linear_1 = nn.Linear(in_features, out_features, bias=False, device=device, dtype=dtype)
        self.activation = nn.SiLU()
        self.linear_2 = nn.Linear(out_features, 3 * out_features, bias=False, device=device, dtype=dtype)

    def forward(self, sample):
        emb = self.linear_1(sample)
        emb = self.activation(emb)
        emb = self.linear_2(emb)
        return sample, emb


class PatchEmbed(nn.Module):
    def __init__(self, spatial_patch_size, temporal_patch_size, in_channels, out_channels, device=None, dtype=None):
        super().__init__()
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        self.proj = nn.Sequential(
            Rearrange("b c (t r) (h m) (w n) -> b t h w (c r m n)",
                       r=temporal_patch_size, m=spatial_patch_size, n=spatial_patch_size),
            nn.Linear(in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size,
                      out_channels, bias=False, device=device, dtype=dtype),
        )

    def forward(self, x):
        return self.proj(x)


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_spatial, patch_temporal, out_channels, adaln_lora_dim=256, device=None, dtype=None):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, device=device, dtype=dtype)
        self.linear = nn.Linear(
            hidden_size,
            patch_spatial * patch_spatial * patch_temporal * out_channels,
            bias=False, device=device, dtype=dtype,
        )
        self.adaln_modulation = AdaLN(hidden_size, lora_dim=adaln_lora_dim, n_chunks=2, device=device, dtype=dtype)

    def forward(self, x, emb, adaln_lora=None):
        shift, scale = self.adaln_modulation(emb, adaln_lora=adaln_lora)
        shift = shift.unsqueeze(1).unsqueeze(1)
        scale = scale.unsqueeze(1).unsqueeze(1)
        x = self.layer_norm(x) * (1 + scale) + shift
        x = self.linear(x)
        return x


class VideoRopePosition3DEmb(nn.Module):
    def __init__(self, head_dim, len_h, len_w, len_t, h_extrapolation_ratio=1.0, w_extrapolation_ratio=1.0,
                 t_extrapolation_ratio=1.0, enable_fps_modulation=True, base_fps=24, device=None):
        super().__init__()
        self.max_h = len_h
        self.max_w = len_w
        self.base_fps = base_fps
        self.enable_fps_modulation = enable_fps_modulation

        dim_h = head_dim // 6 * 2
        dim_w = dim_h
        dim_t = head_dim - 2 * dim_h

        self.register_buffer(
            "dim_spatial_range",
            torch.arange(0, dim_h, 2, device=device)[: (dim_h // 2)].float() / dim_h,
            persistent=False,
        )
        self.register_buffer(
            "dim_temporal_range",
            torch.arange(0, dim_t, 2, device=device)[: (dim_t // 2)].float() / dim_t,
            persistent=False,
        )

        self.h_ntk_factor = h_extrapolation_ratio ** (dim_h / (dim_h - 2))
        self.w_ntk_factor = w_extrapolation_ratio ** (dim_w / (dim_w - 2))
        self.t_ntk_factor = t_extrapolation_ratio ** (dim_t / (dim_t - 2))

    @torch.no_grad()
    def forward(self, x, fps=None, device=None, dtype=None):
        B, T, H, W, C = x.shape
        h_ntk_factor = self.h_ntk_factor
        w_ntk_factor = self.w_ntk_factor
        t_ntk_factor = self.t_ntk_factor

        h_theta = 10000.0 * h_ntk_factor
        w_theta = 10000.0 * w_ntk_factor
        t_theta = 10000.0 * t_ntk_factor

        h_spatial_freqs = 1.0 / (h_theta ** self.dim_spatial_range.to(device=device))
        w_spatial_freqs = 1.0 / (w_theta ** self.dim_spatial_range.to(device=device))
        temporal_freqs = 1.0 / (t_theta ** self.dim_temporal_range.to(device=device))

        seq = torch.arange(max(H, W, T), dtype=torch.float, device=device)
        half_emb_h = torch.outer(seq[:H].to(device=device), h_spatial_freqs)
        half_emb_w = torch.outer(seq[:W].to(device=device), w_spatial_freqs)

        if fps is None or not self.enable_fps_modulation:
            half_emb_t = torch.outer(seq[:T].to(device=device), temporal_freqs)
        else:
            half_emb_t = torch.outer(seq[:T].to(device=device) / fps * self.base_fps, temporal_freqs)

        half_emb_h = torch.stack([torch.cos(half_emb_h), -torch.sin(half_emb_h), torch.sin(half_emb_h), torch.cos(half_emb_h)], dim=-1)
        half_emb_w = torch.stack([torch.cos(half_emb_w), -torch.sin(half_emb_w), torch.sin(half_emb_w), torch.cos(half_emb_w)], dim=-1)
        half_emb_t = torch.stack([torch.cos(half_emb_t), -torch.sin(half_emb_t), torch.sin(half_emb_t), torch.cos(half_emb_t)], dim=-1)

        em_T_H_W_D = torch.cat(
            [
                repeat(half_emb_t, "t d x -> t h w d x", h=H, w=W),
                repeat(half_emb_h, "h d x -> t h w d x", t=T, w=W),
                repeat(half_emb_w, "w d x -> t h w d x", t=T, h=H),
            ],
            dim=-2,
        )

        return rearrange(em_T_H_W_D, "t h w d (i j) -> (t h w) d i j", i=2, j=2).float()


def pad_to_patch_size(x, patch_size):
    t, h, w = patch_size
    _, _, T, H, W = x.shape
    pad_t = (t - T % t) % t
    pad_h = (h - H % h) % h
    pad_w = (w - W % w) % w
    if pad_t or pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_t))
    return x


class AnimaModel(nn.Module):
    def __init__(self, device=None, dtype=None):
        super().__init__()
        model_channels = 2048
        num_blocks = 28
        num_heads = 16
        head_dim = model_channels // num_heads
        in_channels = 16
        out_channels = 16
        patch_spatial = 2
        patch_temporal = 1
        concat_padding_mask = True
        adaln_lora_dim = 256

        self.model_channels = model_channels
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal
        self.concat_padding_mask = concat_padding_mask
        self.num_heads = num_heads

        self.pos_embedder = VideoRopePosition3DEmb(
            head_dim=head_dim, len_h=128, len_w=128, len_t=1,
            device=device,
        )

        self.t_embedder = nn.Sequential(
            Timesteps(model_channels),
            TimestepEmbedding(model_channels, model_channels, device=device, dtype=dtype),
        )

        eff_in_channels = in_channels + 1 if concat_padding_mask else in_channels
        self.x_embedder = PatchEmbed(
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            in_channels=eff_in_channels,
            out_channels=model_channels,
            device=device, dtype=dtype,
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                x_dim=model_channels,
                context_dim=1024,
                num_heads=num_heads,
                head_dim=head_dim,
                adaln_lora_dim=adaln_lora_dim,
                device=device, dtype=dtype,
            )
            for _ in range(num_blocks)
        ])

        self.final_layer = FinalLayer(
            hidden_size=model_channels,
            patch_spatial=patch_spatial,
            patch_temporal=patch_temporal,
            out_channels=out_channels,
            adaln_lora_dim=adaln_lora_dim,
            device=device, dtype=dtype,
        )

        self.t_embedding_norm = RMSNorm(model_channels, eps=1e-6)

        self.llm_adapter = LLMAdapter(device=device, dtype=dtype)

    def prepare_embedded_sequence(self, x, padding_mask=None):
        if self.concat_padding_mask:
            if padding_mask is None:
                padding_mask = torch.zeros(
                    x.shape[0], 1, x.shape[3], x.shape[4],
                    dtype=x.dtype, device=x.device,
                )
            padding_mask = padding_mask.unsqueeze(1).repeat(1, 1, x.shape[2], 1, 1)
            x = torch.cat([x, padding_mask], dim=1)
        x = self.x_embedder(x)
        rope_emb = self.pos_embedder(x, device=x.device)
        return x, rope_emb

    def preprocess_text_embeds(self, text_embeds, text_ids, t5xxl_weights=None):
        if text_ids is not None:
            out = self.llm_adapter(text_embeds, text_ids)
            if t5xxl_weights is not None:
                out = out * t5xxl_weights.unsqueeze(-1)
            if out.shape[1] < 512:
                out = F.pad(out, (0, 0, 0, 512 - out.shape[1]))
            return out
        return text_embeds

    def forward(self, x, timesteps, context, t5xxl_ids=None, t5xxl_weights=None, padding_mask=None):
        orig_shape = x.shape
        if t5xxl_ids is not None:
            context = self.preprocess_text_embeds(context, t5xxl_ids, t5xxl_weights=t5xxl_weights)

        x = pad_to_patch_size(x, (self.patch_temporal, self.patch_spatial, self.patch_spatial))
        x, rope_emb = self.prepare_embedded_sequence(x, padding_mask=padding_mask)

        if timesteps.ndim == 1:
            timesteps = timesteps.unsqueeze(1)
        t_emb, adaln_lora = self.t_embedder[1](self.t_embedder[0](timesteps).to(dtype=x.dtype))
        t_emb = self.t_embedding_norm(t_emb)

        B, T, H, W, D = x.shape
        x = rearrange(x, "b t h w d -> b (t h w) d")
        emb = t_emb[:, 0]

        rope_cos = rope_emb[..., 0, 0]
        rope_sin = rope_emb[..., 1, 0]
        rope_cos = torch.cat([rope_cos, rope_cos], dim=-1).unsqueeze(0).expand(B, -1, -1)
        rope_sin = torch.cat([rope_sin, rope_sin], dim=-1).unsqueeze(0).expand(B, -1, -1)
        position_embeddings = (rope_cos, rope_sin)

        for block in self.blocks:
            x = block(x, emb, context, position_embeddings=position_embeddings, adaln_lora=adaln_lora[:, 0])

        x = rearrange(x, "b (t h w) d -> b t h w d", t=T, h=H, w=W)
        x = self.final_layer(x, t_emb, adaln_lora=adaln_lora)
        x = rearrange(
            x, "b t h w (p1 p2 t2 c) -> b c (t t2) (h p1) (w p2)",
            p1=self.patch_spatial, p2=self.patch_spatial, t2=self.patch_temporal,
        )
        x = x[:, :, :orig_shape[-3], :orig_shape[-2], :orig_shape[-1]]
        return x
