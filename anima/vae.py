from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

CACHE_T = 2


class QwenImageCausalConv3d(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self._padding = (
            self.padding[2], self.padding[2],
            self.padding[1], self.padding[1],
            2 * self.padding[0], 0,
        )
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        return super().forward(x)


class QwenImageRMS_norm(nn.Module):
    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1) if images else (1, 1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)
        self.channel_first = channel_first
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        needs_fp32 = x.dtype in (torch.float16, torch.bfloat16) or any(
            t in str(x.dtype) for t in ("float4_", "float8_")
        )
        inp = x.float() if needs_fp32 else x
        normalized = F.normalize(inp, dim=(1 if self.channel_first else -1)).to(x.dtype)
        return normalized * self.scale * self.gamma + self.bias


class QwenImageUpsample(nn.Upsample):
    def forward(self, x):
        return super().forward(x.float()).type_as(x)


class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.residual = nn.Sequential(
            QwenImageRMS_norm(in_dim, images=False),              # 0
            nn.SiLU(),                                            # 1
            QwenImageCausalConv3d(in_dim, out_dim, 3, padding=1), # 2
            QwenImageRMS_norm(out_dim, images=False),             # 3
            nn.SiLU(),                                            # 4
            nn.Dropout(dropout),                                  # 5
            QwenImageCausalConv3d(out_dim, out_dim, 3, padding=1),# 6
        )
        if in_dim != out_dim:
            self.shortcut = QwenImageCausalConv3d(in_dim, out_dim, 1)

    def forward(self, x, feat_cache=None, feat_idx=None):
        h = self.shortcut(x) if hasattr(self, 'shortcut') else x
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -min(CACHE_T, x.shape[2]):, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.residual[2](self.residual[1](self.residual[0](x)), feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
            idx = feat_idx[0]
            cache_x = x[:, :, -min(CACHE_T, x.shape[2]):, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.residual[6](self.residual[5](self.residual[4](self.residual[3](x))), feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.residual[2](self.residual[1](self.residual[0](x)))
            x = self.residual[6](self.residual[5](self.residual[4](self.residual[3](x))))
        return x + h


class AttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm = QwenImageRMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x, feat_cache=None, feat_idx=None):
        identity = x
        b, c, t, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.norm(x)
        qkv = self.to_qkv(x)
        qkv = qkv.reshape(b * t, 1, c * 3, -1)
        qkv = qkv.permute(0, 1, 3, 2).contiguous()
        q, k, v = qkv.chunk(3, dim=-1)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)
        x = self.proj(x)
        x = x.view(b, t, c, h, w).permute(0, 2, 1, 3, 4)
        return x + identity


class ResampleBlock(nn.Module):
    def __init__(self, dim, mode):
        super().__init__()
        self.dim = dim
        self.mode = mode
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                QwenImageUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim // 2, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                QwenImageUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim // 2, 3, padding=1),
            )
            self.time_conv = QwenImageCausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample2d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)),
            )
        elif mode == "downsample3d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)),
            )
            self.time_conv = QwenImageCausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

    def forward(self, x, feat_cache=None, feat_idx=None):
        b, c, t, h, w = x.size()
        if self.mode == "upsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = "Rep"
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -min(CACHE_T, x.shape[2]):, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] != "Rep":
                        cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] == "Rep":
                        cache_x = torch.cat([torch.zeros_like(cache_x).to(cache_x.device), cache_x], dim=2)
                    if feat_cache[idx] == "Rep":
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.resample(x)
        x = x.view(b, t, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)
        if self.mode == "downsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x


class QwenImageEncoder3d(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = QwenImageCausalConv3d(3, 96, 3, padding=1)
        self.downsamples = nn.ModuleList([
            ResBlock(96, 96),                           # 0
            ResBlock(96, 96),                           # 1
            ResampleBlock(96, "downsample2d"),          # 2
            ResBlock(96, 192),                          # 3
            ResBlock(192, 192),                         # 4
            ResampleBlock(192, "downsample3d"),         # 5
            ResBlock(192, 384),                         # 6
            ResBlock(384, 384),                         # 7
            ResampleBlock(384, "downsample3d"),         # 8
            ResBlock(384, 384),                         # 9
            ResBlock(384, 384),                         # 10
        ])
        self.middle = nn.ModuleList([
            ResBlock(384, 384),                         # 0
            AttentionBlock(384),                        # 1
            ResBlock(384, 384),                         # 2
        ])
        self.head = nn.Sequential(
            QwenImageRMS_norm(384, images=False),       # 0
            nn.SiLU(),                                  # 1
            QwenImageCausalConv3d(384, 32, 3, padding=1), # 2
        )

    def forward(self, x, feat_cache=None, feat_idx=None):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -min(CACHE_T, x.shape[2]):, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)
        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)
        for layer in self.middle:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)
        for layer in self.head:
            x = layer(x)
        return x


class QwenImageDecoder3d(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = QwenImageCausalConv3d(16, 384, 3, padding=1)
        self.middle = nn.ModuleList([
            ResBlock(384, 384),                         # 0
            AttentionBlock(384),                        # 1
            ResBlock(384, 384),                         # 2
        ])
        self.upsamples = nn.ModuleList([
            ResBlock(384, 384),                         # 0
            ResBlock(384, 384),                         # 1
            ResBlock(384, 384),                         # 2
            ResampleBlock(384, "upsample3d"),           # 3
            ResBlock(192, 384),                         # 4
            ResBlock(384, 384),                         # 5
            ResBlock(384, 384),                         # 6
            ResampleBlock(384, "upsample3d"),           # 7
            ResBlock(192, 192),                         # 8
            ResBlock(192, 192),                         # 9
            ResBlock(192, 192),                         # 10
            ResampleBlock(192, "upsample2d"),           # 11
            ResBlock(96, 96),                           # 12
            ResBlock(96, 96),                           # 13
            ResBlock(96, 96),                           # 14
        ])
        self.head = nn.Sequential(
            QwenImageRMS_norm(96, images=False),        # 0
            nn.SiLU(),                                  # 1
            QwenImageCausalConv3d(96, 3, 3, padding=1),  # 2
        )

    def forward(self, x, feat_cache=None, feat_idx=None):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -min(CACHE_T, x.shape[2]):, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)
        for layer in self.middle:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)
        for layer in self.upsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)
        for layer in self.head:
            x = layer(x)
        return x


class QwenImageVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = QwenImageCausalConv3d(32, 32, 1)  # quant_conv
        self.conv2 = QwenImageCausalConv3d(16, 16, 1)  # post_quant_conv
        self.encoder = QwenImageEncoder3d()
        self.decoder = QwenImageDecoder3d()

    def clear_cache(self):
        def _count_conv3d(model):
            return sum(1 for m in model.modules() if isinstance(m, QwenImageCausalConv3d))
        self._conv_num = _count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        self._enc_conv_num = _count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num

    def encode(self, x):
        _, _, t, _, _ = x.shape
        self.clear_cache()
        iter_ = 1 + (t - 1) // 4
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out = self.encoder(x[:, :, :1, :, :], self._enc_feat_map, self._enc_conv_idx)
            else:
                out_ = self.encoder(
                    x[:, :, 1 + 4 * (i - 1): 1 + 4 * i, :, :],
                    self._enc_feat_map, self._enc_conv_idx,
                )
                out = torch.cat([out, out_], 2)
        enc = self.conv1(out)
        self.clear_cache()
        return enc

    def decode(self, z):
        _, _, t, _, _ = z.shape
        self.clear_cache()
        x = self.conv2(z)
        for i in range(t):
            self._conv_idx = [0]
            frame = x[:, :, i:i + 1, :, :]
            if i == 0:
                out = self.decoder(frame, self._feat_map, self._conv_idx)
            else:
                out_ = self.decoder(frame, self._feat_map, self._conv_idx)
                out = torch.cat([out, out_], 2)
        out = torch.clamp(out, min=-1.0, max=1.0)
        self.clear_cache()
        return out

    def forward(self, x):
        return self.decode(x)


def _load_vae_checkpoint(model, path):
    from safetensors.torch import load_file
    state = load_file(str(path))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        print(f"    VAE unexpected keys: {len(unexpected)}")
    if missing:
        print(f"    VAE missing keys: {len(missing)}")
    assert not missing, f"Missing keys: {missing}"
