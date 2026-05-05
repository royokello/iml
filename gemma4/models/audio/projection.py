import torch
import torch.nn as nn

from gemma4.models.audio.config import Gemma4AudioConfig

class Gemma4AudioSubSampleConvProjectionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, norm_eps):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=1,
            bias=False,
        )
        self.norm = nn.LayerNorm(out_channels, eps=norm_eps, elementwise_affine=True, bias=False)
        self.act = nn.ReLU()

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor | None = None):
        if mask is not None:
            mask = mask.to(device=hidden_states.device)
            hidden_states = hidden_states * mask[:, None, :, None]

        hidden_states = self.conv(hidden_states.to(self.conv.weight.dtype))
        hidden_states = self.act(self.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous())

        if mask is not None:
            mask = mask[:, ::2]

        return hidden_states, mask
    

class Gemma4AudioSubSampleConvProjection(nn.Module):
    def __init__(self, config: Gemma4AudioConfig):
        super().__init__()
        self.layer0 = Gemma4AudioSubSampleConvProjectionLayer(
            in_channels=1,
            out_channels=config.subsampling_conv_channels[0],
            norm_eps=config.rms_norm_eps,
        )
        self.layer1 = Gemma4AudioSubSampleConvProjectionLayer(
            in_channels=config.subsampling_conv_channels[0],
            out_channels=config.subsampling_conv_channels[1],
            norm_eps=config.rms_norm_eps,
        )
        proj_input_dim = (config.subsampling_conv_channels[0] // 4) * config.subsampling_conv_channels[1]
        self.input_proj_linear = nn.Linear(proj_input_dim, config.hidden_size, bias=False)

    def forward(
        self,
        input_features: torch.Tensor,
        input_features_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = input_features.unsqueeze(1)
        hidden_states, mask = self.layer0(hidden_states, input_features_mask)
        hidden_states, mask = self.layer1(hidden_states, mask)

        batch_size, _, seq_len, _ = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 3, 1).contiguous().reshape(batch_size, seq_len, -1)
        return self.input_proj_linear(hidden_states), mask
