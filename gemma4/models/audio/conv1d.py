import torch
import torch.nn as nn
from transformers.activations import ACT2FN

from gemma4.models.audio.config import Gemma4AudioConfig
from gemma4.models.linear import Gemma4ClippableLinear
from gemma4.models.norm import Gemma4RMSNorm

# TODO: this could be imported from Voxtral realtime
class Gemma4AudioCausalConv1d(nn.Conv1d):
    # def __init__(
    #     self,
    #     in_channels: int,
    #     out_channels: int,
    #     kernel_size: int,
    #     # cache_key: str,
    #     stride: int = 1,
    #     dilation: int = 1,
    #     bias: bool = True,
    # ):
    #     super().__init__(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, bias=bias)
    # self.cache_key = cache_key

    @cached_property
    def left_pad(self):
        effective_kernel_size = (self.kernel_size[0] - 1) * self.dilation[0] + 1
        return effective_kernel_size - self.stride[0]

    def forward(
        self,
        x: torch.Tensor,
        # padding_cache: VoxtralRealtimeConv1dPaddingCache | None = None,  # TODO: we might want to add a cache?
    ) -> torch.Tensor:
        # if padding_cache is not None:
        #     x = padding_cache.update(x, self.cache_key, self)
        # else:
        #     x = nn.functional.pad(x, (self.left_pad, 0))
        x = nn.functional.pad(x, (self.left_pad, 0))

        return super().forward(x)


class Gemma4AudioLightConv1d(nn.Module):
    def __init__(self, config: Gemma4AudioConfig):
        super().__init__()
        self.config = config

        self.linear_start = Gemma4ClippableLinear(config, config.hidden_size, config.hidden_size * 2)
        self.linear_end = Gemma4ClippableLinear(config, config.hidden_size, config.hidden_size)
        self.depthwise_conv1d = Gemma4AudioCausalConv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=config.conv_kernel_size,
            groups=config.hidden_size,
            bias=False,
        )

        self.pre_layer_norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, with_scale=True)
        self.conv_norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, with_scale=True)
        self.act_fn = ACT2FN[config.hidden_act]

        self.gradient_clipping = config.gradient_clipping

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.pre_layer_norm(hidden_states)
        hidden_states = self.linear_start(hidden_states)
        hidden_states = nn.functional.glu(hidden_states, dim=-1)

        hidden_states = self.depthwise_conv1d(hidden_states.transpose(1, 2)).transpose(1, 2)

        # This is needed to avoid any underflow/overflow issues when clipping
        gradient_clipping = min(self.gradient_clipping, torch.finfo(self.linear_start.linear.weight.dtype).max)
        hidden_states = torch.clamp(hidden_states, -gradient_clipping, gradient_clipping)
        hidden_states = self.conv_norm(hidden_states)

        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.linear_end(hidden_states)
        hidden_states += residual
        return hidden_states
