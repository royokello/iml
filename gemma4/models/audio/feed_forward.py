import torch
import torch.nn as nn
from transformers.activations import ACT2FN

from gemma4.models.audio.config import Gemma4AudioConfig
from gemma4.models.linear import Gemma4ClippableLinear
from gemma4.models.norm import Gemma4RMSNorm

class Gemma4AudioFeedForward(nn.Module):
    def __init__(self, config: Gemma4AudioConfig):
        super().__init__()
        self.config = config

        self.ffw_layer_1 = Gemma4ClippableLinear(config, config.hidden_size, config.hidden_size * 4)
        self.ffw_layer_2 = Gemma4ClippableLinear(config, config.hidden_size * 4, config.hidden_size)

        self.pre_layer_norm = Gemma4RMSNorm(config.hidden_size)
        self.post_layer_norm = Gemma4RMSNorm(config.hidden_size)
        self.act_fn = ACT2FN[config.hidden_act]

        self.gradient_clipping = config.gradient_clipping
        self.post_layer_scale = config.residual_weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # This is needed to avoid any underflow/overflow issues when clipping
        gradient_clipping = min(self.gradient_clipping, torch.finfo(self.ffw_layer_1.linear.weight.dtype).max)

        residual = hidden_states
        hidden_states = torch.clamp(hidden_states, -gradient_clipping, gradient_clipping)
        hidden_states = self.pre_layer_norm(hidden_states)

        hidden_states = self.ffw_layer_1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.ffw_layer_2(hidden_states)

        hidden_states = torch.clamp(hidden_states, -gradient_clipping, gradient_clipping)
        hidden_states = self.post_layer_norm(hidden_states)
        hidden_states *= self.post_layer_scale
        hidden_states += residual

        return hidden_states
