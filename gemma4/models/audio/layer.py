import torch
import torch.nn as nn
from transformers.processing_utils import Unpack

from gemma4.models.audio.config import Gemma4AudioConfig
from gemma4.models.norm import Gemma4RMSNorm

class Gemma4AudioLayer(nn.Module):
    def __init__(self, config: Gemma4AudioConfig, layer_idx: int):
        super().__init__()
        self.config = config

        self.feed_forward1 = Gemma4AudioFeedForward(config)
        self.feed_forward2 = Gemma4AudioFeedForward(config)
        self.self_attn = Gemma4AudioAttention(config, layer_idx)
        self.lconv1d = Gemma4AudioLightConv1d(config)

        self.norm_pre_attn = Gemma4RMSNorm(config.hidden_size)
        self.norm_post_attn = Gemma4RMSNorm(config.hidden_size)
        self.norm_out = Gemma4RMSNorm(config.hidden_size)

        self.gradient_clipping = config.gradient_clipping

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.BoolTensor | None,
        position_embeddings: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        # This is needed to avoid any underflow/overflow issues when clipping
        gradient_clipping = min(self.gradient_clipping, torch.finfo(self.norm_pre_attn.weight.dtype).max)

        hidden_states = self.feed_forward1(hidden_states)
        residual = hidden_states

        hidden_states = torch.clamp(hidden_states, -gradient_clipping, gradient_clipping)
        hidden_states = self.norm_pre_attn(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )

        hidden_states = torch.clamp(hidden_states, -gradient_clipping, gradient_clipping)
        hidden_states = self.norm_post_attn(hidden_states)
        hidden_states += residual

        hidden_states = self.lconv1d(hidden_states)
        hidden_states = self.feed_forward2(hidden_states)

        hidden_states = torch.clamp(hidden_states, -gradient_clipping, gradient_clipping)
        hidden_states = self.norm_out(hidden_states)

        return hidden_states
