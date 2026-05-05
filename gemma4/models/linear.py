import torch
import torch.nn as nn

from gemma4.models.audio.config import Gemma4AudioConfig
from gemma4.models.vision.config import Gemma4VisionConfig

class Gemma4ClippableLinear(nn.Module):
    def __init__(
        self,
        config: Gemma4VisionConfig | Gemma4AudioConfig,
        in_features: int,
        out_features: int,
    ) -> None:
        super().__init__()
        self.use_clipped_linears = config.use_clipped_linears
        self.linear = nn.Linear(in_features, out_features, bias=False)

        if self.use_clipped_linears:
            self.register_buffer("input_min", torch.tensor(-float("inf")))
            self.register_buffer("input_max", torch.tensor(float("inf")))
            self.register_buffer("output_min", torch.tensor(-float("inf")))
            self.register_buffer("output_max", torch.tensor(float("inf")))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_clipped_linears:
            hidden_states = torch.clamp(hidden_states, self.input_min, self.input_max)

        hidden_states = self.linear(hidden_states)

        if self.use_clipped_linears:
            hidden_states = torch.clamp(hidden_states, self.output_min, self.output_max)

        return hidden_states