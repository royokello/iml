import torch
import torch.nn as nn

from gemma4.models.norm import Gemma4RMSNorm
from gemma4.models.text.config import Gemma4TextConfig

class Gemma4TextRouter(nn.Module):
    def __init__(self, config: Gemma4TextConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.scalar_root_size = self.hidden_size**-0.5
        self.eps = config.rms_norm_eps

        self.norm = Gemma4RMSNorm(self.hidden_size, eps=self.eps, with_scale=False)
        self.proj = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.scale = nn.Parameter(torch.ones(self.hidden_size))
        self.per_expert_scale = nn.Parameter(torch.ones(config.num_experts))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states * self.scale * self.scalar_root_size

        expert_scores = self.proj(hidden_states)  # [B*S, E]
        router_probabilities = nn.functional.softmax(expert_scores, dim=-1)

        # topk returns both values (probabilities) and indices directly
        top_k_weights, top_k_index = torch.topk(
            router_probabilities,
            k=self.config.top_k_experts,
            dim=-1,
        )  # both [B*S, K]

        # Normalize the top-k weights so they sum to 1 per token
        top_k_weights /= top_k_weights.sum(dim=-1, keepdim=True)

        # Apply per-expert scale directly to the weights
        top_k_weights = top_k_weights * self.per_expert_scale[top_k_index]

        return router_probabilities, top_k_weights, top_k_index