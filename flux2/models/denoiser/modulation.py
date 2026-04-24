import torch
import torch.nn as nn


class Flux2Modulation(nn.Module):
    def __init__(self, dim: int, mod_param_sets: int = 2, bias: bool = False):
        super().__init__()
        self.mod_param_sets = mod_param_sets

        self.linear = nn.Linear(dim, dim * 3 * self.mod_param_sets, bias=bias)
        self.act_fn = nn.SiLU()

    def forward(self, temb: torch.Tensor) -> torch.Tensor:
        mod = self.act_fn(temb)
        mod = self.linear(mod)
        return mod

    @staticmethod
    # split inside the transformer blocks, to avoid passing tuples into checkpoints https://github.com/huggingface/diffusers/issues/12776
    def split(mod: torch.Tensor, mod_param_sets: int) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]:
        if mod.ndim == 2:
            mod = mod.unsqueeze(1)
        mod_params = torch.chunk(mod, 3 * mod_param_sets, dim=-1)
        # Return tuple of 3-tuples of modulation params shift/scale/gate
        return tuple(mod_params[3 * i : 3 * (i + 1)] for i in range(mod_param_sets))

