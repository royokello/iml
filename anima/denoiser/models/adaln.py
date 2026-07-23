import torch
import torch.nn as nn


class AdaLN(nn.Sequential):
    def __init__(self, hidden_size, lora_dim=256, n_chunks=2, device=None, dtype=None):
        super().__init__(
            nn.SiLU(),
            nn.Linear(hidden_size, lora_dim, bias=False, device=device, dtype=dtype),
            nn.Linear(lora_dim, n_chunks * hidden_size, bias=False, device=device, dtype=dtype),
        )
        self.n_chunks = n_chunks

    def forward(self, emb, adaln_lora=None):
        mod = super().forward(emb)
        if adaln_lora is not None:
            mod = mod + adaln_lora[..., :self.n_chunks * emb.shape[-1]]
        return mod.chunk(self.n_chunks, dim=-1)
