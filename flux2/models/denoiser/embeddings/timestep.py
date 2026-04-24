import torch
import torch.nn as nn

from diffusers.models.embeddings import TimestepEmbedding, Timesteps


class Flux2TimestepGuidanceEmbeddings(nn.Module):
    def __init__(
        self,
        in_channels: int = 256,
        embedding_dim: int = 6144,
        bias: bool = False,
        guidance_embeds: bool = True,
    ):
        super().__init__()

        self.time_proj = Timesteps(num_channels=in_channels, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(
            in_channels=in_channels, time_embed_dim=embedding_dim, sample_proj_bias=bias
        )

        if guidance_embeds:
            self.guidance_embedder = TimestepEmbedding(
                in_channels=in_channels, time_embed_dim=embedding_dim, sample_proj_bias=bias
            )
        else:
            self.guidance_embedder = None

    def forward(self, timestep: torch.Tensor, guidance: torch.Tensor) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(timestep.dtype))  # (N, D)

        if guidance is not None and self.guidance_embedder is not None:
            guidance_proj = self.time_proj(guidance)
            guidance_emb = self.guidance_embedder(guidance_proj.to(guidance.dtype))  # (N, D)
            time_guidance_emb = timesteps_emb + guidance_emb
            return time_guidance_emb
        else:
            return timesteps_emb
