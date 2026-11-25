from __future__ import annotations

import torch
import torch.nn as nn

_BLOCK_SIZE = 32
_SUPPORTED_METHOD = "q8"


class QuantizedEmbedding(nn.Module):
    """An embedding table stored as int8 blocks with one fp16 scale per block.

    The forward pass gathers only the rows that `input_ids` selects. It never
    dequantizes the full table.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        *,
        method: str = _SUPPORTED_METHOD,
    ) -> None:
        super().__init__()
        if method != _SUPPORTED_METHOD:
            raise ValueError(f"QuantizedEmbedding supports {_SUPPORTED_METHOD!r} only, got {method!r}.")
        if embedding_dim % _BLOCK_SIZE != 0:
            raise ValueError(f"embedding_dim={embedding_dim} must be a multiple of {_BLOCK_SIZE}.")

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.method = method
        self.blocks_per_row = embedding_dim // _BLOCK_SIZE
        num_blocks = num_embeddings * self.blocks_per_row

        self.register_buffer("weight", torch.empty((num_blocks, _BLOCK_SIZE), dtype=torch.int8))
        self.register_buffer("sub_scales", torch.empty((num_blocks,), dtype=torch.float16))
        self.compute_dtype = torch.float16

    @classmethod
    def from_prequantized(cls, embedding: nn.Embedding, method: str) -> "QuantizedEmbedding":
        """Build an empty quantized table with the buffers of `embedding`.

        The caller loads the packed tensors afterwards.

        Args:
            embedding: The plain embedding whose shape and padding index are
                copied.
            method: The quantization method. Only `q8` is supported.

        Returns:
            The quantized embedding with uninitialized buffers.
        """
        module = cls(
            num_embeddings=int(embedding.num_embeddings),
            embedding_dim=int(embedding.embedding_dim),
            padding_idx=embedding.padding_idx,
            method=method,
        )
        module.compute_dtype = embedding.weight.dtype if embedding.weight.is_floating_point() else torch.float16
        return module

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        shape = input_ids.shape
        flat_ids = input_ids.reshape(-1)
        row_offsets = torch.arange(self.blocks_per_row, device=input_ids.device)
        block_index = flat_ids[:, None] * self.blocks_per_row + row_offsets
        block_index = block_index.reshape(-1)

        gathered = self.weight.index_select(0, block_index).view(-1, self.blocks_per_row, _BLOCK_SIZE)
        scales = self.sub_scales.index_select(0, block_index).view(-1, self.blocks_per_row, 1)
        output = gathered.to(scales.dtype) * scales
        output = output.reshape(*shape, self.embedding_dim).to(self.compute_dtype)

        if self.padding_idx is not None:
            output = output.masked_fill((input_ids == self.padding_idx).unsqueeze(-1), 0.0)
        return output


__all__ = ["QuantizedEmbedding"]
