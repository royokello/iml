import torch
import torch.nn as nn
import torch.nn.functional as F
from gemma4.models.text.scaled_word_embedding import Gemma4TextScaledWordEmbedding
from utils.quant.cuda.affine_high import dequantize_from_affine_high as dequantize_from_affine_high_cuda
from utils.quant.cuda.affine_low import dequantize_from_affine_low as dequantize_from_affine_low_cuda
from utils.quant.cuda.affine_med import dequantize_from_affine_med as dequantize_from_affine_med_cuda
from utils.quant.cuda.symmetric_high import dequantize_from_symmetric_high as dequantize_from_symmetric_high_cuda
from utils.quant.cuda.symmetric_low import dequantize_from_symmetric_low as dequantize_from_symmetric_low_cuda
from utils.quant.cuda.symmetric_med import dequantize_from_symmetric_med as dequantize_from_symmetric_med_cuda
from utils.quant.linear import (
    SYMMETRIC_LOW_PACKED_WORDS_PER_SUB_BLOCK,
    SYMMETRIC_LOW_SUB_SCALE_BITS,
    SYMMETRIC_MED_PACKED_WORDS_PER_SUB_BLOCK,
)
from utils.quant.to.affine import AFFINE_MODES, quantize_to_affine
from utils.quant.to.symmetric import (
    HIGH_BLOCK_SIZE as SYMMETRIC_HIGH_BLOCK_SIZE,
    SUB_BLOCK_SIZE as SYMMETRIC_SUB_BLOCK_SIZE,
    SUPER_BLOCK_SIZE as SYMMETRIC_SUPER_BLOCK_SIZE,
    quantize_to_symmetric,
)
from utils.quant.validators import normalize_quant_method, quant_method_family, quant_method_mode


class QuantizedTextScaledWordEmbedding(nn.Module):
    def __init__(
        self,
        embedding: "Gemma4TextScaledWordEmbedding",
        method: str,
    ) -> None:
        super().__init__()
        self.num_embeddings = embedding.num_embeddings
        self.embedding_dim = embedding.embedding_dim
        self.padding_idx = embedding.padding_idx
        self.method = normalize_quant_method(method)
        
        embed_scale = getattr(embedding, "scalar_embed_scale", 1.0)
        if not isinstance(embed_scale, torch.Tensor):
            embed_scale = torch.tensor(embed_scale)
        self.register_buffer("embed_scale", embed_scale.clone(), persistent=False)
        
        if quant_method_family(self.method) == "symmetric":
            qweight, sub_scales, super_scales = quantize_to_symmetric(
                embedding.weight.detach(),
                mode=quant_method_mode(self.method),
            )
            self.register_buffer("weight", qweight)
            self.register_buffer("sub_scales", sub_scales)
            self.register_buffer("super_scales", super_scales)
            self.sub_mins = None
            self.super_mins = None
        else:
            qweight, sub_scales, sub_mins, super_scales, super_mins = quantize_to_affine(
                embedding.weight.detach(),
                mode=quant_method_mode(self.method),
            )
            self.register_buffer("weight", qweight)
            self.register_buffer("sub_scales", sub_scales)
            self.register_buffer("sub_mins", sub_mins)
            self.register_buffer("super_scales", super_scales)
            self.register_buffer("super_mins", super_mins)

    @classmethod
    def from_prequantized(
        cls, 
        embedding: "Gemma4TextScaledWordEmbedding", 
        method: str
    ) -> "QuantizedTextScaledWordEmbedding":
        module = cls.__new__(cls)
        nn.Module.__init__(module)
        module.num_embeddings = embedding.num_embeddings
        module.embedding_dim = embedding.embedding_dim
        module.padding_idx = embedding.padding_idx
        module.method = normalize_quant_method(method)
        
        embed_scale = getattr(embedding, "scalar_embed_scale", 1.0)
        if not isinstance(embed_scale, torch.Tensor):
            embed_scale = torch.tensor(embed_scale)
        module.register_buffer("embed_scale", embed_scale.clone(), persistent=False)
        
        # Use the same quantized shape logic as for linear layers
        # Embedding weight shape is (num_embeddings, embedding_dim)
        # which is analogous to (out_features, in_features) in linear layers
        from utils.quant.linear import _affine_quantized_2d_shape, _packed_words_for_values, _quantized_2d_shape
        
        if quant_method_family(module.method) == "symmetric":
            mode = quant_method_mode(module.method)
            if mode == "high":
                num_embeddings, embedding_dim = embedding.weight.shape
                blocks_per_row = (embedding_dim + SYMMETRIC_HIGH_BLOCK_SIZE - 1) // SYMMETRIC_HIGH_BLOCK_SIZE
                num_blocks = num_embeddings * blocks_per_row
                qweight_shape = (num_blocks, SYMMETRIC_HIGH_BLOCK_SIZE)
                qweight_dtype = torch.int8
                sub_scales_shape = (num_blocks,)
                sub_scales_dtype = torch.float16
                super_scales_shape = None
            else:
                num_super_blocks, super_block_size = _quantized_2d_shape(
                    embedding.weight.shape,
                    super_block_size=SYMMETRIC_SUPER_BLOCK_SIZE,
                    sub_block_size=SYMMETRIC_SUB_BLOCK_SIZE,
                )
                sub_blocks_per_super = super_block_size // SYMMETRIC_SUB_BLOCK_SIZE
                if mode == "med":
                    qweight_shape = (
                        num_super_blocks,
                        sub_blocks_per_super * SYMMETRIC_MED_PACKED_WORDS_PER_SUB_BLOCK,
                    )
                    qweight_dtype = torch.int32
                    sub_scales_shape = (num_super_blocks, sub_blocks_per_super)
                    sub_scales_dtype = torch.int8
                else:
                    qweight_shape = (
                        num_super_blocks,
                        sub_blocks_per_super * SYMMETRIC_LOW_PACKED_WORDS_PER_SUB_BLOCK,
                    )
                    qweight_dtype = torch.int32
                    sub_scale_words = (sub_blocks_per_super * SYMMETRIC_LOW_SUB_SCALE_BITS + 31) // 32
                    sub_scales_shape = (num_super_blocks, sub_scale_words)
                    sub_scales_dtype = torch.int32
                super_scales_shape = (num_super_blocks,)
            
            qweight = torch.empty(qweight_shape, device=embedding.weight.device, dtype=qweight_dtype)
            sub_scales = torch.empty(
                sub_scales_shape,
                device=embedding.weight.device,
                dtype=sub_scales_dtype,
            )
            super_scales = (
                None
                if super_scales_shape is None
                else torch.empty(super_scales_shape, device=embedding.weight.device, dtype=torch.float16)
            )
            module.register_buffer("weight", qweight)
            module.register_buffer("sub_scales", sub_scales)
            module.register_buffer("super_scales", super_scales)
            module.sub_mins = None
            module.super_mins = None
        else:
            mode = quant_method_mode(module.method)
            num_super_blocks, super_block_size, sub_block_size = _affine_quantized_2d_shape(
                embedding.weight.shape,
                mode=mode,
            )
            sub_blocks_per_super = super_block_size // sub_block_size
            weight_bits = int(AFFINE_MODES[mode]["weight_bits"])
            qweight_shape = (
                num_super_blocks,
                sub_blocks_per_super * _packed_words_for_values(sub_block_size, weight_bits),
            )
            qweight_dtype = torch.int32
            qweight = torch.empty(qweight_shape, device=embedding.weight.device, dtype=qweight_dtype)
            meta_bits = int(AFFINE_MODES[mode]["meta_bits"])
            sub_metadata_words = _packed_words_for_values(sub_blocks_per_super, meta_bits)
            sub_scales = torch.empty(
                (num_super_blocks, sub_metadata_words),
                device=embedding.weight.device,
                dtype=torch.int32,
            )
            sub_mins = torch.empty_like(sub_scales)
            super_scales = torch.empty((num_super_blocks,), device=embedding.weight.device, dtype=torch.float16)
            super_mins = torch.empty_like(super_scales)
            module.register_buffer("weight", qweight)
            module.register_buffer("sub_scales", sub_scales)
            module.register_buffer("sub_mins", sub_mins)
            module.register_buffer("super_scales", super_scales)
            module.register_buffer("super_mins", super_mins)
        
        return module

    def dequantize_weight(self) -> torch.Tensor:
        if quant_method_family(self.method) == "symmetric":
            mode = quant_method_mode(self.method)
            if mode == "high":
                dequantize = dequantize_from_symmetric_high_cuda
            elif mode == "med":
                dequantize = dequantize_from_symmetric_med_cuda
            else:
                dequantize = dequantize_from_symmetric_low_cuda
            if mode == "high":
                weight = dequantize(
                    self.weight,
                    self.sub_scales,
                    original_shape=(self.num_embeddings, self.embedding_dim),
                )
            else:
                weight = dequantize(
                    self.weight,
                    self.sub_scales,
                    self.super_scales,
                    original_shape=(self.num_embeddings, self.embedding_dim),
                )
        else:
            mode = quant_method_mode(self.method)
            if mode == "high":
                dequantize = dequantize_from_affine_high_cuda
            elif mode == "med":
                dequantize = dequantize_from_affine_med_cuda
            else:
                dequantize = dequantize_from_affine_low_cuda
            weight = dequantize(
                self.weight,
                self.sub_scales,
                self.sub_mins,
                self.super_scales,
                self.super_mins,
                original_shape=(self.num_embeddings, self.embedding_dim),
            )

        return weight.view(self.num_embeddings, self.embedding_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        weight = self.dequantize_weight()

        # Perform embedding lookup
        embedded = F.embedding(input_ids, weight, self.padding_idx)

        # Apply the scalar embedding scale
        return embedded * self.embed_scale.to(embedded.dtype)
