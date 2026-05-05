from transformers.configuration_utils import PretrainedConfig

def auto_docstring(*args, **kwargs):
    def decorator(obj):
        return obj
    return decorator


def strict(obj):
    return obj


@auto_docstring(checkpoint="google/gemma-4-e2b-it")
@strict
class Gemma4VisionConfig(PretrainedConfig):
    r"""
    pooling_kernel_size (`int`, *optional*):
        Spatial pooling kernel size applied after patchification.
    position_embedding_size (`int`, defaults to 10240):
        Maximum number of position embeddings for the vision encoder. Controls the size of
        the learned 2D position embedding table used by the patch embedder.
    use_clipped_linears (`bool`, defaults to `False`):
        Whether to use weight-clipped linear layers. When enabled, linear layer weights are
        clamped to a fixed range during the forward pass to improve numerical stability.
    standardize (`bool`, defaults to `False`):
        If true, applies a bias and scale to the soft tokens returned from the pooler.
    """

    model_type = "gemma4_vision"
    base_model_tp_plan = {
        "encoder.layers.*.self_attn.q_proj": "colwise",
        "encoder.layers.*.self_attn.k_proj": "colwise",
        "encoder.layers.*.self_attn.v_proj": "colwise",
        "encoder.layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
        "encoder.layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
        "encoder.layers.*.self_attn.o_proj": "rowwise",
        "encoder.layers.*.mlp.gate_proj": "colwise",
        "encoder.layers.*.mlp.up_proj": "colwise",
        "encoder.layers.*.mlp.down_proj": "rowwise",
    }
    default_theta = 100.0

    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 16
    num_attention_heads: int = 12
    num_key_value_heads: int = 12
    head_dim: int = 64
    hidden_activation: str = "gelu_pytorch_tanh"
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 131_072
    attention_bias: bool | None = False
    attention_dropout: float | None = 0.0
    rope_parameters: dict | None = None
    pooling_kernel_size: int = 3
    patch_size: int = 16
    position_embedding_size: int = 10 * 1024
    use_clipped_linears: bool = False
    standardize: bool = False
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        if self.rope_parameters is None:
            self.rope_parameters = {"rope_type": "default", "rope_theta": 100.0}

        super().__post_init__(**kwargs)
