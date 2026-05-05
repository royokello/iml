from transformers.configuration_utils import PretrainedConfig
from typing import Any, Literal


# @auto_docstring(checkpoint="google/gemma-4-e2b-it")
# @strict
class Gemma4TextConfig(PretrainedConfig):
    r"""
    use_bidirectional_attention (`str`, *optional*):
        Controls bidirectional attention behavior. When set to `"vision"`, vision tokens
        attend bidirectionally while text tokens use causal attention. When set to `"all"`,
        all tokens use bidirectional attention.
    vocab_size_per_layer_input (`int`, defaults to 262144):
        Vocabulary size for the per-layer input embeddings (PLE). Used by models with
        per-layer residual streams where a smaller embedding is added at each decoder layer.
    hidden_size_per_layer_input (`int`, defaults to 256):
        Per-layer hidden dimension for the PLE system. The actual embedding weight has shape
        `[vocab_size_per_layer_input, num_hidden_layers * hidden_size_per_layer_input]`
        because all layers are packed into a single table. See the [Gemma4](https://huggingface.co/docs/transformers/main/en/model_doc/gemma4#per-layer-embeddings-ple) docs
        for a description of the full PLE pipeline.
    num_global_key_value_heads (`int`, *optional*):
        Number of key-value heads for global (full) attention layers. If `None`, defaults
        to `num_key_value_heads`.
    global_head_dim (`int`, defaults to 512):
        Dimension of each attention head in global (full) attention layers.
    attention_k_eq_v (`bool`, defaults to `False`):
        Whether keys and values share the same projection weights. When `True`, the key
        projection output is reused as the value projection.
    num_kv_shared_layers (`int`, defaults to 0):
        Number of consecutive decoder layers that share the same key-value projections.
        A value of 0 means no sharing (each layer has independent KV projections).
    enable_moe_block (`bool`, defaults to `False`):
        Whether to enable Mixture-of-Experts (MoE) blocks in the decoder layers. When
        `True`, eligible layers will use a sparse MoE feed-forward network.
    use_double_wide_mlp (`bool`, defaults to `False`):
        Whether to use a double-width MLP with fused gate and up projections.
    top_k_experts (`int`, *optional*):
        Number of experts activated per token in MoE layers. Only used when
        `enable_moe_block=True`.
    moe_intermediate_size (`int`, *optional*):
        Intermediate (hidden) size of each expert's feed-forward network in MoE layers.
        Only used when `enable_moe_block=True`.
    """

    model_type = "gemma4_text"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
        "layers.*.experts.gate_up_proj": "packed_colwise",
        "layers.*.experts.down_proj": "rowwise",
        "layers.*.experts": "moe_tp_experts",
    }
    base_model_ep_plan = {
        # EP plan for google/gemma-4-26B-A4B-it: do not tp in attention (num_global_key_value_heads=2 too small to partition)
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
        "layers.*.router": "ep_router",
        "layers.*.experts.gate_up_proj": "grouped_gemm",
        "layers.*.experts.down_proj": "grouped_gemm",
        "layers.*.experts": "moe_tp_experts",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    vocab_size: int = 262_144
    hidden_size: int = 2304
    intermediate_size: int = 9216
    num_hidden_layers: int = 30
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: int = 256
    hidden_activation: str = "gelu_pytorch_tanh"
    max_position_embeddings: int = 131_072
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 1
    bos_token_id: int | None = 2
    tie_word_embeddings: bool = True
    rope_parameters: dict | None = None
    attention_bias: bool = False
    attention_dropout: int | float | None = 0.0
    sliding_window: int = 512
    layer_types: list[str] | None = None
    final_logit_softcapping: float | None = None
    use_bidirectional_attention: Literal["all", "vision"] | None = None
    vocab_size_per_layer_input: int = 262_144
    hidden_size_per_layer_input: int = 256
    num_global_key_value_heads: int | None = None
    global_head_dim: int = 512
    attention_k_eq_v: bool = False
    num_kv_shared_layers: int = 0
    enable_moe_block: bool = False
    use_double_wide_mlp: bool = False
    num_experts: int | None = None
    top_k_experts: int | None = None
    moe_intermediate_size: int | None = None

    def __post_init__(self, **kwargs):
        if self.use_bidirectional_attention == "all":
            self.sliding_window = (self.sliding_window // 2) + 1  # due to fa we set exclusive bounds

        if self.layer_types is None:
            sliding_window_pattern = 6  # by default 5:1
            self.layer_types = [
                "sliding_attention" if bool((i + 1) % sliding_window_pattern) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

        if self.layer_types and (last_layer_type := self.layer_types[-1]) != "full_attention":
            # logger.warning(
            #     f"Last layer must use `full_attention`, but got `{last_layer_type}`. Forcing last layer to `full_attention`."
            # )
            self.layer_types[-1] = "full_attention"

        default_rope_params: dict[Literal["full_attention", "sliding_attention"] : dict[str, Any]] = {
            "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
            "full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1_000_000.0},
        }
        if self.rope_parameters is None:
            self.rope_parameters = default_rope_params

        super().__post_init__(**kwargs)

    def convert_rope_params_to_dict(self, **kwargs):
        # No need to handle BC for new models, because they have no old-format `rope_scaling`
        return kwargs
