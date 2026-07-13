from ...configuration_utils import PretrainedConfig

"""openai model configuration"""

class GptOssConfig(PretrainedConfig):
    model_type = ...
    base_model_pp_plan = ...
    base_model_tp_plan = ...
    def __init__(
        self,
        num_hidden_layers: int = ...,
        num_local_experts: int = ...,
        vocab_size: int = ...,
        hidden_size: int = ...,
        intermediate_size: int = ...,
        head_dim: int = ...,
        num_attention_heads: int = ...,
        num_key_value_heads: int = ...,
        sliding_window: int = ...,
        rope_theta: float = ...,
        tie_word_embeddings=...,
        hidden_act: str = ...,
        initializer_range: float = ...,
        max_position_embeddings=...,
        rms_norm_eps: float = ...,
        rope_scaling=...,
        attention_dropout: float = ...,
        num_experts_per_tok=...,
        router_aux_loss_coef: float = ...,
        output_router_logits=...,
        use_cache=...,
        layer_types=...,
        **kwargs,
    ) -> None: ...

__all__ = ["GptOssConfig"]
