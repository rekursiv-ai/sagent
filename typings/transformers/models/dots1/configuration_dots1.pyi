from ...configuration_utils import PretrainedConfig

logger = ...

class Dots1Config(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    base_model_tp_plan = ...
    base_model_pp_plan = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        intermediate_size=...,
        moe_intermediate_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        num_key_value_heads=...,
        n_shared_experts=...,
        n_routed_experts=...,
        n_group=...,
        topk_group=...,
        num_experts_per_tok=...,
        first_k_dense_replace=...,
        norm_topk_prob=...,
        hidden_act=...,
        max_position_embeddings=...,
        initializer_range=...,
        rms_norm_eps=...,
        use_cache=...,
        tie_word_embeddings=...,
        rope_theta=...,
        rope_scaling=...,
        attention_bias=...,
        attention_dropout=...,
        routed_scaling_factor=...,
        sliding_window=...,
        max_window_layers=...,
        layer_types=...,
        **kwargs,
    ) -> None: ...

__all__ = ["Dots1Config"]
