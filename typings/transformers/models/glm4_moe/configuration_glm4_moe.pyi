from ...configuration_utils import PretrainedConfig

class Glm4MoeConfig(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    base_model_tp_plan = ...
    base_model_pp_plan = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        intermediate_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        partial_rotary_factor=...,
        num_key_value_heads=...,
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
        moe_intermediate_size=...,
        num_experts_per_tok=...,
        n_shared_experts=...,
        n_routed_experts=...,
        routed_scaling_factor=...,
        n_group=...,
        topk_group=...,
        first_k_dense_replace=...,
        norm_topk_prob=...,
        use_qk_norm=...,
        **kwargs,
    ) -> None: ...

__all__ = ["Glm4MoeConfig"]
