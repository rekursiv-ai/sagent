from ...configuration_utils import PretrainedConfig

"""Ernie 4.5 MoE model configuration"""
logger = ...

class Ernie4_5_MoeConfig(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    attribute_map = ...
    base_model_tp_plan = ...
    base_model_pp_plan = ...
    def __init__(
        self,
        vocab_size=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        hidden_size=...,
        intermediate_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        num_key_value_heads=...,
        hidden_act=...,
        max_position_embeddings=...,
        initializer_range=...,
        rms_norm_eps=...,
        use_cache=...,
        tie_word_embeddings=...,
        rope_theta=...,
        rope_scaling=...,
        use_bias=...,
        moe_intermediate_size=...,
        moe_k=...,
        moe_num_experts=...,
        moe_num_shared_experts=...,
        moe_layer_start_index=...,
        moe_layer_end_index=...,
        moe_layer_interval=...,
        moe_norm_min=...,
        output_router_logits=...,
        router_aux_loss_coef=...,
        **kwargs,
    ) -> None: ...

__all__ = ["Ernie4_5_MoeConfig"]
