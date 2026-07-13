from ...configuration_utils import PretrainedConfig

"""Qwen3-Next model configuration"""
logger = ...

class Qwen3NextConfig(PretrainedConfig):
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
        num_key_value_heads=...,
        hidden_act=...,
        max_position_embeddings=...,
        initializer_range=...,
        rms_norm_eps=...,
        use_cache=...,
        tie_word_embeddings=...,
        rope_theta=...,
        rope_scaling=...,
        partial_rotary_factor=...,
        attention_bias=...,
        attention_dropout=...,
        head_dim=...,
        linear_conv_kernel_dim=...,
        linear_key_head_dim=...,
        linear_value_head_dim=...,
        linear_num_key_heads=...,
        linear_num_value_heads=...,
        decoder_sparse_step=...,
        moe_intermediate_size=...,
        shared_expert_intermediate_size=...,
        num_experts_per_tok=...,
        num_experts=...,
        norm_topk_prob=...,
        output_router_logits=...,
        router_aux_loss_coef=...,
        mlp_only_layers=...,
        layer_types=...,
        **kwargs,
    ) -> None: ...

__all__ = ["Qwen3NextConfig"]
