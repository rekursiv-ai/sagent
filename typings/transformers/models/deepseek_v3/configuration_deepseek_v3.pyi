from ...configuration_utils import PretrainedConfig

"""DeepSeekV3 model configuration"""
DEEPSEEK_PRETRAINED_CONFIG_ARCHIVE_MAP = ...

class DeepseekV3Config(PretrainedConfig):
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
        routed_scaling_factor=...,
        kv_lora_rank=...,
        q_lora_rank=...,
        qk_rope_head_dim=...,
        v_head_dim=...,
        qk_nope_head_dim=...,
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
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        pretraining_tp=...,
        tie_word_embeddings=...,
        rope_theta=...,
        rope_scaling=...,
        rope_interleave=...,
        attention_bias=...,
        attention_dropout=...,
        **kwargs,
    ) -> None: ...

__all__ = ["DeepseekV3Config"]
