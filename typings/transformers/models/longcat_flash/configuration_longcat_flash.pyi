from ...configuration_utils import PretrainedConfig

"""LongCat Flash model configuration"""

class LongcatFlashConfig(PretrainedConfig):
    keys_to_ignore_at_inference = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        num_hidden_layers=...,
        num_layers=...,
        num_attention_heads=...,
        num_key_value_heads=...,
        hidden_act=...,
        max_position_embeddings=...,
        initializer_range=...,
        rms_norm_eps=...,
        use_cache=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        tie_word_embeddings=...,
        rope_theta=...,
        rope_scaling=...,
        attention_bias=...,
        attention_dropout=...,
        ffn_hidden_size=...,
        q_lora_rank=...,
        kv_lora_rank=...,
        qk_nope_head_dim=...,
        qk_rope_head_dim=...,
        head_dim=...,
        v_head_dim=...,
        qk_head_dim=...,
        moe_topk=...,
        n_routed_experts=...,
        zero_expert_num=...,
        expert_ffn_hidden_size=...,
        routed_scaling_factor=...,
        **kwargs,
    ) -> None: ...

__all__ = ["LongcatFlashConfig"]
