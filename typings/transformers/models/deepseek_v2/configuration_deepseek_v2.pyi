from ...configuration_utils import PretrainedConfig

class DeepseekV2Config(PretrainedConfig):
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
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        tie_word_embeddings=...,
        rope_theta=...,
        rope_scaling=...,
        attention_bias=...,
        attention_dropout=...,
        mlp_bias=...,
        aux_loss_alpha=...,
        first_k_dense_replace=...,
        kv_lora_rank=...,
        q_lora_rank=...,
        n_group=...,
        n_routed_experts=...,
        n_shared_experts=...,
        qk_nope_head_dim=...,
        qk_rope_head_dim=...,
        routed_scaling_factor=...,
        seq_aux=...,
        topk_group=...,
        topk_method=...,
        v_head_dim=...,
        num_experts_per_tok=...,
        norm_topk_prob=...,
        moe_intermediate_size=...,
        **kwargs,
    ) -> None: ...

__all__ = ["DeepseekV2Config"]
