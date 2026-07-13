from ...configuration_utils import PretrainedConfig

class DogeConfig(PretrainedConfig):
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
        hidden_dropout=...,
        hidden_act=...,
        initializer_range=...,
        rms_norm_eps=...,
        use_cache=...,
        tie_word_embeddings=...,
        max_position_embeddings=...,
        rope_theta=...,
        rope_scaling=...,
        num_attention_heads=...,
        num_key_value_heads=...,
        attention_bias=...,
        attention_dropout=...,
        mlp_bias=...,
        sliding_window=...,
        keep_window_size=...,
        is_moe=...,
        num_experts=...,
        num_experts_per_tok=...,
        norm_topk_prob=...,
        output_router_logits=...,
        router_aux_loss_coef=...,
        **kwargs,
    ) -> None: ...

__all__ = ["DogeConfig"]
