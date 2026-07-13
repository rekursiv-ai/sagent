from ...configuration_utils import PretrainedConfig

class MiniMaxConfig(PretrainedConfig):
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
        head_dim=...,
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
        sliding_window=...,
        attention_dropout=...,
        num_experts_per_tok=...,
        num_local_experts=...,
        output_router_logits=...,
        router_aux_loss_coef=...,
        router_jitter_noise=...,
        layer_types=...,
        block_size=...,
        full_attn_alpha_factor=...,
        full_attn_beta_factor=...,
        linear_attn_alpha_factor=...,
        linear_attn_beta_factor=...,
        mlp_alpha_factor=...,
        mlp_beta_factor=...,
        **kwargs,
    ) -> None: ...

__all__ = ["MiniMaxConfig"]
