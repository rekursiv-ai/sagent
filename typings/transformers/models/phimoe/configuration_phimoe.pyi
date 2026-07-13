from ...configuration_utils import PretrainedConfig

"""PyTorch Phi-MoE model."""
logger = ...

class PhimoeConfig(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
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
        sliding_window=...,
        attention_dropout=...,
        num_experts_per_tok=...,
        num_local_experts=...,
        output_router_logits=...,
        router_aux_loss_coef=...,
        router_jitter_noise=...,
        input_jitter_noise=...,
        attention_bias=...,
        lm_head_bias=...,
        **kwargs,
    ) -> None: ...

__all__ = ["PhimoeConfig"]
