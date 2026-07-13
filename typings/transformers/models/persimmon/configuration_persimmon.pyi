from ...configuration_utils import PretrainedConfig

"""Persimmon model configuration"""
logger = ...

class PersimmonConfig(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        intermediate_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        hidden_act=...,
        max_position_embeddings=...,
        initializer_range=...,
        layer_norm_eps=...,
        use_cache=...,
        tie_word_embeddings=...,
        rope_theta=...,
        rope_scaling=...,
        qk_layernorm=...,
        hidden_dropout=...,
        attention_dropout=...,
        partial_rotary_factor=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        **kwargs,
    ) -> None: ...

__all__ = ["PersimmonConfig"]
