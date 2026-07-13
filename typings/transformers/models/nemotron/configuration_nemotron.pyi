from ...configuration_utils import PretrainedConfig

"""Nemotron model configuration"""
logger = ...

class NemotronConfig(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        intermediate_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        head_dim=...,
        num_key_value_heads=...,
        hidden_act=...,
        max_position_embeddings=...,
        initializer_range=...,
        norm_eps=...,
        use_cache=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        tie_word_embeddings=...,
        rope_theta=...,
        partial_rotary_factor=...,
        attention_bias=...,
        attention_dropout=...,
        mlp_bias=...,
        **kwargs,
    ) -> None: ...

__all__ = ["NemotronConfig"]
