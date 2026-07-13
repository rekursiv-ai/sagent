from ...configuration_utils import PretrainedConfig

"""StableLM model configuration"""
logger = ...

class StableLmConfig(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    def __init__(
        self,
        vocab_size=...,
        intermediate_size=...,
        hidden_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        num_key_value_heads=...,
        hidden_act=...,
        max_position_embeddings=...,
        initializer_range=...,
        layer_norm_eps=...,
        use_cache=...,
        tie_word_embeddings=...,
        rope_theta=...,
        rope_scaling=...,
        use_qkv_bias=...,
        qk_layernorm=...,
        use_parallel_residual=...,
        hidden_dropout=...,
        attention_dropout=...,
        partial_rotary_factor=...,
        bos_token_id=...,
        eos_token_id=...,
        **kwargs,
    ) -> None: ...

__all__ = ["StableLmConfig"]
