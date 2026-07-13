from ...configuration_utils import PretrainedConfig

class MoonshineConfig(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    attribute_map = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        intermediate_size=...,
        encoder_num_hidden_layers=...,
        decoder_num_hidden_layers=...,
        encoder_num_attention_heads=...,
        decoder_num_attention_heads=...,
        encoder_num_key_value_heads=...,
        decoder_num_key_value_heads=...,
        pad_head_dim_to_multiple_of=...,
        encoder_hidden_act=...,
        decoder_hidden_act=...,
        max_position_embeddings=...,
        initializer_range=...,
        decoder_start_token_id=...,
        use_cache=...,
        rope_theta=...,
        rope_scaling=...,
        partial_rotary_factor=...,
        is_encoder_decoder=...,
        attention_bias=...,
        attention_dropout=...,
        bos_token_id=...,
        eos_token_id=...,
        **kwargs,
    ) -> None: ...

__all__ = ["MoonshineConfig"]
