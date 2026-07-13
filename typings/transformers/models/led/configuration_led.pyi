from ...configuration_utils import PretrainedConfig

"""LED model configuration"""
logger = ...

class LEDConfig(PretrainedConfig):
    model_type = ...
    attribute_map = ...
    def __init__(
        self,
        vocab_size=...,
        max_encoder_position_embeddings=...,
        max_decoder_position_embeddings=...,
        encoder_layers=...,
        encoder_ffn_dim=...,
        encoder_attention_heads=...,
        decoder_layers=...,
        decoder_ffn_dim=...,
        decoder_attention_heads=...,
        encoder_layerdrop=...,
        decoder_layerdrop=...,
        use_cache=...,
        is_encoder_decoder=...,
        activation_function=...,
        d_model=...,
        dropout=...,
        attention_dropout=...,
        activation_dropout=...,
        init_std=...,
        decoder_start_token_id=...,
        classifier_dropout=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        attention_window: list[int] | int = ...,
        **kwargs,
    ) -> None: ...

__all__ = ["LEDConfig"]
