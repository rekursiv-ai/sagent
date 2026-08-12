from ...configuration_utils import PretrainedConfig

"""MusicGen model configuration"""
logger = ...

class MusicgenDecoderConfig(PretrainedConfig):
    keys_to_ignore_at_inference = ...
    def __init__(
        self,
        vocab_size=...,
        max_position_embeddings=...,
        num_hidden_layers=...,
        ffn_dim=...,
        num_attention_heads=...,
        layerdrop=...,
        use_cache=...,
        activation_function=...,
        hidden_size=...,
        dropout=...,
        attention_dropout=...,
        activation_dropout=...,
        initializer_factor=...,
        scale_embedding=...,
        num_codebooks=...,
        audio_channels=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        tie_word_embeddings=...,
        **kwargs,
    ) -> None: ...

class MusicgenConfig(PretrainedConfig):
    def __init__(self, **kwargs) -> None: ...
    @classmethod
    def from_sub_models_config(
        cls,
        text_encoder_config: PretrainedConfig,
        audio_encoder_config: PretrainedConfig,
        decoder_config: MusicgenDecoderConfig,
        **kwargs,
    ):  # -> Self:
        ...
    @property
    def sampling_rate(self):  # -> Any:
        ...

__all__ = ["MusicgenConfig", "MusicgenDecoderConfig"]
