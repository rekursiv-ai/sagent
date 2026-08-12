from ...configuration_utils import PretrainedConfig

class VoxtralEncoderConfig(PretrainedConfig):
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        intermediate_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        scale_embedding=...,
        activation_function=...,
        num_mel_bins=...,
        max_source_positions=...,
        initializer_range=...,
        attention_dropout=...,
        **kwargs,
    ) -> None: ...

class VoxtralConfig(PretrainedConfig):
    _default_text_config_kwargs = ...
    def __init__(
        self,
        audio_config=...,
        text_config=...,
        audio_token_id=...,
        projector_hidden_act=...,
        **kwargs,
    ) -> None: ...

__all__ = ["VoxtralConfig", "VoxtralEncoderConfig"]
