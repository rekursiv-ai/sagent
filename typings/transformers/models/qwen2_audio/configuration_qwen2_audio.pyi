from ...configuration_utils import PretrainedConfig

"""Qwen2Audio model configuration"""
logger = ...

class Qwen2AudioEncoderConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        num_mel_bins=...,
        encoder_layers=...,
        encoder_attention_heads=...,
        encoder_ffn_dim=...,
        encoder_layerdrop=...,
        d_model=...,
        dropout=...,
        attention_dropout=...,
        activation_function=...,
        activation_dropout=...,
        scale_embedding=...,
        initializer_range=...,
        max_source_positions=...,
        **kwargs,
    ) -> None: ...

class Qwen2AudioConfig(PretrainedConfig):
    model_type = ...
    attribute_map = ...
    sub_configs = ...
    def __init__(
        self, audio_config=..., text_config=..., audio_token_index=..., **kwargs
    ) -> None: ...

__all__ = ["Qwen2AudioConfig", "Qwen2AudioEncoderConfig"]
