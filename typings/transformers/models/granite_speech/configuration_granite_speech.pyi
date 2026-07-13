from ...configuration_utils import PretrainedConfig

"""Config class for Granite Speech."""

class GraniteSpeechEncoderConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        input_dim=...,
        num_layers=...,
        hidden_dim=...,
        feedforward_mult=...,
        num_heads=...,
        dim_head=...,
        output_dim=...,
        context_size=...,
        max_pos_emb=...,
        dropout=...,
        conv_kernel_size=...,
        conv_expansion_factor=...,
        **kwargs,
    ) -> None: ...

class GraniteSpeechConfig(PretrainedConfig):
    model_type = ...
    attribute_map = ...
    sub_configs = ...
    def __init__(
        self,
        text_config=...,
        encoder_config=...,
        projector_config=...,
        audio_token_index=...,
        initializer_range=...,
        has_lora_adapter=...,
        downsample_rate=...,
        window_size=...,
        **kwargs,
    ) -> None: ...

__all__ = ["GraniteSpeechConfig", "GraniteSpeechEncoderConfig"]
