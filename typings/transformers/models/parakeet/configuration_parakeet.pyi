from ...configuration_utils import PretrainedConfig

"""Parakeet model configuration."""
logger = ...

class ParakeetEncoderConfig(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    def __init__(
        self,
        hidden_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        intermediate_size=...,
        hidden_act=...,
        attention_bias=...,
        conv_kernel_size=...,
        subsampling_factor=...,
        subsampling_conv_channels=...,
        num_mel_bins=...,
        subsampling_conv_kernel_size=...,
        subsampling_conv_stride=...,
        dropout=...,
        dropout_positions=...,
        layerdrop=...,
        activation_dropout=...,
        attention_dropout=...,
        max_position_embeddings=...,
        scale_input=...,
        initializer_range=...,
        **kwargs,
    ) -> None: ...

class ParakeetCTCConfig(PretrainedConfig):
    model_type = ...
    sub_configs = ...
    def __init__(
        self,
        vocab_size=...,
        ctc_loss_reduction=...,
        ctc_zero_infinity=...,
        encoder_config: dict | ParakeetEncoderConfig = ...,
        pad_token_id=...,
        **kwargs,
    ) -> None: ...
    @classmethod
    def from_encoder_config(
        cls, encoder_config: ParakeetEncoderConfig, **kwargs
    ):  # -> Self:
        ...

__all__ = ["ParakeetCTCConfig", "ParakeetEncoderConfig"]
