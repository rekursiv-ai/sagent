from ...configuration_utils import PretrainedConfig

"""FastSpeech2Conformer model configuration"""
logger = ...

class FastSpeech2ConformerConfig(PretrainedConfig):
    model_type = ...
    base_config_key = ...
    attribute_map = ...
    def __init__(
        self,
        hidden_size=...,
        vocab_size=...,
        num_mel_bins=...,
        encoder_num_attention_heads=...,
        encoder_layers=...,
        encoder_linear_units=...,
        decoder_layers=...,
        decoder_num_attention_heads=...,
        decoder_linear_units=...,
        speech_decoder_postnet_layers=...,
        speech_decoder_postnet_units=...,
        speech_decoder_postnet_kernel=...,
        positionwise_conv_kernel_size=...,
        encoder_normalize_before=...,
        decoder_normalize_before=...,
        encoder_concat_after=...,
        decoder_concat_after=...,
        reduction_factor=...,
        speaking_speed=...,
        use_macaron_style_in_conformer=...,
        use_cnn_in_conformer=...,
        encoder_kernel_size=...,
        decoder_kernel_size=...,
        duration_predictor_layers=...,
        duration_predictor_channels=...,
        duration_predictor_kernel_size=...,
        energy_predictor_layers=...,
        energy_predictor_channels=...,
        energy_predictor_kernel_size=...,
        energy_predictor_dropout=...,
        energy_embed_kernel_size=...,
        energy_embed_dropout=...,
        stop_gradient_from_energy_predictor=...,
        pitch_predictor_layers=...,
        pitch_predictor_channels=...,
        pitch_predictor_kernel_size=...,
        pitch_predictor_dropout=...,
        pitch_embed_kernel_size=...,
        pitch_embed_dropout=...,
        stop_gradient_from_pitch_predictor=...,
        encoder_dropout_rate=...,
        encoder_positional_dropout_rate=...,
        encoder_attention_dropout_rate=...,
        decoder_dropout_rate=...,
        decoder_positional_dropout_rate=...,
        decoder_attention_dropout_rate=...,
        duration_predictor_dropout_rate=...,
        speech_decoder_postnet_dropout=...,
        max_source_positions=...,
        use_masking=...,
        use_weighted_masking=...,
        num_speakers=...,
        num_languages=...,
        speaker_embed_dim=...,
        is_encoder_decoder=...,
        **kwargs,
    ) -> None: ...

class FastSpeech2ConformerHifiGanConfig(PretrainedConfig):
    model_type = ...
    base_config_key = ...
    def __init__(
        self,
        model_in_dim=...,
        upsample_initial_channel=...,
        upsample_rates=...,
        upsample_kernel_sizes=...,
        resblock_kernel_sizes=...,
        resblock_dilation_sizes=...,
        initializer_range=...,
        leaky_relu_slope=...,
        normalize_before=...,
        **kwargs,
    ) -> None: ...

class FastSpeech2ConformerWithHifiGanConfig(PretrainedConfig):
    model_type = ...
    sub_configs = ...
    def __init__(
        self,
        model_config: dict | None = ...,
        vocoder_config: dict | None = ...,
        **kwargs,
    ) -> None: ...

__all__ = [
    "FastSpeech2ConformerConfig",
    "FastSpeech2ConformerHifiGanConfig",
    "FastSpeech2ConformerWithHifiGanConfig",
]
