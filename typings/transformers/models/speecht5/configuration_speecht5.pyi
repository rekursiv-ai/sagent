from ...configuration_utils import PretrainedConfig

"""SpeechT5 model configuration"""
logger = ...

class SpeechT5Config(PretrainedConfig):
    model_type = ...
    attribute_map = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        encoder_layers=...,
        encoder_attention_heads=...,
        encoder_ffn_dim=...,
        encoder_layerdrop=...,
        decoder_layers=...,
        decoder_ffn_dim=...,
        decoder_attention_heads=...,
        decoder_layerdrop=...,
        hidden_act=...,
        positional_dropout=...,
        hidden_dropout=...,
        attention_dropout=...,
        activation_dropout=...,
        initializer_range=...,
        layer_norm_eps=...,
        scale_embedding=...,
        feat_extract_norm=...,
        feat_proj_dropout=...,
        feat_extract_activation=...,
        conv_dim=...,
        conv_stride=...,
        conv_kernel=...,
        conv_bias=...,
        num_conv_pos_embeddings=...,
        num_conv_pos_embedding_groups=...,
        apply_spec_augment=...,
        mask_time_prob=...,
        mask_time_length=...,
        mask_time_min_masks=...,
        mask_feature_prob=...,
        mask_feature_length=...,
        mask_feature_min_masks=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        decoder_start_token_id=...,
        num_mel_bins=...,
        speech_decoder_prenet_layers=...,
        speech_decoder_prenet_units=...,
        speech_decoder_prenet_dropout=...,
        speaker_embedding_dim=...,
        speech_decoder_postnet_layers=...,
        speech_decoder_postnet_units=...,
        speech_decoder_postnet_kernel=...,
        speech_decoder_postnet_dropout=...,
        reduction_factor=...,
        max_speech_positions=...,
        max_text_positions=...,
        encoder_max_relative_position=...,
        use_guided_attention_loss=...,
        guided_attention_loss_num_heads=...,
        guided_attention_loss_sigma=...,
        guided_attention_loss_scale=...,
        use_cache=...,
        is_encoder_decoder=...,
        **kwargs,
    ) -> None: ...
    def inputs_to_logits_ratio(self):  # -> Any:
        ...

class SpeechT5HifiGanConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        model_in_dim=...,
        sampling_rate=...,
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

__all__ = ["SpeechT5Config", "SpeechT5HifiGanConfig"]
