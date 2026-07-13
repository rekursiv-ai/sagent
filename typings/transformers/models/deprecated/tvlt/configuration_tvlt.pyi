from ....configuration_utils import PretrainedConfig

"""TVLT model configuration"""
logger = ...

class TvltConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        image_size=...,
        spectrogram_length=...,
        frequency_length=...,
        image_patch_size=...,
        audio_patch_size=...,
        num_image_channels=...,
        num_audio_channels=...,
        num_frames=...,
        hidden_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        intermediate_size=...,
        hidden_act=...,
        hidden_dropout_prob=...,
        attention_probs_dropout_prob=...,
        initializer_range=...,
        layer_norm_eps=...,
        qkv_bias=...,
        use_mean_pooling=...,
        decoder_num_attention_heads=...,
        decoder_hidden_size=...,
        decoder_num_hidden_layers=...,
        decoder_intermediate_size=...,
        pixel_mask_ratio=...,
        audio_mask_ratio=...,
        audio_mask_type=...,
        task_matching=...,
        task_mae=...,
        loss_type=...,
        **kwargs,
    ) -> None: ...

__all__ = ["TvltConfig"]
