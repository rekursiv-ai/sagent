from ...configuration_utils import PretrainedConfig

"""VideoMAE model configuration"""
logger = ...

class VideoMAEConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        image_size=...,
        patch_size=...,
        num_channels=...,
        num_frames=...,
        tubelet_size=...,
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
        norm_pix_loss=...,
        **kwargs,
    ) -> None: ...

__all__ = ["VideoMAEConfig"]
