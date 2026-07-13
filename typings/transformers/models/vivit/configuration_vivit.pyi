from ...configuration_utils import PretrainedConfig

"""ViViT model configuration"""
logger = ...

class VivitConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        image_size=...,
        num_frames=...,
        tubelet_size=...,
        num_channels=...,
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
        **kwargs,
    ) -> None: ...

__all__ = ["VivitConfig"]
