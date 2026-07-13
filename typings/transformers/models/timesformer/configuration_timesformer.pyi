from ...configuration_utils import PretrainedConfig

"""TimeSformer model configuration"""
logger = ...

class TimesformerConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        image_size=...,
        patch_size=...,
        num_channels=...,
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
        attention_type=...,
        drop_path_rate=...,
        **kwargs,
    ) -> None: ...

__all__ = ["TimesformerConfig"]
