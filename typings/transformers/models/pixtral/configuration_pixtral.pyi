from ...configuration_utils import PretrainedConfig

"""Pixtral model configuration"""
logger = ...

class PixtralVisionConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        hidden_size=...,
        intermediate_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        num_channels=...,
        image_size=...,
        patch_size=...,
        hidden_act=...,
        attention_dropout=...,
        rope_theta=...,
        initializer_range=...,
        **kwargs,
    ) -> None: ...

__all__ = ["PixtralVisionConfig"]
