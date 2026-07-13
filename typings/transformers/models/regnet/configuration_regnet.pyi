from ...configuration_utils import PretrainedConfig

"""RegNet model configuration"""
logger = ...

class RegNetConfig(PretrainedConfig):
    model_type = ...
    layer_types = ...
    def __init__(
        self,
        num_channels=...,
        embedding_size=...,
        hidden_sizes=...,
        depths=...,
        groups_width=...,
        layer_type=...,
        hidden_act=...,
        **kwargs,
    ) -> None: ...

__all__ = ["RegNetConfig"]
