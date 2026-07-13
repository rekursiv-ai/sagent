from ...configuration_utils import PretrainedConfig
from ...utils.backbone_utils import BackboneConfigMixin

"""BiT model configuration"""
logger = ...

class BitConfig(BackboneConfigMixin, PretrainedConfig):
    model_type = ...
    layer_types = ...
    supported_padding = ...
    def __init__(
        self,
        num_channels=...,
        embedding_size=...,
        hidden_sizes=...,
        depths=...,
        layer_type=...,
        hidden_act=...,
        global_padding=...,
        num_groups=...,
        drop_path_rate=...,
        embedding_dynamic_padding=...,
        output_stride=...,
        width_factor=...,
        out_features=...,
        out_indices=...,
        **kwargs,
    ) -> None: ...

__all__ = ["BitConfig"]
