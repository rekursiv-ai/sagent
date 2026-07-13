from ...configuration_utils import PretrainedConfig
from ...utils.backbone_utils import BackboneConfigMixin

"""RT-DETR ResNet model configuration"""
logger = ...

class RTDetrResNetConfig(BackboneConfigMixin, PretrainedConfig):
    model_type = ...
    layer_types = ...
    def __init__(
        self,
        num_channels=...,
        embedding_size=...,
        hidden_sizes=...,
        depths=...,
        layer_type=...,
        hidden_act=...,
        downsample_in_first_stage=...,
        downsample_in_bottleneck=...,
        out_features=...,
        out_indices=...,
        **kwargs,
    ) -> None: ...

__all__ = ["RTDetrResNetConfig"]
