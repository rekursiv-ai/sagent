from ...configuration_utils import PretrainedConfig
from ...utils.backbone_utils import BackboneConfigMixin

"""ConvNeXTV2 model configuration"""
logger = ...

class ConvNextV2Config(BackboneConfigMixin, PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        num_channels=...,
        patch_size=...,
        num_stages=...,
        hidden_sizes=...,
        depths=...,
        hidden_act=...,
        initializer_range=...,
        layer_norm_eps=...,
        drop_path_rate=...,
        image_size=...,
        out_features=...,
        out_indices=...,
        **kwargs,
    ) -> None: ...

__all__ = ["ConvNextV2Config"]
