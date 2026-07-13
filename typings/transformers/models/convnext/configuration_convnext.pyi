from collections.abc import Mapping

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils.backbone_utils import BackboneConfigMixin

"""ConvNeXT model configuration"""
logger = ...

class ConvNextConfig(BackboneConfigMixin, PretrainedConfig):
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
        layer_scale_init_value=...,
        drop_path_rate=...,
        image_size=...,
        out_features=...,
        out_indices=...,
        **kwargs,
    ) -> None: ...

class ConvNextOnnxConfig(OnnxConfig):
    torch_onnx_minimum_version = ...
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]: ...
    @property
    def atol_for_validation(self) -> float: ...

__all__ = ["ConvNextConfig", "ConvNextOnnxConfig"]
