from collections.abc import Mapping

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils.backbone_utils import BackboneConfigMixin

"""ResNet model configuration"""
logger = ...

class ResNetConfig(BackboneConfigMixin, PretrainedConfig):
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

class ResNetOnnxConfig(OnnxConfig):
    torch_onnx_minimum_version = ...
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]: ...
    @property
    def atol_for_validation(self) -> float: ...

__all__ = ["ResNetConfig", "ResNetOnnxConfig"]
