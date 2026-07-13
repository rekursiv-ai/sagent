from collections.abc import Mapping

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig

"""MobileNetV1 model configuration"""
logger = ...

class MobileNetV1Config(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        num_channels=...,
        image_size=...,
        depth_multiplier=...,
        min_depth=...,
        hidden_act=...,
        tf_padding=...,
        classifier_dropout_prob=...,
        initializer_range=...,
        layer_norm_eps=...,
        **kwargs,
    ) -> None: ...

class MobileNetV1OnnxConfig(OnnxConfig):
    torch_onnx_minimum_version = ...
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]: ...
    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]: ...
    @property
    def atol_for_validation(self) -> float: ...

__all__ = ["MobileNetV1Config", "MobileNetV1OnnxConfig"]
