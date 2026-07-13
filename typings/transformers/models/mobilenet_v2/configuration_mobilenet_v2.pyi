from collections.abc import Mapping

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig

"""MobileNetV2 model configuration"""
logger = ...

class MobileNetV2Config(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        num_channels=...,
        image_size=...,
        depth_multiplier=...,
        depth_divisible_by=...,
        min_depth=...,
        expand_ratio=...,
        output_stride=...,
        first_layer_is_expansion=...,
        finegrained_output=...,
        hidden_act=...,
        tf_padding=...,
        classifier_dropout_prob=...,
        initializer_range=...,
        layer_norm_eps=...,
        semantic_loss_ignore_index=...,
        **kwargs,
    ) -> None: ...

class MobileNetV2OnnxConfig(OnnxConfig):
    torch_onnx_minimum_version = ...
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]: ...
    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]: ...
    @property
    def atol_for_validation(self) -> float: ...

__all__ = ["MobileNetV2Config", "MobileNetV2OnnxConfig"]
