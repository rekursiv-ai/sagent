from collections.abc import Mapping

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig

"""DeiT model configuration"""
logger = ...

class DeiTConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        hidden_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        intermediate_size=...,
        hidden_act=...,
        hidden_dropout_prob=...,
        attention_probs_dropout_prob=...,
        initializer_range=...,
        layer_norm_eps=...,
        image_size=...,
        patch_size=...,
        num_channels=...,
        qkv_bias=...,
        encoder_stride=...,
        pooler_output_size=...,
        pooler_act=...,
        **kwargs,
    ) -> None: ...

class DeiTOnnxConfig(OnnxConfig):
    torch_onnx_minimum_version = ...
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]: ...
    @property
    def atol_for_validation(self) -> float: ...

__all__ = ["DeiTConfig", "DeiTOnnxConfig"]
