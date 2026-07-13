from collections.abc import Mapping

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig

"""MobileViTV2 model configuration"""
logger = ...

class MobileViTV2Config(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        num_channels=...,
        image_size=...,
        patch_size=...,
        expand_ratio=...,
        hidden_act=...,
        conv_kernel_size=...,
        output_stride=...,
        classifier_dropout_prob=...,
        initializer_range=...,
        layer_norm_eps=...,
        aspp_out_channels=...,
        atrous_rates=...,
        aspp_dropout_prob=...,
        semantic_loss_ignore_index=...,
        n_attn_blocks=...,
        base_attn_unit_dims=...,
        width_multiplier=...,
        ffn_multiplier=...,
        attn_dropout=...,
        ffn_dropout=...,
        **kwargs,
    ) -> None: ...

class MobileViTV2OnnxConfig(OnnxConfig):
    torch_onnx_minimum_version = ...
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]: ...
    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]: ...
    @property
    def atol_for_validation(self) -> float: ...

__all__ = ["MobileViTV2Config", "MobileViTV2OnnxConfig"]
