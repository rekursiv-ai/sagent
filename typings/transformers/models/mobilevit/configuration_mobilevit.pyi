from collections.abc import Mapping

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig

"""MobileViT model configuration"""
logger = ...

class MobileViTConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        num_channels=...,
        image_size=...,
        patch_size=...,
        hidden_sizes=...,
        neck_hidden_sizes=...,
        num_attention_heads=...,
        mlp_ratio=...,
        expand_ratio=...,
        hidden_act=...,
        conv_kernel_size=...,
        output_stride=...,
        hidden_dropout_prob=...,
        attention_probs_dropout_prob=...,
        classifier_dropout_prob=...,
        initializer_range=...,
        layer_norm_eps=...,
        qkv_bias=...,
        aspp_out_channels=...,
        atrous_rates=...,
        aspp_dropout_prob=...,
        semantic_loss_ignore_index=...,
        **kwargs,
    ) -> None: ...

class MobileViTOnnxConfig(OnnxConfig):
    torch_onnx_minimum_version = ...
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]: ...
    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]: ...
    @property
    def atol_for_validation(self) -> float: ...

__all__ = ["MobileViTConfig", "MobileViTOnnxConfig"]
