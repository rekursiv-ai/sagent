from collections.abc import Mapping

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig

"""YOLOS model configuration"""
logger = ...

class YolosConfig(PretrainedConfig):
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
        num_detection_tokens=...,
        use_mid_position_embeddings=...,
        auxiliary_loss=...,
        class_cost=...,
        bbox_cost=...,
        giou_cost=...,
        bbox_loss_coefficient=...,
        giou_loss_coefficient=...,
        eos_coefficient=...,
        **kwargs,
    ) -> None: ...

class YolosOnnxConfig(OnnxConfig):
    torch_onnx_minimum_version = ...
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]: ...
    @property
    def atol_for_validation(self) -> float: ...
    @property
    def default_onnx_opset(self) -> int: ...

__all__ = ["YolosConfig", "YolosOnnxConfig"]
