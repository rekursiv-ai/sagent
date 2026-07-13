from collections.abc import Callable, Mapping

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig

"""Pvt model configuration"""
logger = ...

class PvtConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        image_size: int = ...,
        num_channels: int = ...,
        num_encoder_blocks: int = ...,
        depths: list[int] = ...,
        sequence_reduction_ratios: list[int] = ...,
        hidden_sizes: list[int] = ...,
        patch_sizes: list[int] = ...,
        strides: list[int] = ...,
        num_attention_heads: list[int] = ...,
        mlp_ratios: list[int] = ...,
        hidden_act: Mapping[str, Callable] = ...,
        hidden_dropout_prob: float = ...,
        attention_probs_dropout_prob: float = ...,
        initializer_range: float = ...,
        drop_path_rate: float = ...,
        layer_norm_eps: float = ...,
        qkv_bias: bool = ...,
        num_labels: int = ...,
        **kwargs,
    ) -> None: ...

class PvtOnnxConfig(OnnxConfig):
    torch_onnx_minimum_version = ...
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]: ...
    @property
    def atol_for_validation(self) -> float: ...
    @property
    def default_onnx_opset(self) -> int: ...

__all__ = ["PvtConfig", "PvtOnnxConfig"]
