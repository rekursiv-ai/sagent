from collections.abc import Mapping

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig

"""EfficientNet model configuration"""
logger = ...

class EfficientNetConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        num_channels: int = ...,
        image_size: int = ...,
        width_coefficient: float = ...,
        depth_coefficient: float = ...,
        depth_divisor: int = ...,
        kernel_sizes: list[int] = ...,
        in_channels: list[int] = ...,
        out_channels: list[int] = ...,
        depthwise_padding: list[int] = ...,
        strides: list[int] = ...,
        num_block_repeats: list[int] = ...,
        expand_ratios: list[int] = ...,
        squeeze_expansion_ratio: float = ...,
        hidden_act: str = ...,
        hidden_dim: int = ...,
        pooling_type: str = ...,
        initializer_range: float = ...,
        batch_norm_eps: float = ...,
        batch_norm_momentum: float = ...,
        dropout_rate: float = ...,
        drop_connect_rate: float = ...,
        **kwargs,
    ) -> None: ...

class EfficientNetOnnxConfig(OnnxConfig):
    torch_onnx_minimum_version = ...
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]: ...
    @property
    def atol_for_validation(self) -> float: ...

__all__ = ["EfficientNetConfig", "EfficientNetOnnxConfig"]
