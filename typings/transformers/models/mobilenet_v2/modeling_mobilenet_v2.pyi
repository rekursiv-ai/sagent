from typing import Any

from torch import nn

import torch

from .configuration_mobilenet_v2 import MobileNetV2Config
from ...modeling_outputs import (
    BaseModelOutputWithPoolingAndNoAttention,
    ImageClassifierOutputWithNoAttention,
    SemanticSegmenterOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring

"""PyTorch MobileNetV2 model."""
logger = ...

def load_tf_weights_in_mobilenet_v2(model, config, tf_checkpoint_path): ...
def make_divisible(
    value: int, divisor: int = ..., min_value: int | None = ...
) -> int: ...
def apply_depth_multiplier(config: MobileNetV2Config, channels: int) -> int: ...
def apply_tf_padding(features: torch.Tensor, conv_layer: nn.Conv2d) -> torch.Tensor: ...

class MobileNetV2ConvLayer(nn.Module):
    def __init__(
        self,
        config: MobileNetV2Config,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = ...,
        groups: int = ...,
        bias: bool = ...,
        dilation: int = ...,
        use_normalization: bool = ...,
        use_activation: bool | str = ...,
        layer_norm_eps: float | None = ...,
    ) -> None: ...
    def forward(self, features: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class MobileNetV2InvertedResidual(nn.Module):
    def __init__(
        self,
        config: MobileNetV2Config,
        in_channels: int,
        out_channels: int,
        stride: int,
        dilation: int = ...,
    ) -> None: ...
    def forward(self, features: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class MobileNetV2Stem(nn.Module):
    def __init__(
        self,
        config: MobileNetV2Config,
        in_channels: int,
        expanded_channels: int,
        out_channels: int,
    ) -> None: ...
    def forward(self, features: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

@auto_docstring
class MobileNetV2PreTrainedModel(PreTrainedModel):
    config: MobileNetV2Config
    load_tf_weights = ...
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...

@auto_docstring
class MobileNetV2Model(MobileNetV2PreTrainedModel):
    def __init__(
        self, config: MobileNetV2Config, add_pooling_layer: bool = ...
    ) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutputWithPoolingAndNoAttention: ...

@auto_docstring(custom_intro=...)
class MobileNetV2ForImageClassification(MobileNetV2PreTrainedModel):
    def __init__(self, config: MobileNetV2Config) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor | None = ...,
        output_hidden_states: bool | None = ...,
        labels: torch.Tensor | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | ImageClassifierOutputWithNoAttention: ...

class MobileNetV2DeepLabV3Plus(nn.Module):
    def __init__(self, config: MobileNetV2Config) -> None: ...
    def forward(self, features: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

@auto_docstring(custom_intro=...)
class MobileNetV2ForSemanticSegmentation(MobileNetV2PreTrainedModel):
    def __init__(self, config: MobileNetV2Config) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | SemanticSegmenterOutput: ...

__all__ = [
    "MobileNetV2ForImageClassification",
    "MobileNetV2ForSemanticSegmentation",
    "MobileNetV2Model",
    "MobileNetV2PreTrainedModel",
    "load_tf_weights_in_mobilenet_v2",
]
