from typing import Any

from torch import Tensor, nn

import torch

from .configuration_resnet import ResNetConfig
from ...modeling_outputs import (
    BackboneOutput,
    BaseModelOutputWithNoAttention,
    BaseModelOutputWithPoolingAndNoAttention,
    ImageClassifierOutputWithNoAttention,
)
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring
from ...utils.backbone_utils import BackboneMixin

"""PyTorch ResNet model."""
logger = ...

class ResNetConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = ...,
        stride: int = ...,
        activation: str = ...,
    ) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...

class ResNetEmbeddings(nn.Module):
    def __init__(self, config: ResNetConfig) -> None: ...
    def forward(self, pixel_values: Tensor) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...

class ResNetShortCut(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, stride: int = ...
    ) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...

class ResNetBasicLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = ...,
        activation: str = ...,
    ) -> None: ...
    def forward(self, hidden_state): ...

class ResNetBottleNeckLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = ...,
        activation: str = ...,
        reduction: int = ...,
        downsample_in_bottleneck: bool = ...,
    ) -> None: ...
    def forward(self, hidden_state): ...

class ResNetStage(nn.Module):
    def __init__(
        self,
        config: ResNetConfig,
        in_channels: int,
        out_channels: int,
        stride: int = ...,
        depth: int = ...,
    ) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...

class ResNetEncoder(nn.Module):
    def __init__(self, config: ResNetConfig) -> None: ...
    def forward(
        self,
        hidden_state: Tensor,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ) -> BaseModelOutputWithNoAttention: ...
    def __call__(self, *args: Any, **kwargs: Any) -> BaseModelOutputWithNoAttention: ...

@auto_docstring
class ResNetPreTrainedModel(PreTrainedModel):
    config: ResNetConfig
    base_model_prefix = ...
    main_input_name = ...
    _no_split_modules = ...

@auto_docstring
class ResNetModel(ResNetPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: Tensor,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> BaseModelOutputWithPoolingAndNoAttention: ...

@auto_docstring(custom_intro=...)
class ResNetForImageClassification(ResNetPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> ImageClassifierOutputWithNoAttention: ...

@auto_docstring(custom_intro=...)
class ResNetBackbone(ResNetPreTrainedModel, BackboneMixin):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: Tensor,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> BackboneOutput: ...

__all__ = [
    "ResNetBackbone",
    "ResNetForImageClassification",
    "ResNetModel",
    "ResNetPreTrainedModel",
]
