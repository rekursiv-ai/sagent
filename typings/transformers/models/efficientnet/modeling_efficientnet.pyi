from torch import nn

import torch

from .configuration_efficientnet import EfficientNetConfig
from ...modeling_outputs import (
    BaseModelOutputWithNoAttention,
    BaseModelOutputWithPoolingAndNoAttention,
    ImageClassifierOutputWithNoAttention,
)
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring

"""PyTorch EfficientNet model."""
logger = ...

def round_filters(config: EfficientNetConfig, num_channels: int):  # -> int:
    ...
def correct_pad(
    kernel_size: int | tuple, adjust: bool = ...
):  # -> tuple[Any, Any, Any, Any]:
    ...

class EfficientNetEmbeddings(nn.Module):
    def __init__(self, config: EfficientNetConfig) -> None: ...
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor: ...

class EfficientNetDepthwiseConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        depth_multiplier=...,
        kernel_size=...,
        stride=...,
        padding=...,
        dilation=...,
        bias=...,
        padding_mode=...,
    ) -> None: ...

class EfficientNetExpansionLayer(nn.Module):
    def __init__(
        self, config: EfficientNetConfig, in_dim: int, out_dim: int, stride: int
    ) -> None: ...
    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor: ...

class EfficientNetDepthwiseLayer(nn.Module):
    def __init__(
        self,
        config: EfficientNetConfig,
        in_dim: int,
        stride: int,
        kernel_size: int,
        adjust_padding: bool,
    ) -> None: ...
    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor: ...

class EfficientNetSqueezeExciteLayer(nn.Module):
    def __init__(
        self,
        config: EfficientNetConfig,
        in_dim: int,
        expand_dim: int,
        expand: bool = ...,
    ) -> None: ...
    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor: ...

class EfficientNetFinalBlockLayer(nn.Module):
    def __init__(
        self,
        config: EfficientNetConfig,
        in_dim: int,
        out_dim: int,
        stride: int,
        drop_rate: float,
        id_skip: bool,
    ) -> None: ...
    def forward(
        self, embeddings: torch.FloatTensor, hidden_states: torch.FloatTensor
    ) -> torch.Tensor: ...

class EfficientNetBlock(nn.Module):
    def __init__(
        self,
        config: EfficientNetConfig,
        in_dim: int,
        out_dim: int,
        stride: int,
        expand_ratio: int,
        kernel_size: int,
        drop_rate: float,
        id_skip: bool,
        adjust_padding: bool,
    ) -> None: ...
    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor: ...

class EfficientNetEncoder(nn.Module):
    def __init__(self, config: EfficientNetConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> BaseModelOutputWithNoAttention: ...

@auto_docstring
class EfficientNetPreTrainedModel(PreTrainedModel):
    config: EfficientNetConfig
    base_model_prefix = ...
    main_input_name = ...
    _no_split_modules = ...

@auto_docstring
class EfficientNetModel(EfficientNetPreTrainedModel):
    def __init__(self, config: EfficientNetConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutputWithPoolingAndNoAttention: ...

@auto_docstring(custom_intro=...)
class EfficientNetForImageClassification(EfficientNetPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | ImageClassifierOutputWithNoAttention: ...

__all__ = [
    "EfficientNetForImageClassification",
    "EfficientNetModel",
    "EfficientNetPreTrainedModel",
]
