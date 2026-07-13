from torch import Tensor, nn

import torch

from .configuration_regnet import RegNetConfig
from ...modeling_outputs import (
    BaseModelOutputWithNoAttention,
    BaseModelOutputWithPoolingAndNoAttention,
    ImageClassifierOutputWithNoAttention,
)
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring

"""PyTorch RegNet model."""
logger = ...

class RegNetConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = ...,
        stride: int = ...,
        groups: int = ...,
        activation: str | None = ...,
    ) -> None: ...
    def forward(self, hidden_state):  # -> Any:
        ...

class RegNetEmbeddings(nn.Module):
    def __init__(self, config: RegNetConfig) -> None: ...
    def forward(self, pixel_values):  # -> Any:
        ...

class RegNetShortCut(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, stride: int = ...
    ) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class RegNetSELayer(nn.Module):
    def __init__(self, in_channels: int, reduced_channels: int) -> None: ...
    def forward(self, hidden_state): ...

class RegNetXLayer(nn.Module):
    def __init__(
        self,
        config: RegNetConfig,
        in_channels: int,
        out_channels: int,
        stride: int = ...,
    ) -> None: ...
    def forward(self, hidden_state): ...

class RegNetYLayer(nn.Module):
    def __init__(
        self,
        config: RegNetConfig,
        in_channels: int,
        out_channels: int,
        stride: int = ...,
    ) -> None: ...
    def forward(self, hidden_state): ...

class RegNetStage(nn.Module):
    def __init__(
        self,
        config: RegNetConfig,
        in_channels: int,
        out_channels: int,
        stride: int = ...,
        depth: int = ...,
    ) -> None: ...
    def forward(self, hidden_state):  # -> Any:
        ...

class RegNetEncoder(nn.Module):
    def __init__(self, config: RegNetConfig) -> None: ...
    def forward(
        self,
        hidden_state: Tensor,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ) -> BaseModelOutputWithNoAttention: ...

@auto_docstring
class RegNetPreTrainedModel(PreTrainedModel):
    config: RegNetConfig
    base_model_prefix = ...
    main_input_name = ...
    _no_split_modules = ...

@auto_docstring
class RegNetModel(RegNetPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: Tensor,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> BaseModelOutputWithPoolingAndNoAttention: ...

@auto_docstring(custom_intro=...)
class RegNetForImageClassification(RegNetPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> ImageClassifierOutputWithNoAttention: ...

__all__ = ["RegNetForImageClassification", "RegNetModel", "RegNetPreTrainedModel"]
