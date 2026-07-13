from torch import Tensor, nn

import torch

from .configuration_bit import BitConfig
from ...modeling_outputs import (
    BackboneOutput,
    BaseModelOutputWithNoAttention,
    BaseModelOutputWithPoolingAndNoAttention,
    ImageClassifierOutputWithNoAttention,
)
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring
from ...utils.backbone_utils import BackboneMixin

"""PyTorch BiT model. Also supports backbone for ViT hybrid."""
logger = ...

def get_padding_value(
    padding=..., kernel_size=..., stride=..., dilation=...
) -> tuple[tuple, bool]: ...

class WeightStandardizedConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channel,
        out_channels,
        kernel_size,
        stride=...,
        padding=...,
        dilation=...,
        groups=...,
        bias=...,
        eps=...,
    ) -> None: ...
    def forward(self, hidden_state):  # -> Tensor:
        ...

class BitGroupNormActivation(nn.GroupNorm):
    def __init__(
        self, config, num_channels, eps=..., affine=..., apply_activation=...
    ) -> None: ...
    def forward(self, hidden_state):  # -> Any:
        ...

class DynamicPad2d(nn.Module):
    def __init__(self, kernel_size, stride, dilation, value=...) -> None: ...
    def forward(self, input):  # -> Tensor:
        ...

class BitMaxPool2d(nn.MaxPool2d):
    def __init__(
        self,
        kernel_size: int,
        stride=...,
        dilation=...,
        ceil_mode=...,
        padding=...,
        padding_value=...,
        use_dynamic_padding=...,
    ) -> None: ...
    def forward(self, hidden_states):  # -> Tensor:
        ...

class BitEmbeddings(nn.Module):
    def __init__(self, config: BitConfig) -> None: ...
    def forward(self, pixel_values: Tensor) -> Tensor: ...

def drop_path(
    input: torch.Tensor, drop_prob: float = ..., training: bool = ...
) -> torch.Tensor: ...

class BitDropPath(nn.Module):
    def __init__(self, drop_prob: float | None = ...) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def extra_repr(self) -> str: ...

def make_div(value, divisor=...):  # -> int:
    ...

class BitPreActivationBottleneckLayer(nn.Module):
    def __init__(
        self,
        config,
        in_channels,
        out_channels=...,
        bottle_ratio=...,
        stride=...,
        dilation=...,
        first_dilation=...,
        groups=...,
        drop_path_rate=...,
        is_first_layer=...,
    ) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class BitBottleneckLayer(nn.Module):
    def __init__(
        self,
        config,
        in_channels,
        out_channels=...,
        bottle_ratio=...,
        stride=...,
        dilation=...,
        first_dilation=...,
        groups=...,
        drop_path_rate=...,
        is_first_layer=...,
    ) -> None: ...
    def forward(self, hidden_states): ...

class BitDownsampleConv(nn.Module):
    def __init__(
        self, config, in_channels, out_channels, stride=..., preact=...
    ) -> None: ...
    def forward(self, x):  # -> Any:
        ...

class BitStage(nn.Module):
    def __init__(
        self,
        config,
        in_channels,
        out_channels,
        stride,
        dilation,
        depth,
        bottle_ratio=...,
        layer_dropout=...,
    ) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class BitEncoder(nn.Module):
    def __init__(self, config: BitConfig) -> None: ...
    def forward(
        self,
        hidden_state: Tensor,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ) -> BaseModelOutputWithNoAttention: ...

@auto_docstring
class BitPreTrainedModel(PreTrainedModel):
    config: BitConfig
    base_model_prefix = ...
    main_input_name = ...
    _no_split_modules = ...

@auto_docstring
class BitModel(BitPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: Tensor,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> BaseModelOutputWithPoolingAndNoAttention: ...

@auto_docstring(custom_intro=...)
class BitForImageClassification(BitPreTrainedModel):
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
class BitBackbone(BitPreTrainedModel, BackboneMixin):
    has_attentions = ...
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: Tensor,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> BackboneOutput: ...

__all__ = ["BitBackbone", "BitForImageClassification", "BitModel", "BitPreTrainedModel"]
