from torch import nn

import torch

from .configuration_mobilevitv2 import MobileViTV2Config
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutputWithNoAttention,
    BaseModelOutputWithPoolingAndNoAttention,
    ImageClassifierOutputWithNoAttention,
    SemanticSegmenterOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring

"""PyTorch MobileViTV2 model."""
logger = ...

def make_divisible(
    value: int, divisor: int = ..., min_value: int | None = ...
) -> int: ...
def clip(value: float, min_val: float = ..., max_val: float = ...) -> float: ...

class MobileViTV2ConvLayer(nn.Module):
    def __init__(
        self,
        config: MobileViTV2Config,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = ...,
        groups: int = ...,
        bias: bool = ...,
        dilation: int = ...,
        use_normalization: bool = ...,
        use_activation: bool | str = ...,
    ) -> None: ...
    def forward(self, features: torch.Tensor) -> torch.Tensor: ...

class MobileViTV2InvertedResidual(nn.Module):
    def __init__(
        self,
        config: MobileViTV2Config,
        in_channels: int,
        out_channels: int,
        stride: int,
        dilation: int = ...,
    ) -> None: ...
    def forward(self, features: torch.Tensor) -> torch.Tensor: ...

class MobileViTV2MobileNetLayer(nn.Module):
    def __init__(
        self,
        config: MobileViTV2Config,
        in_channels: int,
        out_channels: int,
        stride: int = ...,
        num_stages: int = ...,
    ) -> None: ...
    def forward(self, features: torch.Tensor) -> torch.Tensor: ...

class MobileViTV2LinearSelfAttention(nn.Module):
    def __init__(self, config: MobileViTV2Config, embed_dim: int) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class MobileViTV2FFN(nn.Module):
    def __init__(
        self,
        config: MobileViTV2Config,
        embed_dim: int,
        ffn_latent_dim: int,
        ffn_dropout: float = ...,
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class MobileViTV2TransformerLayer(nn.Module):
    def __init__(
        self,
        config: MobileViTV2Config,
        embed_dim: int,
        ffn_latent_dim: int,
        dropout: float = ...,
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class MobileViTV2Transformer(nn.Module):
    def __init__(
        self, config: MobileViTV2Config, n_layers: int, d_model: int
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class MobileViTV2Layer(GradientCheckpointingLayer):
    def __init__(
        self,
        config: MobileViTV2Config,
        in_channels: int,
        out_channels: int,
        attn_unit_dim: int,
        n_attn_blocks: int = ...,
        dilation: int = ...,
        stride: int = ...,
    ) -> None: ...
    def unfolding(
        self, feature_map: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, int]]: ...
    def folding(
        self, patches: torch.Tensor, output_size: tuple[int, int]
    ) -> torch.Tensor: ...
    def forward(self, features: torch.Tensor) -> torch.Tensor: ...

class MobileViTV2Encoder(nn.Module):
    def __init__(self, config: MobileViTV2Config) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ) -> tuple | BaseModelOutputWithNoAttention: ...

@auto_docstring
class MobileViTV2PreTrainedModel(PreTrainedModel):
    config: MobileViTV2Config
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...

@auto_docstring
class MobileViTV2Model(MobileViTV2PreTrainedModel):
    def __init__(
        self, config: MobileViTV2Config, expand_output: bool = ...
    ) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutputWithPoolingAndNoAttention: ...

@auto_docstring(custom_intro=...)
class MobileViTV2ForImageClassification(MobileViTV2PreTrainedModel):
    def __init__(self, config: MobileViTV2Config) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor | None = ...,
        output_hidden_states: bool | None = ...,
        labels: torch.Tensor | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | ImageClassifierOutputWithNoAttention: ...

class MobileViTV2ASPPPooling(nn.Module):
    def __init__(
        self, config: MobileViTV2Config, in_channels: int, out_channels: int
    ) -> None: ...
    def forward(self, features: torch.Tensor) -> torch.Tensor: ...

class MobileViTV2ASPP(nn.Module):
    def __init__(self, config: MobileViTV2Config) -> None: ...
    def forward(self, features: torch.Tensor) -> torch.Tensor: ...

class MobileViTV2DeepLabV3(nn.Module):
    def __init__(self, config: MobileViTV2Config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

@auto_docstring(custom_intro=...)
class MobileViTV2ForSemanticSegmentation(MobileViTV2PreTrainedModel):
    def __init__(self, config: MobileViTV2Config) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | SemanticSegmenterOutput: ...

__all__ = [
    "MobileViTV2ForImageClassification",
    "MobileViTV2ForSemanticSegmentation",
    "MobileViTV2Model",
    "MobileViTV2PreTrainedModel",
]
