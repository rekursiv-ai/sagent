from typing import Any

from torch import nn

import torch

from .configuration_mobilevit import MobileViTConfig
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutputWithNoAttention,
    BaseModelOutputWithPoolingAndNoAttention,
    ImageClassifierOutputWithNoAttention,
    SemanticSegmenterOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring

"""PyTorch MobileViT model."""
logger = ...

def make_divisible(
    value: int, divisor: int = ..., min_value: int | None = ...
) -> int: ...

class MobileViTConvLayer(nn.Module):
    def __init__(
        self,
        config: MobileViTConfig,
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
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class MobileViTInvertedResidual(nn.Module):
    def __init__(
        self,
        config: MobileViTConfig,
        in_channels: int,
        out_channels: int,
        stride: int,
        dilation: int = ...,
    ) -> None: ...
    def forward(self, features: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class MobileViTMobileNetLayer(nn.Module):
    def __init__(
        self,
        config: MobileViTConfig,
        in_channels: int,
        out_channels: int,
        stride: int = ...,
        num_stages: int = ...,
    ) -> None: ...
    def forward(self, features: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class MobileViTSelfAttention(nn.Module):
    def __init__(self, config: MobileViTConfig, hidden_size: int) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class MobileViTSelfOutput(nn.Module):
    def __init__(self, config: MobileViTConfig, hidden_size: int) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class MobileViTAttention(nn.Module):
    def __init__(self, config: MobileViTConfig, hidden_size: int) -> None: ...
    def prune_heads(self, heads: set[int]) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class MobileViTIntermediate(nn.Module):
    def __init__(
        self, config: MobileViTConfig, hidden_size: int, intermediate_size: int
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class MobileViTOutput(nn.Module):
    def __init__(
        self, config: MobileViTConfig, hidden_size: int, intermediate_size: int
    ) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class MobileViTTransformerLayer(nn.Module):
    def __init__(
        self, config: MobileViTConfig, hidden_size: int, intermediate_size: int
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class MobileViTTransformer(nn.Module):
    def __init__(
        self, config: MobileViTConfig, hidden_size: int, num_stages: int
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class MobileViTLayer(GradientCheckpointingLayer):
    def __init__(
        self,
        config: MobileViTConfig,
        in_channels: int,
        out_channels: int,
        stride: int,
        hidden_size: int,
        num_stages: int,
        dilation: int = ...,
    ) -> None: ...
    def unfolding(self, features: torch.Tensor) -> tuple[torch.Tensor, dict]: ...
    def folding(self, patches: torch.Tensor, info_dict: dict) -> torch.Tensor: ...
    def forward(self, features: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class MobileViTEncoder(nn.Module):
    def __init__(self, config: MobileViTConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ) -> tuple | BaseModelOutputWithNoAttention: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple | BaseModelOutputWithNoAttention: ...

@auto_docstring
class MobileViTPreTrainedModel(PreTrainedModel):
    config: MobileViTConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...

@auto_docstring
class MobileViTModel(MobileViTPreTrainedModel):
    def __init__(self, config: MobileViTConfig, expand_output: bool = ...) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutputWithPoolingAndNoAttention: ...

@auto_docstring(custom_intro=...)
class MobileViTForImageClassification(MobileViTPreTrainedModel):
    def __init__(self, config: MobileViTConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor | None = ...,
        output_hidden_states: bool | None = ...,
        labels: torch.Tensor | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | ImageClassifierOutputWithNoAttention: ...

class MobileViTASPPPooling(nn.Module):
    def __init__(
        self, config: MobileViTConfig, in_channels: int, out_channels: int
    ) -> None: ...
    def forward(self, features: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class MobileViTASPP(nn.Module):
    def __init__(self, config: MobileViTConfig) -> None: ...
    def forward(self, features: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class MobileViTDeepLabV3(nn.Module):
    def __init__(self, config: MobileViTConfig) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

@auto_docstring(custom_intro=...)
class MobileViTForSemanticSegmentation(MobileViTPreTrainedModel):
    def __init__(self, config: MobileViTConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | SemanticSegmenterOutput: ...

__all__ = [
    "MobileViTForImageClassification",
    "MobileViTForSemanticSegmentation",
    "MobileViTModel",
    "MobileViTPreTrainedModel",
]
