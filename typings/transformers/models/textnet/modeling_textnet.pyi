from typing import Any

from torch import Tensor, nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import (
    BackboneOutput,
    BaseModelOutputWithNoAttention,
    BaseModelOutputWithPoolingAndNoAttention,
    ImageClassifierOutputWithNoAttention,
)
from transformers.models.textnet.configuration_textnet import TextNetConfig
from transformers.utils.backbone_utils import BackboneMixin

import torch

from ...utils import auto_docstring

"""PyTorch TextNet model."""
logger = ...

class TextNetConvLayer(nn.Module):
    def __init__(self, config: TextNetConfig) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class TextNetRepConvLayer(nn.Module):
    def __init__(
        self,
        config: TextNetConfig,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class TextNetStage(nn.Module):
    def __init__(self, config: TextNetConfig, depth: int) -> None: ...
    def forward(self, hidden_state):  # -> Any:
        ...

class TextNetEncoder(nn.Module):
    def __init__(self, config: TextNetConfig) -> None: ...
    def forward(
        self,
        hidden_state: torch.Tensor,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> BaseModelOutputWithNoAttention: ...

@auto_docstring
class TextNetPreTrainedModel(PreTrainedModel):
    config: TextNetConfig
    base_model_prefix = ...
    main_input_name = ...

@auto_docstring
class TextNetModel(TextNetPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: Tensor,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> (
        tuple[Any, list[Any]] | tuple[Any] | BaseModelOutputWithPoolingAndNoAttention
    ): ...

@auto_docstring(custom_intro=...)
class TextNetForImageClassification(TextNetPreTrainedModel):
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
class TextNetBackbone(TextNetPreTrainedModel, BackboneMixin):
    has_attentions = ...
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: Tensor,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[tuple] | BackboneOutput: ...

__all__ = [
    "TextNetBackbone",
    "TextNetForImageClassification",
    "TextNetModel",
    "TextNetPreTrainedModel",
]
