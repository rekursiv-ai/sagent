from typing import Any
from torch import nn

import torch

from .configuration_pvt_v2 import PvtV2Config
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BackboneOutput, BaseModelOutput, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring
from ...utils.backbone_utils import BackboneMixin

"""PyTorch PVTv2 model."""
logger = ...

def drop_path(
    input: torch.Tensor, drop_prob: float = ..., training: bool = ...
) -> torch.Tensor: ...

class PvtV2DropPath(nn.Module):
    def __init__(self, drop_prob: float | None = ...) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...
    def extra_repr(self) -> str: ...

class PvtV2OverlapPatchEmbeddings(nn.Module):
    def __init__(self, config: PvtV2Config, layer_idx: int) -> None: ...
    def forward(self, pixel_values):  # -> tuple[Any, Any, Any]:
        ...

class PvtV2DepthWiseConv(nn.Module):
    def __init__(self, config: PvtV2Config, dim: int = ...) -> None: ...
    def forward(self, hidden_states, height, width):  # -> Any:
        ...

class PvtV2SelfAttention(nn.Module):
    def __init__(
        self,
        config: PvtV2Config,
        hidden_size: int,
        num_attention_heads: int,
        spatial_reduction_ratio: int,
    ) -> None: ...
    def transpose_for_scores(self, hidden_states) -> torch.Tensor: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        height: int,
        width: int,
        output_attentions: bool = ...,
    ) -> tuple[torch.Tensor]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor]: ...
    def prune_heads(self, heads):  # -> None:
        ...

class PvtV2ConvFeedForwardNetwork(nn.Module):
    def __init__(
        self,
        config: PvtV2Config,
        in_features: int,
        hidden_features: int | None = ...,
        out_features: int | None = ...,
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor, height, width) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class PvtV2BlockLayer(nn.Module):
    def __init__(
        self, config: PvtV2Config, layer_idx: int, drop_path: float = ...
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        height: int,
        width: int,
        output_attentions: bool = ...,
    ):  # -> Any:
        ...

class PvtV2EncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: PvtV2Config, layer_idx: int) -> None: ...
    def forward(
        self, hidden_states, output_attentions
    ):  # -> tuple[tuple[Any, tuple[()] | tuple[Any, ...] | None] | tuple[Any], Any, Any]:
        ...

class PvtV2Encoder(nn.Module):
    def __init__(self, config: PvtV2Config) -> None: ...
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutput: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple | BaseModelOutput: ...

@auto_docstring
class PvtV2PreTrainedModel(PreTrainedModel):
    config: PvtV2Config
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...

@auto_docstring
class PvtV2Model(PvtV2PreTrainedModel):
    def __init__(self, config: PvtV2Config) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutput: ...

@auto_docstring(custom_intro=...)
class PvtV2ForImageClassification(PvtV2PreTrainedModel):
    def __init__(self, config: PvtV2Config) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor | None,
        labels: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | ImageClassifierOutput: ...

@auto_docstring(custom_intro=...)
class PvtV2Backbone(PvtV2Model, BackboneMixin):
    def __init__(self, config: PvtV2Config) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> BackboneOutput: ...

__all__ = [
    "PvtV2Backbone",
    "PvtV2ForImageClassification",
    "PvtV2Model",
    "PvtV2PreTrainedModel",
]
