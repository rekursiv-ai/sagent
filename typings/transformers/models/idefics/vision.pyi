from dataclasses import dataclass

from torch import nn

import torch

from .configuration_idefics import IdeficsVisionConfig
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...utils import ModelOutput, can_return_tuple

"""PyTorch IdeficsVision model: a copy of CLIPVisionModel using a simpler config object"""
logger = ...

@dataclass
class IdeficsVisionModelOutput(ModelOutput):
    image_embeds: torch.FloatTensor | None = ...
    last_hidden_state: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...

class IdeficsVisionEmbeddings(nn.Module):
    def __init__(self, config: IdeficsVisionConfig) -> None: ...
    def interpolate_pos_encoding(
        self, embeddings: torch.Tensor, height: int, width: int
    ) -> torch.Tensor: ...
    def forward(
        self, pixel_values: torch.FloatTensor, interpolate_pos_encoding: bool = ...
    ) -> torch.Tensor: ...

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = ...,
    **kwargs,
):  # -> tuple[Tensor, Tensor]:
    ...

class IdeficsVisionAttention(nn.Module):
    def __init__(self, config: IdeficsVisionConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        causal_attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class IdeficsVisionMLP(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class IdeficsVisionEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: IdeficsVisionConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.FloatTensor]: ...

class IdeficsVisionEncoder(nn.Module):
    def __init__(self, config: IdeficsVisionConfig) -> None: ...
    @can_return_tuple
    def forward(
        self,
        inputs_embeds,
        attention_mask: torch.Tensor | None = ...,
        causal_attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutput: ...

class IdeficsVisionTransformer(nn.Module):
    def __init__(self, config: IdeficsVisionConfig) -> None: ...
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        interpolate_pos_encoding: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutputWithPooling: ...
