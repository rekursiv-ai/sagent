from dataclasses import dataclass
from typing import Any

from torch import nn

import torch

from .configuration_groupvit import (
    GroupViTConfig,
    GroupViTTextConfig,
    GroupViTVisionConfig,
)
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring, filter_out_non_signature_kwargs

"""PyTorch GroupViT model."""
logger = ...

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor: ...
def groupvit_loss(similarity: torch.Tensor) -> torch.Tensor: ...
def hard_softmax(logits: torch.Tensor, dim: int):  # -> Tensor:
    ...
def gumbel_softmax(
    logits: torch.Tensor, tau: float = ..., hard: bool = ..., dim: int = ...
) -> torch.Tensor: ...
def resize_attention_map(attentions, height, width, align_corners=...):  # -> Tensor:
    ...
def get_grouping_from_attentions(attentions, hw_shape): ...

class GroupViTCrossAttentionLayer(nn.Module):
    def __init__(self, config: GroupViTVisionConfig) -> None: ...
    def forward(self, query, key):  # -> Any:
        ...

class GroupViTAssignAttention(nn.Module):
    def __init__(self, config: GroupViTVisionConfig) -> None: ...
    def get_attn(self, attn, gumbel=..., hard=...):  # -> Tensor:
        ...
    def forward(self, query, key):  # -> tuple[Any, Tensor]:
        ...

class GroupViTTokenAssign(nn.Module):
    def __init__(
        self, config: GroupViTVisionConfig, num_group_token, num_output_group
    ) -> None: ...
    def project_group_token(self, group_tokens):  # -> Any:
        ...
    def forward(self, image_tokens, group_tokens):  # -> tuple[Any, Any]:
        ...

@dataclass
@auto_docstring
class GroupViTModelOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits_per_image: torch.FloatTensor | None = ...
    logits_per_text: torch.FloatTensor | None = ...
    segmentation_logits: torch.FloatTensor | None = ...
    text_embeds: torch.FloatTensor | None = ...
    image_embeds: torch.FloatTensor | None = ...
    text_model_output: BaseModelOutputWithPooling = ...
    vision_model_output: BaseModelOutputWithPooling = ...
    def to_tuple(self) -> tuple[Any]: ...

class GroupViTPatchEmbeddings(nn.Module):
    def __init__(
        self,
        image_size: int = ...,
        patch_size: int | tuple[int, int] = ...,
        num_channels: int = ...,
        embed_dim: int = ...,
    ) -> None: ...
    def forward(
        self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = ...
    ) -> torch.Tensor: ...

class GroupViTVisionEmbeddings(nn.Module):
    def __init__(self, config: GroupViTVisionConfig) -> None: ...
    def interpolate_pos_encoding(
        self, embeddings: torch.Tensor, height: int, width: int
    ) -> torch.Tensor: ...
    def forward(
        self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = ...
    ) -> torch.Tensor: ...

class GroupViTTextEmbeddings(nn.Module):
    def __init__(self, config: GroupViTTextConfig) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
    ) -> torch.Tensor: ...

class GroupViTStage(nn.Module):
    def __init__(
        self,
        config: GroupViTVisionConfig,
        depth: int,
        num_prev_group_token: int,
        num_group_token: int,
        num_output_group: int,
    ) -> None: ...
    @property
    def with_group_token(self):  # -> bool:
        ...
    def split_x(self, x):  # -> tuple[Any, Any] | tuple[Any, None]:
        ...
    def concat_x(
        self, x: torch.Tensor, group_token: torch.Tensor | None = ...
    ) -> torch.Tensor: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_group_token: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.FloatTensor]: ...

class GroupViTMLP(nn.Module):
    def __init__(
        self,
        config: GroupViTVisionConfig,
        hidden_size: int | None = ...,
        intermediate_size: int | None = ...,
        output_size: int | None = ...,
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class GroupViTMixerMLP(GroupViTMLP):
    def forward(self, x):  # -> Tensor:
        ...

class GroupViTAttention(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        causal_attention_mask: torch.Tensor | None = ...,
        encoder_hidden_states: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class GroupViTEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: GroupViTConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.FloatTensor]: ...

@auto_docstring
class GroupViTPreTrainedModel(PreTrainedModel):
    config: GroupViTConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...

class GroupViTVisionEncoder(nn.Module):
    def __init__(self, config: GroupViTVisionConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        output_hidden_states: bool | None = ...,
        output_attentions: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutput: ...

class GroupViTTextEncoder(nn.Module):
    def __init__(self, config: GroupViTTextConfig) -> None: ...
    def forward(
        self,
        inputs_embeds,
        attention_mask: torch.Tensor | None = ...,
        causal_attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutput: ...

class GroupViTTextTransformer(nn.Module):
    def __init__(self, config: GroupViTTextConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutputWithPooling: ...

class GroupViTTextModel(GroupViTPreTrainedModel):
    config: GroupViTTextConfig
    def __init__(self, config: GroupViTTextConfig) -> None: ...
    def get_input_embeddings(self) -> nn.Module: ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutputWithPooling: ...

class GroupViTVisionTransformer(nn.Module):
    def __init__(self, config: GroupViTVisionConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        output_hidden_states: bool | None = ...,
        output_attentions: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutputWithPooling: ...

class GroupViTVisionModel(GroupViTPreTrainedModel):
    config: GroupViTVisionConfig
    main_input_name = ...
    def __init__(self, config: GroupViTVisionConfig) -> None: ...
    def get_input_embeddings(self) -> GroupViTPatchEmbeddings: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutputWithPooling: ...

@auto_docstring
class GroupViTModel(GroupViTPreTrainedModel):
    config: GroupViTConfig
    def __init__(self, config: GroupViTConfig) -> None: ...
    @filter_out_non_signature_kwargs()
    @auto_docstring
    def get_text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
    ) -> torch.FloatTensor: ...
    @filter_out_non_signature_kwargs()
    @auto_docstring
    def get_image_features(self, pixel_values: torch.Tensor) -> torch.FloatTensor: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        return_loss: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        output_segmentation: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | GroupViTModelOutput: ...

__all__ = [
    "GroupViTModel",
    "GroupViTPreTrainedModel",
    "GroupViTTextModel",
    "GroupViTVisionModel",
]
