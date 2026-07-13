from dataclasses import dataclass
from typing import Any

from torch import nn

import torch

from .configuration_align import AlignConfig, AlignTextConfig, AlignVisionConfig
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    BaseModelOutputWithPoolingAndNoAttention,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    auto_docstring,
    can_return_tuple,
    filter_out_non_signature_kwargs,
)

"""PyTorch ALIGN model."""
logger = ...

@dataclass
@auto_docstring(custom_intro=...)
class AlignVisionModelOutput(ModelOutput):
    image_embeds: torch.FloatTensor | None = ...
    last_hidden_state: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class AlignTextModelOutput(ModelOutput):
    text_embeds: torch.FloatTensor | None = ...
    last_hidden_state: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...

@dataclass
@auto_docstring
class AlignOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits_per_image: torch.FloatTensor | None = ...
    logits_per_text: torch.FloatTensor | None = ...
    text_embeds: torch.FloatTensor | None = ...
    image_embeds: torch.FloatTensor | None = ...
    text_model_output: BaseModelOutputWithPooling = ...
    vision_model_output: BaseModelOutputWithPoolingAndNoAttention = ...
    def to_tuple(self) -> tuple[Any]: ...

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor: ...
def align_loss(similarity: torch.Tensor) -> torch.Tensor: ...
def round_filters(config: AlignVisionConfig, num_channels: int):  # -> int:
    ...
def correct_pad(
    kernel_size: int | tuple, adjust: bool = ...
):  # -> tuple[Any, Any, Any, Any]:
    ...

class AlignVisionEmbeddings(nn.Module):
    def __init__(self, config: AlignVisionConfig) -> None: ...
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor: ...

class AlignVisionDepthwiseConv2d(nn.Conv2d):
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

class AlignVisionExpansionLayer(nn.Module):
    def __init__(
        self, config: AlignVisionConfig, in_dim: int, out_dim: int, stride: int
    ) -> None: ...
    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor: ...

class AlignVisionDepthwiseLayer(nn.Module):
    def __init__(
        self,
        config: AlignVisionConfig,
        in_dim: int,
        stride: int,
        kernel_size: int,
        adjust_padding: bool,
    ) -> None: ...
    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor: ...

class AlignVisionSqueezeExciteLayer(nn.Module):
    def __init__(
        self,
        config: AlignVisionConfig,
        in_dim: int,
        expand_dim: int,
        expand: bool = ...,
    ) -> None: ...
    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor: ...

class AlignVisionFinalBlockLayer(nn.Module):
    def __init__(
        self,
        config: AlignVisionConfig,
        in_dim: int,
        out_dim: int,
        stride: int,
        drop_rate: float,
        id_skip: bool,
    ) -> None: ...
    def forward(
        self, embeddings: torch.FloatTensor, hidden_states: torch.FloatTensor
    ) -> torch.Tensor: ...

class AlignVisionBlock(nn.Module):
    def __init__(
        self,
        config: AlignVisionConfig,
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

class AlignVisionEncoder(nn.Module):
    def __init__(self, config: AlignVisionConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> BaseModelOutputWithPoolingAndNoAttention: ...

class AlignTextEmbeddings(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
    ) -> torch.Tensor: ...

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = ...,
    head_mask: torch.Tensor | None = ...,
    **kwargs,
):  # -> tuple[Tensor, Tensor]:
    ...

class AlignTextSelfAttention(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor]: ...

class AlignTextSelfOutput(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor: ...

class AlignTextAttention(nn.Module):
    def __init__(self, config) -> None: ...
    def prune_heads(self, heads):  # -> None:
        ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor]: ...

class AlignTextIntermediate(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class AlignTextOutput(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor: ...

class AlignTextLayer(GradientCheckpointingLayer):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor]: ...
    def feed_forward_chunk(self, attention_output):  # -> Any:
        ...

class AlignTextEncoder(nn.Module):
    def __init__(self, config) -> None: ...
    @can_return_tuple
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor] | BaseModelOutput: ...

class AlignTextPooler(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

@auto_docstring
class AlignPreTrainedModel(PreTrainedModel):
    config: AlignConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...

@auto_docstring(custom_intro=...)
class AlignTextModel(AlignPreTrainedModel):
    config: AlignTextConfig
    _no_split_modules = ...
    def __init__(
        self, config: AlignTextConfig, add_pooling_layer: bool = ...
    ) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        **kwargs,
    ) -> tuple | BaseModelOutputWithPooling: ...

@auto_docstring(custom_intro=...)
class AlignVisionModel(AlignPreTrainedModel):
    config: AlignVisionConfig
    main_input_name = ...
    supports_gradient_checkpointing = ...
    def __init__(self, config: AlignVisionConfig) -> None: ...
    def get_input_embeddings(self) -> nn.Module: ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutputWithPoolingAndNoAttention: ...

@auto_docstring
class AlignModel(AlignPreTrainedModel):
    config: AlignConfig
    def __init__(self, config: AlignConfig) -> None: ...
    @filter_out_non_signature_kwargs()
    @auto_docstring
    def get_text_features(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
    ) -> torch.FloatTensor: ...
    @filter_out_non_signature_kwargs()
    @auto_docstring
    def get_image_features(
        self, pixel_values: torch.FloatTensor
    ) -> torch.FloatTensor: ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        return_loss: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | AlignOutput: ...

__all__ = ["AlignModel", "AlignPreTrainedModel", "AlignTextModel", "AlignVisionModel"]
