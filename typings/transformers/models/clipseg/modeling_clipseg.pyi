from dataclasses import dataclass
from typing import Any

from torch import nn

import torch

from .configuration_clipseg import CLIPSegConfig, CLIPSegTextConfig, CLIPSegVisionConfig
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    auto_docstring,
    can_return_tuple,
    filter_out_non_signature_kwargs,
)

"""PyTorch CLIPSeg model."""
logger = ...

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor: ...
def clipseg_loss(similarity: torch.Tensor) -> torch.Tensor: ...

@dataclass
@auto_docstring
class CLIPSegOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits_per_image: torch.FloatTensor | None = ...
    logits_per_text: torch.FloatTensor | None = ...
    text_embeds: torch.FloatTensor | None = ...
    image_embeds: torch.FloatTensor | None = ...
    text_model_output: BaseModelOutputWithPooling = ...
    vision_model_output: BaseModelOutputWithPooling = ...
    def to_tuple(self) -> tuple[Any]: ...

@dataclass
@auto_docstring
class CLIPSegDecoderOutput(ModelOutput):
    logits: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...

@dataclass
@auto_docstring
class CLIPSegImageSegmentationOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    conditional_embeddings: torch.FloatTensor | None = ...
    pooled_output: torch.FloatTensor | None = ...
    vision_model_output: BaseModelOutputWithPooling = ...
    decoder_output: CLIPSegDecoderOutput = ...
    def to_tuple(self) -> tuple[Any]: ...

class CLIPSegVisionEmbeddings(nn.Module):
    def __init__(self, config: CLIPSegVisionConfig) -> None: ...
    def interpolate_pos_encoding(
        self, embeddings: torch.Tensor, height: int, width: int
    ) -> torch.Tensor: ...
    def forward(
        self, pixel_values: torch.FloatTensor, interpolate_pos_encoding=...
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class CLIPSegTextEmbeddings(nn.Module):
    def __init__(self, config: CLIPSegTextConfig) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

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

class CLIPSegAttention(nn.Module):
    def __init__(self, config: CLIPSegVisionConfig | CLIPSegTextConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        causal_attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class CLIPSegMLP(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class CLIPSegEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: CLIPSegConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.FloatTensor]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple[torch.FloatTensor]: ...

@auto_docstring
class CLIPSegPreTrainedModel(PreTrainedModel):
    config: CLIPSegConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...

class CLIPSegEncoder(nn.Module):
    def __init__(self, config: CLIPSegConfig) -> None: ...
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

class CLIPSegTextTransformer(nn.Module):
    def __init__(self, config: CLIPSegTextConfig) -> None: ...
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

class CLIPSegTextModel(CLIPSegPreTrainedModel):
    config: CLIPSegTextConfig
    _no_split_modules = ...
    def __init__(self, config: CLIPSegTextConfig) -> None: ...
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

class CLIPSegVisionTransformer(nn.Module):
    def __init__(self, config: CLIPSegVisionConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        interpolate_pos_encoding: bool | None = ...,
    ) -> tuple | BaseModelOutputWithPooling: ...

class CLIPSegVisionModel(CLIPSegPreTrainedModel):
    config: CLIPSegVisionConfig
    main_input_name = ...
    def __init__(self, config: CLIPSegVisionConfig) -> None: ...
    def get_input_embeddings(self) -> nn.Module: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        interpolate_pos_encoding: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutputWithPooling: ...

@auto_docstring
class CLIPSegModel(CLIPSegPreTrainedModel):
    config: CLIPSegConfig
    def __init__(self, config: CLIPSegConfig) -> None: ...
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
    def get_image_features(
        self, pixel_values: torch.FloatTensor, interpolate_pos_encoding: bool = ...
    ) -> torch.FloatTensor: ...
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
        interpolate_pos_encoding: bool = ...,
        return_dict: bool | None = ...,
    ) -> tuple | CLIPSegOutput: ...

class CLIPSegDecoderLayer(nn.Module):
    def __init__(self, config: CLIPSegConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.FloatTensor]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple[torch.FloatTensor]: ...

class CLIPSegDecoder(CLIPSegPreTrainedModel):
    def __init__(self, config: CLIPSegConfig) -> None: ...
    def forward(
        self,
        hidden_states: tuple[torch.Tensor],
        conditional_embeddings: torch.Tensor,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ):  # -> tuple[Any | tuple[()] | tuple[Any, ...], ...] | CLIPSegDecoderOutput:
        ...

@auto_docstring(custom_intro=...)
class CLIPSegForImageSegmentation(CLIPSegPreTrainedModel):
    config: CLIPSegConfig
    def __init__(self, config: CLIPSegConfig) -> None: ...
    def get_conditional_embeddings(
        self,
        batch_size: int | None = ...,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        conditional_pixel_values: torch.Tensor | None = ...,
    ):  # -> FloatTensor:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.FloatTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        conditional_pixel_values: torch.FloatTensor | None = ...,
        conditional_embeddings: torch.FloatTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        interpolate_pos_encoding: bool = ...,
        return_dict: bool | None = ...,
    ) -> tuple | CLIPSegOutput: ...

__all__ = [
    "CLIPSegForImageSegmentation",
    "CLIPSegModel",
    "CLIPSegPreTrainedModel",
    "CLIPSegTextModel",
    "CLIPSegVisionModel",
]
