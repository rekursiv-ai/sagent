from dataclasses import dataclass
from typing import Any

from torch import nn

import torch

from .configuration_metaclip_2 import (
    MetaClip2Config,
    MetaClip2TextConfig,
    MetaClip2VisionConfig,
)
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    ImageClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import (
    ModelOutput,
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    filter_out_non_signature_kwargs,
)
from ...utils.generic import check_model_inputs

class MetaClip2TextEmbeddings(nn.Module):
    def __init__(self, config: MetaClip2TextConfig) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
    ) -> torch.Tensor: ...

class MetaClip2VisionEmbeddings(nn.Module):
    def __init__(self, config: MetaClip2VisionConfig) -> None: ...
    def interpolate_pos_encoding(
        self, embeddings: torch.Tensor, height: int, width: int
    ) -> torch.Tensor: ...
    def forward(
        self, pixel_values: torch.FloatTensor, interpolate_pos_encoding=...
    ) -> torch.Tensor: ...

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = ...,
    output_attentions: bool = ...,
    **kwargs,
):  # -> tuple[Tensor, Tensor | None]:
    ...

class MetaClip2Attention(nn.Module):
    def __init__(self, config: MetaClip2VisionConfig | MetaClip2TextConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        causal_attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class MetaClip2MLP(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

@auto_docstring
class MetaClip2PreTrainedModel(PreTrainedModel):
    config: MetaClip2Config
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _supports_sdpa = ...
    _supports_flash_attn = ...
    _supports_flex_attn = ...
    _supports_attention_backend = ...

class MetaClip2EncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: MetaClip2VisionConfig | MetaClip2TextConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.FloatTensor]: ...

class MetaClip2Encoder(nn.Module):
    def __init__(self, config: MetaClip2Config) -> None: ...
    def forward(
        self,
        inputs_embeds,
        attention_mask: torch.Tensor | None = ...,
        causal_attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
    ) -> BaseModelOutput: ...

class MetaClip2TextTransformer(nn.Module):
    def __init__(self, config: MetaClip2TextConfig) -> None: ...
    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        use_cache: bool | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling: ...

@auto_docstring(custom_intro=...)
class MetaClip2TextModel(MetaClip2PreTrainedModel):
    config: MetaClip2TextConfig
    _no_split_modules = ...
    _supports_flash_attn = ...
    def __init__(self, config: MetaClip2TextConfig) -> None: ...
    def get_input_embeddings(self) -> nn.Module: ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
    ) -> BaseModelOutputWithPooling: ...

@dataclass
@auto_docstring(custom_intro=...)
class MetaClip2TextModelOutput(ModelOutput):
    text_embeds: torch.FloatTensor | None = ...
    last_hidden_state: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...

@auto_docstring
class MetaClip2TextModelWithProjection(MetaClip2PreTrainedModel):
    config: MetaClip2TextConfig
    _supports_flash_attn = ...
    _no_split_modules = ...
    def __init__(self, config: MetaClip2TextConfig) -> None: ...
    def get_input_embeddings(self) -> nn.Module: ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
    ) -> MetaClip2TextModelOutput: ...

@dataclass
@auto_docstring
class MetaClip2Output(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits_per_image: torch.FloatTensor | None = ...
    logits_per_text: torch.FloatTensor | None = ...
    text_embeds: torch.FloatTensor | None = ...
    image_embeds: torch.FloatTensor | None = ...
    text_model_output: BaseModelOutputWithPooling = ...
    vision_model_output: BaseModelOutputWithPooling = ...
    def to_tuple(self) -> tuple[Any]: ...

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor: ...
def metaclip_2_loss(similarity: torch.Tensor) -> torch.Tensor: ...

@auto_docstring
class MetaClip2Model(MetaClip2PreTrainedModel):
    config: MetaClip2Config
    _no_split_modules = ...
    _supports_flash_attn = ...
    def __init__(self, config: MetaClip2Config) -> None: ...
    @filter_out_non_signature_kwargs()
    @auto_docstring
    def get_text_features(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
    ) -> torch.FloatTensor: ...
    @filter_out_non_signature_kwargs()
    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        interpolate_pos_encoding: bool = ...,
    ) -> torch.FloatTensor: ...
    @can_return_tuple
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
    ) -> MetaClip2Output: ...

class MetaClip2VisionTransformer(nn.Module):
    def __init__(self, config: MetaClip2VisionConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        interpolate_pos_encoding: bool | None = ...,
    ) -> BaseModelOutputWithPooling: ...

@auto_docstring(custom_intro=...)
class MetaClip2VisionModel(MetaClip2PreTrainedModel):
    config: MetaClip2VisionConfig
    main_input_name = ...
    _no_split_modules = ...
    def __init__(self, config: MetaClip2VisionConfig) -> None: ...
    def get_input_embeddings(self) -> nn.Module: ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        interpolate_pos_encoding: bool = ...,
    ) -> BaseModelOutputWithPooling: ...

@dataclass
@auto_docstring(custom_intro=...)
class MetaClip2VisionModelOutput(ModelOutput):
    image_embeds: torch.FloatTensor | None = ...
    last_hidden_state: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...

@auto_docstring
class MetaClip2VisionModelWithProjection(MetaClip2PreTrainedModel):
    config: MetaClip2VisionConfig
    main_input_name = ...
    def __init__(self, config: MetaClip2VisionConfig) -> None: ...
    def get_input_embeddings(self) -> nn.Module: ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        interpolate_pos_encoding: bool = ...,
    ) -> MetaClip2VisionModelOutput: ...

@auto_docstring(custom_intro=...)
class MetaClip2ForImageClassification(MetaClip2PreTrainedModel):
    main_input_name = ...
    def __init__(self, config: MetaClip2Config) -> None: ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
    ) -> ImageClassifierOutput: ...

__all__ = [
    "MetaClip2ForImageClassification",
    "MetaClip2Model",
    "MetaClip2PreTrainedModel",
    "MetaClip2TextModel",
    "MetaClip2TextModelWithProjection",
    "MetaClip2VisionModel",
    "MetaClip2VisionModelWithProjection",
]
