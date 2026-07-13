from dataclasses import dataclass
from typing import Any

from torch import nn

import torch

from .configuration_siglip2 import Siglip2Config, Siglip2TextConfig, Siglip2VisionConfig
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

@dataclass
@auto_docstring(custom_intro=...)
class Siglip2VisionOutput(ModelOutput):
    image_embeds: torch.FloatTensor | None = ...
    last_hidden_state: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class Siglip2TextOutput(ModelOutput):
    text_embeds: torch.FloatTensor | None = ...
    last_hidden_state: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...

@dataclass
@auto_docstring
class Siglip2Output(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits_per_image: torch.FloatTensor | None = ...
    logits_per_text: torch.FloatTensor | None = ...
    text_embeds: torch.FloatTensor | None = ...
    image_embeds: torch.FloatTensor | None = ...
    text_model_output: BaseModelOutputWithPooling = ...
    vision_model_output: BaseModelOutputWithPooling = ...
    def to_tuple(self) -> tuple[Any]: ...

class Siglip2VisionEmbeddings(nn.Module):
    def __init__(self, config: Siglip2VisionConfig) -> None: ...
    @staticmethod
    def resize_positional_embeddings(
        positional_embeddings: torch.Tensor,
        spatial_shapes: torch.LongTensor,
        max_length: int,
    ) -> torch.Tensor: ...
    def forward(
        self, pixel_values: torch.FloatTensor, spatial_shapes: torch.LongTensor
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

class Siglip2Attention(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class Siglip2MLP(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class Siglip2EncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Siglip2VisionConfig | Siglip2TextConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.FloatTensor: ...

class Siglip2Encoder(nn.Module):
    def __init__(self, config: Siglip2Config) -> None: ...
    @auto_docstring
    def forward(
        self,
        inputs_embeds,
        attention_mask: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput: ...

class Siglip2VisionTransformer(nn.Module):
    def __init__(self, config: Siglip2VisionConfig) -> None: ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        attention_mask: torch.Tensor,
        spatial_shapes: torch.LongTensor,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
    ) -> BaseModelOutputWithPooling: ...

def trunc_normal_tf_(
    tensor: torch.Tensor,
    mean: float = ...,
    std: float = ...,
    a: float = ...,
    b: float = ...,
) -> torch.Tensor: ...
def variance_scaling_(tensor, scale=..., mode=..., distribution=...):  # -> None:
    ...
def lecun_normal_(tensor):  # -> None:
    ...
def default_flax_embed_init(tensor):  # -> None:
    ...

@auto_docstring
class Siglip2PreTrainedModel(PreTrainedModel):
    config: Siglip2Config
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _supports_flex_attn = ...
    _supports_attention_backend = ...
    _can_record_outputs = ...

class Siglip2TextEmbeddings(nn.Module):
    def __init__(self, config: Siglip2TextConfig) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
    ) -> torch.Tensor: ...

class Siglip2TextTransformer(nn.Module):
    def __init__(self, config: Siglip2TextConfig) -> None: ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling: ...

@auto_docstring(custom_intro=...)
class Siglip2TextModel(Siglip2PreTrainedModel):
    config: Siglip2TextConfig
    def __init__(self, config: Siglip2TextConfig) -> None: ...
    def get_input_embeddings(self) -> nn.Module: ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling: ...

class Siglip2MultiheadAttentionPoolingHead(nn.Module):
    def __init__(self, config: Siglip2VisionConfig) -> None: ...
    def forward(
        self, hidden_state: torch.Tensor, attention_mask: torch.Tensor | None = ...
    ) -> torch.Tensor: ...

@auto_docstring(custom_intro=...)
class Siglip2VisionModel(Siglip2PreTrainedModel):
    config: Siglip2VisionConfig
    main_input_name = ...
    def __init__(self, config: Siglip2VisionConfig) -> None: ...
    def get_input_embeddings(self) -> nn.Module: ...
    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_attention_mask: torch.Tensor,
        spatial_shapes: torch.LongTensor,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
    ) -> BaseModelOutputWithPooling: ...

@auto_docstring
class Siglip2Model(Siglip2PreTrainedModel):
    config: Siglip2Config
    def __init__(self, config: Siglip2Config) -> None: ...
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
        self,
        pixel_values: torch.FloatTensor | None = ...,
        pixel_attention_mask: torch.Tensor | None = ...,
        spatial_shapes: torch.LongTensor | None = ...,
    ) -> torch.FloatTensor: ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        pixel_attention_mask: torch.Tensor | None = ...,
        spatial_shapes: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        return_loss: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
    ) -> Siglip2Output: ...

@auto_docstring(custom_intro=...)
class Siglip2ForImageClassification(Siglip2PreTrainedModel):
    main_input_name = ...
    def __init__(self, config: Siglip2Config) -> None: ...
    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor | None = ...,
        pixel_attention_mask: torch.Tensor | None = ...,
        spatial_shapes: torch.LongTensor | None = ...,
        labels: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
    ) -> ImageClassifierOutput: ...

__all__ = [
    "Siglip2ForImageClassification",
    "Siglip2Model",
    "Siglip2PreTrainedModel",
    "Siglip2TextModel",
    "Siglip2VisionModel",
]
