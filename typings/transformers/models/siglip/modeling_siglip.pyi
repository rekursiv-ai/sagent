from dataclasses import dataclass
from typing import Any

from torch import nn

import torch

from .configuration_siglip import SiglipConfig, SiglipTextConfig, SiglipVisionConfig
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

"""PyTorch Siglip model."""

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

@dataclass
@auto_docstring(custom_intro=...)
class SiglipVisionModelOutput(ModelOutput):
    image_embeds: torch.FloatTensor | None = ...
    last_hidden_state: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class SiglipTextModelOutput(ModelOutput):
    text_embeds: torch.FloatTensor | None = ...
    last_hidden_state: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...

@dataclass
@auto_docstring
class SiglipOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits_per_image: torch.FloatTensor | None = ...
    logits_per_text: torch.FloatTensor | None = ...
    text_embeds: torch.FloatTensor | None = ...
    image_embeds: torch.FloatTensor | None = ...
    text_model_output: BaseModelOutputWithPooling = ...
    vision_model_output: BaseModelOutputWithPooling = ...
    def to_tuple(self) -> tuple[Any]: ...

class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig) -> None: ...
    def interpolate_pos_encoding(
        self, embeddings: torch.Tensor, height: int, width: int
    ) -> torch.Tensor: ...
    def forward(
        self, pixel_values: torch.FloatTensor, interpolate_pos_encoding=...
    ) -> torch.Tensor: ...

class SiglipTextEmbeddings(nn.Module):
    def __init__(self, config: SiglipTextConfig) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
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
    **kwargs,
):  # -> tuple[Tensor, Tensor]:
    ...

class SiglipAttention(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class SiglipMLP(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class SiglipEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: SiglipVisionConfig | SiglipTextConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.FloatTensor: ...

@auto_docstring
class SiglipPreTrainedModel(PreTrainedModel):
    config: SiglipConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _supports_flex_attn = ...
    _supports_attention_backend = ...
    _can_record_outputs = ...

class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        inputs_embeds,
        attention_mask: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput: ...

class SiglipTextTransformer(nn.Module):
    def __init__(self, config: SiglipTextConfig) -> None: ...
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
class SiglipTextModel(SiglipPreTrainedModel):
    config: SiglipTextConfig
    def __init__(self, config: SiglipTextConfig) -> None: ...
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

class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig) -> None: ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values,
        interpolate_pos_encoding: bool | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling: ...

class SiglipMultiheadAttentionPoolingHead(nn.Module):
    def __init__(self, config: SiglipVisionConfig) -> None: ...
    def forward(self, hidden_state):  # -> Any:
        ...

@auto_docstring(custom_intro=...)
class SiglipVisionModel(SiglipPreTrainedModel):
    config: SiglipVisionConfig
    main_input_name = ...
    def __init__(self, config: SiglipVisionConfig) -> None: ...
    def get_input_embeddings(self) -> nn.Module: ...
    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        pixel_values,
        interpolate_pos_encoding: bool = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling: ...

@auto_docstring
class SiglipModel(SiglipPreTrainedModel):
    config: SiglipConfig
    def __init__(self, config: SiglipConfig) -> None: ...
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
        pixel_values: torch.FloatTensor,
        interpolate_pos_encoding: bool = ...,
        **kwargs: Unpack[TransformersKwargs],
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
        interpolate_pos_encoding: bool = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> SiglipOutput: ...

@auto_docstring(custom_intro=...)
class SiglipForImageClassification(SiglipPreTrainedModel):
    main_input_name = ...
    def __init__(self, config: SiglipConfig) -> None: ...
    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        interpolate_pos_encoding: bool = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> ImageClassifierOutput: ...

__all__ = [
    "SiglipForImageClassification",
    "SiglipModel",
    "SiglipPreTrainedModel",
    "SiglipTextModel",
    "SiglipVisionModel",
]
