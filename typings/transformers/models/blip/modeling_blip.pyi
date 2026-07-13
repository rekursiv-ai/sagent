from dataclasses import dataclass
from typing import Any

from torch import nn

import torch

from .configuration_blip import BlipConfig, BlipTextConfig, BlipVisionConfig
from ...generation import GenerationMixin
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import check_model_inputs

"""PyTorch BLIP model."""
logger = ...

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor: ...
def blip_loss(similarity: torch.Tensor) -> torch.Tensor: ...

@dataclass
@auto_docstring(custom_intro=...)
class BlipForConditionalGenerationModelOutput(ModelOutput):
    loss: tuple[torch.FloatTensor] | None = ...
    logits: tuple[torch.FloatTensor] | None = ...
    image_embeds: torch.FloatTensor | None = ...
    last_hidden_state: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...
    @property
    def decoder_logits(self):  # -> tuple[FloatTensor] | None:
        ...

@dataclass
@auto_docstring(custom_intro=...)
class BlipTextVisionModelOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    image_embeds: torch.FloatTensor | None = ...
    last_hidden_state: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class BlipImageTextMatchingModelOutput(ModelOutput):
    itm_score: torch.FloatTensor | None = ...
    loss: torch.FloatTensor | None = ...
    image_embeds: torch.FloatTensor | None = ...
    last_hidden_state: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    vision_pooler_output: torch.FloatTensor | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...
    question_embeds: tuple[torch.FloatTensor] | None = ...

@dataclass
@auto_docstring
class BlipOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits_per_image: torch.FloatTensor | None = ...
    logits_per_text: torch.FloatTensor | None = ...
    text_embeds: torch.FloatTensor | None = ...
    image_embeds: torch.FloatTensor | None = ...
    text_model_output: BaseModelOutputWithPooling = ...
    vision_model_output: BaseModelOutputWithPooling = ...
    def to_tuple(self) -> tuple[Any]: ...

class BlipVisionEmbeddings(nn.Module):
    def __init__(self, config: BlipVisionConfig) -> None: ...
    def interpolate_pos_encoding(
        self, embeddings: torch.Tensor, height: int, width: int
    ) -> torch.Tensor: ...
    def forward(
        self, pixel_values: torch.FloatTensor, interpolate_pos_encoding: bool = ...
    ) -> torch.Tensor: ...

class BlipTextEmbeddings(nn.Module):
    def __init__(self, config: BlipTextConfig) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
    ) -> torch.Tensor: ...

class BlipAttention(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

class BlipMLP(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class BlipEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: BlipConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.FloatTensor: ...

@auto_docstring
class BlipPreTrainedModel(PreTrainedModel):
    config: BlipConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _skip_keys_device_placement = ...

class BlipEncoder(nn.Module):
    def __init__(self, config: BlipConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        inputs_embeds,
        attention_mask: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutput: ...

class BlipVisionModel(BlipPreTrainedModel):
    main_input_name = ...
    config: BlipVisionConfig
    _can_record_outputs = ...
    def __init__(self, config: BlipVisionConfig) -> None: ...
    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        interpolate_pos_encoding: bool = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling: ...
    def get_input_embeddings(self):  # -> BlipVisionEmbeddings:
        ...

@auto_docstring(custom_intro=...)
class BlipModel(BlipPreTrainedModel):
    config: BlipConfig
    def __init__(self, config: BlipConfig) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @auto_docstring
    def get_text_features(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
    ) -> torch.FloatTensor: ...
    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        interpolate_pos_encoding: bool = ...,
    ) -> torch.FloatTensor: ...
    @auto_docstring
    def get_multimodal_features(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
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
        interpolate_pos_encoding: bool = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BlipOutput: ...

@auto_docstring(custom_intro=...)
class BlipForConditionalGeneration(BlipPreTrainedModel, GenerationMixin):
    config: BlipConfig
    _tied_weights_keys = ...
    main_input_name = ...
    def __init__(self, config: BlipConfig) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        interpolate_pos_encoding: bool = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BlipForConditionalGenerationModelOutput: ...
    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        interpolate_pos_encoding: bool = ...,
        **generate_kwargs,
    ) -> torch.LongTensor: ...

@auto_docstring(custom_intro=...)
class BlipForQuestionAnswering(BlipPreTrainedModel, GenerationMixin):
    config: BlipConfig
    _tied_weights_keys = ...
    def __init__(self, config: BlipConfig) -> None: ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        decoder_input_ids: torch.LongTensor | None = ...,
        decoder_attention_mask: torch.LongTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        interpolate_pos_encoding: bool = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BlipTextVisionModelOutput: ...
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        attention_mask: torch.LongTensor | None = ...,
        interpolate_pos_encoding: bool = ...,
        **generate_kwargs,
    ) -> torch.LongTensor: ...

@auto_docstring(custom_intro=...)
class BlipForImageTextRetrieval(BlipPreTrainedModel):
    config: BlipConfig
    def __init__(self, config: BlipConfig) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        use_itm_head: bool | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        interpolate_pos_encoding: bool = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BlipTextVisionModelOutput: ...

__all__ = [
    "BlipForConditionalGeneration",
    "BlipForImageTextRetrieval",
    "BlipForQuestionAnswering",
    "BlipModel",
    "BlipPreTrainedModel",
    "BlipTextModel",
    "BlipVisionModel",
]
