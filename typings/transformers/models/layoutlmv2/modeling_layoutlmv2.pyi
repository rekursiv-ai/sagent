from typing import Any
from torch import nn

import torch

from .configuration_layoutlmv2 import LayoutLMv2Config
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutputWithPooling,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring

"""PyTorch LayoutLMv2 model."""
logger = ...

class LayoutLMv2Embeddings(nn.Module):
    def __init__(self, config) -> None: ...

class LayoutLMv2SelfAttention(nn.Module):
    def __init__(self, config) -> None: ...
    def compute_qkv(
        self, hidden_states
    ):  # -> tuple[Tensor | Any, Tensor | Any, Tensor | Any]:
        ...
    def forward(
        self,
        hidden_states,
        attention_mask=...,
        head_mask=...,
        output_attentions=...,
        rel_pos=...,
        rel_2d_pos=...,
    ):  # -> tuple[Tensor, Any] | tuple[Tensor]:
        ...

class LayoutLMv2Attention(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states,
        attention_mask=...,
        head_mask=...,
        output_attentions=...,
        rel_pos=...,
        rel_2d_pos=...,
    ):  # -> Any:
        ...

class LayoutLMv2SelfOutput(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states, input_tensor):  # -> Any:
        ...

class LayoutLMv2Intermediate(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class LayoutLMv2Output(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class LayoutLMv2Layer(GradientCheckpointingLayer):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states,
        attention_mask=...,
        head_mask=...,
        output_attentions=...,
        rel_pos=...,
        rel_2d_pos=...,
    ):  # -> Any:
        ...
    def feed_forward_chunk(self, attention_output):  # -> Any:
        ...

def relative_position_bucket(
    relative_position, bidirectional=..., num_buckets=..., max_distance=...
):  # -> Tensor:
    ...

class LayoutLMv2Encoder(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states,
        attention_mask=...,
        head_mask=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
        bbox=...,
        position_ids=...,
    ):  # -> tuple[Any | tuple[Any, ...] | tuple[()], ...] | BaseModelOutput:
        ...

@auto_docstring
class LayoutLMv2PreTrainedModel(PreTrainedModel):
    config: LayoutLMv2Config
    base_model_prefix = ...

def my_convert_sync_batchnorm(module, process_group=...):  # -> SyncBatchNorm:
    ...

class LayoutLMv2VisualBackbone(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, images):  # -> Any:
        ...
    def synchronize_batch_norm(self):  # -> None:
        ...

class LayoutLMv2Pooler(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

@auto_docstring
class LayoutLMv2Model(LayoutLMv2PreTrainedModel):
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        bbox: torch.LongTensor | None = ...,
        image: torch.FloatTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutputWithPooling: ...

@auto_docstring(custom_intro=...)
class LayoutLMv2ForSequenceClassification(LayoutLMv2PreTrainedModel):
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        bbox: torch.LongTensor | None = ...,
        image: torch.FloatTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | SequenceClassifierOutput: ...

@auto_docstring(custom_intro=...)
class LayoutLMv2ForTokenClassification(LayoutLMv2PreTrainedModel):
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        bbox: torch.LongTensor | None = ...,
        image: torch.FloatTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | TokenClassifierOutput: ...

@auto_docstring
class LayoutLMv2ForQuestionAnswering(LayoutLMv2PreTrainedModel):
    def __init__(self, config, has_visual_segment_embedding=...) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        bbox: torch.LongTensor | None = ...,
        image: torch.FloatTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        start_positions: torch.LongTensor | None = ...,
        end_positions: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | QuestionAnsweringModelOutput: ...

__all__ = [
    "LayoutLMv2ForQuestionAnswering",
    "LayoutLMv2ForSequenceClassification",
    "LayoutLMv2ForTokenClassification",
    "LayoutLMv2Layer",
    "LayoutLMv2Model",
    "LayoutLMv2PreTrainedModel",
]
