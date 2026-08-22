from typing import Any

from torch import nn

import torch

from .configuration_layoutlm import LayoutLMConfig
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    MaskedLMOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring, can_return_tuple

"""PyTorch LayoutLM model."""
logger = ...
LayoutLMLayerNorm = nn.LayerNorm

class LayoutLMEmbeddings(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        input_ids=...,
        bbox=...,
        token_type_ids=...,
        position_ids=...,
        inputs_embeds=...,
    ):  # -> Any:
        ...

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

class LayoutLMSelfAttention(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor]: ...

class LayoutLMSelfOutput(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class LayoutLMAttention(nn.Module):
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
    def __call__(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor]: ...

class LayoutLMIntermediate(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class LayoutLMOutput(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class LayoutLMLayer(GradientCheckpointingLayer):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor]: ...
    def feed_forward_chunk(self, attention_output):  # -> Any:
        ...

class LayoutLMEncoder(nn.Module):
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

class LayoutLMPooler(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class LayoutLMPredictionHeadTransform(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class LayoutLMLMPredictionHead(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class LayoutLMOnlyMLMHead(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

@auto_docstring
class LayoutLMPreTrainedModel(PreTrainedModel):
    config: LayoutLMConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...

@auto_docstring
class LayoutLMModel(LayoutLMPreTrainedModel):
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        bbox: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutputWithPooling: ...

@auto_docstring
class LayoutLMForMaskedLM(LayoutLMPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def get_output_embeddings(self):  # -> Linear:
        ...
    def set_output_embeddings(self, new_embeddings):  # -> None:
        ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        bbox: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | MaskedLMOutput: ...

@auto_docstring(custom_intro=...)
class LayoutLMForSequenceClassification(LayoutLMPreTrainedModel):
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        bbox: torch.LongTensor | None = ...,
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
class LayoutLMForTokenClassification(LayoutLMPreTrainedModel):
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        bbox: torch.LongTensor | None = ...,
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
class LayoutLMForQuestionAnswering(LayoutLMPreTrainedModel):
    def __init__(self, config, has_visual_segment_embedding=...) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        bbox: torch.LongTensor | None = ...,
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
    "LayoutLMForMaskedLM",
    "LayoutLMForQuestionAnswering",
    "LayoutLMForSequenceClassification",
    "LayoutLMForTokenClassification",
    "LayoutLMModel",
    "LayoutLMPreTrainedModel",
]
