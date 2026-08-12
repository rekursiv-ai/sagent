from typing import Any
from dataclasses import dataclass

from torch import nn

import torch

from .configuration_albert import AlbertConfig
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    MaskedLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring

"""PyTorch ALBERT model."""
logger = ...

def load_tf_weights_in_albert(model, config, tf_checkpoint_path): ...

class AlbertEmbeddings(nn.Module):
    def __init__(self, config: AlbertConfig) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        past_key_values_length: int = ...,
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class AlbertAttention(nn.Module):
    def __init__(self, config: AlbertConfig) -> None: ...
    def prune_heads(self, heads: list[int]) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        output_attentions: bool = ...,
    ) -> tuple[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]: ...

class AlbertSdpaAttention(AlbertAttention):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        output_attentions: bool = ...,
    ) -> tuple[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]: ...

ALBERT_ATTENTION_CLASSES = ...

class AlbertLayer(nn.Module):
    def __init__(self, config: AlbertConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def ff_chunk(self, attention_output: torch.Tensor) -> torch.Tensor: ...

class AlbertLayerGroup(nn.Module):
    def __init__(self, config: AlbertConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
    ) -> tuple[torch.Tensor | tuple[torch.Tensor], ...]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor | tuple[torch.Tensor], ...]: ...

class AlbertTransformer(nn.Module):
    def __init__(self, config: AlbertConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ) -> BaseModelOutput | tuple: ...
    def __call__(self, *args: Any, **kwargs: Any) -> BaseModelOutput | tuple: ...

@auto_docstring
class AlbertPreTrainedModel(PreTrainedModel):
    config: AlbertConfig
    load_tf_weights = ...
    base_model_prefix = ...
    _supports_sdpa = ...

@dataclass
@auto_docstring(
    custom_intro="""
    Output type of [`AlbertForPreTraining`].
    """
)
class AlbertForPreTrainingOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    prediction_logits: torch.FloatTensor | None = ...
    sop_logits: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...

@auto_docstring
class AlbertModel(AlbertPreTrainedModel):
    config: AlbertConfig
    base_model_prefix = ...
    def __init__(self, config: AlbertConfig, add_pooling_layer: bool = ...) -> None: ...
    def get_input_embeddings(self) -> nn.Embedding: ...
    def set_input_embeddings(self, value: nn.Embedding) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> BaseModelOutputWithPooling | tuple: ...

@auto_docstring(custom_intro=...)
class AlbertForPreTraining(AlbertPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config: AlbertConfig) -> None: ...
    def get_output_embeddings(self) -> nn.Linear: ...
    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None: ...
    def get_input_embeddings(self) -> nn.Embedding: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        sentence_order_label: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> AlbertForPreTrainingOutput | tuple: ...

class AlbertMLMHead(nn.Module):
    def __init__(self, config: AlbertConfig) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class AlbertSOPHead(nn.Module):
    def __init__(self, config: AlbertConfig) -> None: ...
    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

@auto_docstring
class AlbertForMaskedLM(AlbertPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    def get_output_embeddings(self) -> nn.Linear: ...
    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None: ...
    def get_input_embeddings(self) -> nn.Embedding: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> MaskedLMOutput | tuple: ...

@auto_docstring(custom_intro=...)
class AlbertForSequenceClassification(AlbertPreTrainedModel):
    def __init__(self, config: AlbertConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> SequenceClassifierOutput | tuple: ...

@auto_docstring
class AlbertForTokenClassification(AlbertPreTrainedModel):
    def __init__(self, config: AlbertConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> TokenClassifierOutput | tuple: ...

@auto_docstring
class AlbertForQuestionAnswering(AlbertPreTrainedModel):
    def __init__(self, config: AlbertConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
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
    ) -> AlbertForPreTrainingOutput | tuple: ...

@auto_docstring
class AlbertForMultipleChoice(AlbertPreTrainedModel):
    def __init__(self, config: AlbertConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> AlbertForPreTrainingOutput | tuple: ...

__all__ = [
    "AlbertForMaskedLM",
    "AlbertForMultipleChoice",
    "AlbertForPreTraining",
    "AlbertForQuestionAnswering",
    "AlbertForSequenceClassification",
    "AlbertForTokenClassification",
    "AlbertModel",
    "AlbertPreTrainedModel",
    "load_tf_weights_in_albert",
]
