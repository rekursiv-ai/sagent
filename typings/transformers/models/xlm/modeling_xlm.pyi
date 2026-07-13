from dataclasses import dataclass

from torch import nn

import torch

from .configuration_xlm import XLMConfig
from ...generation import GenerationMixin
from ...modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring

"""
PyTorch XLM model.
"""
logger = ...

def create_sinusoidal_embeddings(n_pos, dim, out):  # -> None:
    ...
def get_masks(slen, lengths, causal, padding_mask=...):  # -> tuple[Any, Tensor | Any]:
    ...

@dataclass
@auto_docstring(custom_intro=...)
class XLMSquadHeadOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    start_top_log_probs: torch.FloatTensor | None = ...
    start_top_index: torch.LongTensor | None = ...
    end_top_log_probs: torch.FloatTensor | None = ...
    end_top_index: torch.LongTensor | None = ...
    cls_logits: torch.FloatTensor | None = ...

class XLMPoolerStartLogits(nn.Module):
    def __init__(self, config: XLMConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        p_mask: torch.FloatTensor | None = ...,
    ) -> torch.FloatTensor: ...

class XLMPoolerEndLogits(nn.Module):
    def __init__(self, config: XLMConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        start_states: torch.FloatTensor | None = ...,
        start_positions: torch.LongTensor | None = ...,
        p_mask: torch.FloatTensor | None = ...,
    ) -> torch.FloatTensor: ...

class XLMPoolerAnswerClass(nn.Module):
    def __init__(self, config: XLMConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        start_states: torch.FloatTensor | None = ...,
        start_positions: torch.LongTensor | None = ...,
        cls_index: torch.LongTensor | None = ...,
    ) -> torch.FloatTensor: ...

class XLMSQuADHead(nn.Module):
    def __init__(self, config: XLMConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        start_positions: torch.LongTensor | None = ...,
        end_positions: torch.LongTensor | None = ...,
        cls_index: torch.LongTensor | None = ...,
        is_impossible: torch.LongTensor | None = ...,
        p_mask: torch.FloatTensor | None = ...,
        return_dict: bool = ...,
    ) -> XLMSquadHeadOutput | tuple[torch.FloatTensor]: ...

class XLMSequenceSummary(nn.Module):
    def __init__(self, config: XLMConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        cls_index: torch.LongTensor | None = ...,
    ) -> torch.FloatTensor: ...

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, dim, config, layer_idx: int = ...) -> None: ...
    def prune_heads(self, heads):  # -> None:
        ...
    def forward(
        self,
        input,
        mask,
        kv=...,
        cache=...,
        head_mask=...,
        output_attentions=...,
        cache_position=...,
    ):  # -> tuple[Any, Any | Tensor] | tuple[Any]:
        ...

class TransformerFFN(nn.Module):
    def __init__(self, in_dim, dim_hidden, out_dim, config) -> None: ...
    def forward(self, input):  # -> Tensor:
        ...
    def ff_chunk(self, input):  # -> Tensor:
        ...

@auto_docstring
class XLMPreTrainedModel(PreTrainedModel):
    config: XLMConfig
    load_tf_weights = ...
    base_model_prefix = ...
    def __init__(self, *inputs, **kwargs) -> None: ...
    @property
    def dummy_inputs(self):  # -> dict[str, Tensor | None]:
        ...

@dataclass
@auto_docstring(custom_intro=...)
class XLMForQuestionAnsweringOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    start_top_log_probs: torch.FloatTensor | None = ...
    start_top_index: torch.LongTensor | None = ...
    end_top_log_probs: torch.FloatTensor | None = ...
    end_top_index: torch.LongTensor | None = ...
    cls_logits: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...

@auto_docstring
class XLMModel(XLMPreTrainedModel):
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, new_embeddings):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        langs: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        lengths: torch.Tensor | None = ...,
        cache: dict[str, torch.Tensor] | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
        **kwargs,
    ) -> tuple | BaseModelOutput: ...

class XLMPredLayer(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, x, y=...
    ):  # -> tuple[Tensor, Any] | tuple[Any] | tuple[Any, Any | Tensor] | tuple[Any | Tensor]:
        ...

@auto_docstring(custom_intro=...)
class XLMWithLMHeadModel(XLMPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    def get_output_embeddings(self):  # -> Linear | AdaptiveLogSoftmaxWithLoss:
        ...
    def set_output_embeddings(self, new_embeddings):  # -> None:
        ...
    def prepare_inputs_for_generation(
        self, input_ids, **kwargs
    ):  # -> dict[str, Tensor | None]:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        langs: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        lengths: torch.Tensor | None = ...,
        cache: dict[str, torch.Tensor] | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
        **kwargs,
    ) -> tuple | MaskedLMOutput: ...

@auto_docstring(custom_intro=...)
class XLMForSequenceClassification(XLMPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        langs: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        lengths: torch.Tensor | None = ...,
        cache: dict[str, torch.Tensor] | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | SequenceClassifierOutput: ...

@auto_docstring(custom_intro=...)
class XLMForQuestionAnsweringSimple(XLMPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        langs: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        lengths: torch.Tensor | None = ...,
        cache: dict[str, torch.Tensor] | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        start_positions: torch.Tensor | None = ...,
        end_positions: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | QuestionAnsweringModelOutput: ...

@auto_docstring
class XLMForQuestionAnswering(XLMPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        langs: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        lengths: torch.Tensor | None = ...,
        cache: dict[str, torch.Tensor] | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        start_positions: torch.Tensor | None = ...,
        end_positions: torch.Tensor | None = ...,
        is_impossible: torch.Tensor | None = ...,
        cls_index: torch.Tensor | None = ...,
        p_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | XLMForQuestionAnsweringOutput: ...

@auto_docstring
class XLMForTokenClassification(XLMPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        langs: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        lengths: torch.Tensor | None = ...,
        cache: dict[str, torch.Tensor] | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | TokenClassifierOutput: ...

@auto_docstring
class XLMForMultipleChoice(XLMPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        langs: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        lengths: torch.Tensor | None = ...,
        cache: dict[str, torch.Tensor] | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | MultipleChoiceModelOutput: ...

__all__ = [
    "XLMForMultipleChoice",
    "XLMForQuestionAnswering",
    "XLMForQuestionAnsweringSimple",
    "XLMForSequenceClassification",
    "XLMForTokenClassification",
    "XLMModel",
    "XLMPreTrainedModel",
    "XLMWithLMHeadModel",
]
