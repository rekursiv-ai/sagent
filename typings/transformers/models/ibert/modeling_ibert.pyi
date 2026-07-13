from torch import nn

import torch

from .configuration_ibert import IBertConfig
from ...modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring

"""PyTorch I-BERT model."""
logger = ...

class IBertEmbeddings(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        input_ids=...,
        token_type_ids=...,
        position_ids=...,
        inputs_embeds=...,
        past_key_values_length=...,
    ):  # -> tuple[Any, Any]:
        ...
    def create_position_ids_from_inputs_embeds(self, inputs_embeds):  # -> Tensor:
        ...

class IBertSelfAttention(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states,
        hidden_states_scaling_factor,
        attention_mask=...,
        head_mask=...,
        output_attentions=...,
    ):  # -> tuple[tuple[Any, Any] | tuple[Any], tuple[Any, Any] | tuple[Any]]:
        ...

class IBertSelfOutput(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states,
        hidden_states_scaling_factor,
        input_tensor,
        input_tensor_scaling_factor,
    ):  # -> tuple[Any, Any]:
        ...

class IBertAttention(nn.Module):
    def __init__(self, config) -> None: ...
    def prune_heads(self, heads):  # -> None:
        ...
    def forward(
        self,
        hidden_states,
        hidden_states_scaling_factor,
        attention_mask=...,
        head_mask=...,
        output_attentions=...,
    ):  # -> tuple[Any, Any]:
        ...

class IBertIntermediate(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states, hidden_states_scaling_factor
    ):  # -> tuple[Any, Any]:
        ...

class IBertOutput(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states,
        hidden_states_scaling_factor,
        input_tensor,
        input_tensor_scaling_factor,
    ):  # -> tuple[Any, Any]:
        ...

class IBertLayer(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states,
        hidden_states_scaling_factor,
        attention_mask=...,
        head_mask=...,
        output_attentions=...,
    ):  # -> Any:
        ...
    def feed_forward_chunk(
        self, attention_output, attention_output_scaling_factor
    ):  # -> tuple[Any, Any]:
        ...

class IBertEncoder(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states,
        hidden_states_scaling_factor,
        attention_mask=...,
        head_mask=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
    ):  # -> tuple[Any | tuple[Any, ...] | tuple[()], ...] | BaseModelOutputWithPastAndCrossAttentions:
        ...

class IBertPooler(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

@auto_docstring
class IBertPreTrainedModel(PreTrainedModel):
    config: IBertConfig
    base_model_prefix = ...
    def resize_token_embeddings(self, new_num_tokens=...): ...

@auto_docstring
class IBertModel(IBertPreTrainedModel):
    def __init__(self, config, add_pooling_layer=...) -> None: ...
    def get_input_embeddings(self):  # -> QuantEmbedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
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
    ) -> BaseModelOutputWithPoolingAndCrossAttentions | tuple[torch.FloatTensor]: ...

@auto_docstring
class IBertForMaskedLM(IBertPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    def get_output_embeddings(self):  # -> Linear:
        ...
    def set_output_embeddings(self, new_embeddings):  # -> None:
        ...
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
    ) -> MaskedLMOutput | tuple[torch.FloatTensor]: ...

class IBertLMHead(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, features, **kwargs):  # -> Any:
        ...

@auto_docstring(custom_intro=...)
class IBertForSequenceClassification(IBertPreTrainedModel):
    def __init__(self, config) -> None: ...
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
    ) -> SequenceClassifierOutput | tuple[torch.FloatTensor]: ...

@auto_docstring
class IBertForMultipleChoice(IBertPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> MultipleChoiceModelOutput | tuple[torch.FloatTensor]: ...

@auto_docstring
class IBertForTokenClassification(IBertPreTrainedModel):
    def __init__(self, config) -> None: ...
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
    ) -> TokenClassifierOutput | tuple[torch.FloatTensor]: ...

class IBertClassificationHead(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, features, **kwargs):  # -> Any:
        ...

@auto_docstring
class IBertForQuestionAnswering(IBertPreTrainedModel):
    def __init__(self, config) -> None: ...
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
    ) -> QuestionAnsweringModelOutput | tuple[torch.FloatTensor]: ...

def create_position_ids_from_input_ids(
    input_ids, padding_idx, past_key_values_length=...
): ...

__all__ = [
    "IBertForMaskedLM",
    "IBertForMultipleChoice",
    "IBertForQuestionAnswering",
    "IBertForSequenceClassification",
    "IBertForTokenClassification",
    "IBertModel",
    "IBertPreTrainedModel",
]
