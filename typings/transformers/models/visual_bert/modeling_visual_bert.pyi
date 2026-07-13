from dataclasses import dataclass

from torch import nn

import torch

from .configuration_visual_bert import VisualBertConfig
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutputWithPooling,
    MultipleChoiceModelOutput,
    SequenceClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring

"""PyTorch VisualBERT model."""
logger = ...

class VisualBertEmbeddings(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        input_ids=...,
        token_type_ids=...,
        position_ids=...,
        inputs_embeds=...,
        visual_embeds=...,
        visual_token_type_ids=...,
        image_text_alignment=...,
    ):  # -> Any:
        ...

class VisualBertSelfAttention(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states, attention_mask=..., head_mask=..., output_attentions=...
    ):  # -> tuple[Tensor, Any] | tuple[Tensor]:
        ...

class VisualBertSelfOutput(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor: ...

class VisualBertAttention(nn.Module):
    def __init__(self, config) -> None: ...
    def prune_heads(self, heads):  # -> None:
        ...
    def forward(
        self, hidden_states, attention_mask=..., head_mask=..., output_attentions=...
    ):  # -> Any:
        ...

class VisualBertIntermediate(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class VisualBertOutput(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor: ...

class VisualBertLayer(GradientCheckpointingLayer):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states, attention_mask=..., head_mask=..., output_attentions=...
    ):  # -> Any:
        ...
    def feed_forward_chunk(self, attention_output):  # -> Any:
        ...

class VisualBertEncoder(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states,
        attention_mask=...,
        head_mask=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
    ):  # -> tuple[Any | tuple[Any, ...] | tuple[()], ...] | BaseModelOutput:
        ...

class VisualBertPooler(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class VisualBertPredictionHeadTransform(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class VisualBertLMPredictionHead(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class VisualBertPreTrainingHeads(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, sequence_output, pooled_output):  # -> tuple[Any, Any]:
        ...

@auto_docstring
class VisualBertPreTrainedModel(PreTrainedModel):
    config: VisualBertConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...

@dataclass
@auto_docstring(custom_intro=...)
class VisualBertForPreTrainingOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    prediction_logits: torch.FloatTensor | None = ...
    seq_relationship_logits: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...

@auto_docstring(custom_intro=...)
class VisualBertModel(VisualBertPreTrainedModel):
    def __init__(self, config, add_pooling_layer=...) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        head_mask: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        visual_embeds: torch.FloatTensor | None = ...,
        visual_attention_mask: torch.LongTensor | None = ...,
        visual_token_type_ids: torch.LongTensor | None = ...,
        image_text_alignment: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor] | BaseModelOutputWithPooling: ...

@auto_docstring(custom_intro=...)
class VisualBertForPreTraining(VisualBertPreTrainedModel):
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
        attention_mask: torch.LongTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        head_mask: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        visual_embeds: torch.FloatTensor | None = ...,
        visual_attention_mask: torch.LongTensor | None = ...,
        visual_token_type_ids: torch.LongTensor | None = ...,
        image_text_alignment: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: torch.LongTensor | None = ...,
        sentence_image_labels: torch.LongTensor | None = ...,
    ) -> tuple[torch.Tensor] | VisualBertForPreTrainingOutput: ...

@auto_docstring
class VisualBertForMultipleChoice(VisualBertPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        head_mask: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        visual_embeds: torch.FloatTensor | None = ...,
        visual_attention_mask: torch.LongTensor | None = ...,
        visual_token_type_ids: torch.LongTensor | None = ...,
        image_text_alignment: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: torch.LongTensor | None = ...,
    ) -> tuple[torch.Tensor] | MultipleChoiceModelOutput: ...

@auto_docstring(custom_intro=...)
class VisualBertForQuestionAnswering(VisualBertPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        head_mask: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        visual_embeds: torch.FloatTensor | None = ...,
        visual_attention_mask: torch.LongTensor | None = ...,
        visual_token_type_ids: torch.LongTensor | None = ...,
        image_text_alignment: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: torch.LongTensor | None = ...,
    ) -> tuple[torch.Tensor] | SequenceClassifierOutput: ...

@auto_docstring(custom_intro=...)
class VisualBertForVisualReasoning(VisualBertPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        head_mask: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        visual_embeds: torch.FloatTensor | None = ...,
        visual_attention_mask: torch.LongTensor | None = ...,
        visual_token_type_ids: torch.LongTensor | None = ...,
        image_text_alignment: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: torch.LongTensor | None = ...,
    ) -> tuple[torch.Tensor] | SequenceClassifierOutput: ...

class VisualBertRegionToPhraseAttention(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, query, key, attention_mask): ...

@auto_docstring(custom_intro=...)
class VisualBertForRegionToPhraseAlignment(VisualBertPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        head_mask: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        visual_embeds: torch.FloatTensor | None = ...,
        visual_attention_mask: torch.LongTensor | None = ...,
        visual_token_type_ids: torch.LongTensor | None = ...,
        image_text_alignment: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        region_to_phrase_position: torch.LongTensor | None = ...,
        labels: torch.LongTensor | None = ...,
    ) -> tuple[torch.Tensor] | SequenceClassifierOutput: ...

__all__ = [
    "VisualBertForMultipleChoice",
    "VisualBertForPreTraining",
    "VisualBertForQuestionAnswering",
    "VisualBertForRegionToPhraseAlignment",
    "VisualBertForVisualReasoning",
    "VisualBertLayer",
    "VisualBertModel",
    "VisualBertPreTrainedModel",
]
