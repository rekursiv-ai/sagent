from dataclasses import dataclass

from torch import nn

import torch

from .configuration_luke import LukeConfig
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring

"""PyTorch LUKE model."""
logger = ...

@dataclass
@auto_docstring(custom_intro=...)
class BaseLukeModelOutputWithPooling(BaseModelOutputWithPooling):
    entity_last_hidden_state: torch.FloatTensor | None = ...
    entity_hidden_states: tuple[torch.FloatTensor, ...] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class BaseLukeModelOutput(BaseModelOutput):
    entity_last_hidden_state: torch.FloatTensor | None = ...
    entity_hidden_states: tuple[torch.FloatTensor, ...] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class LukeMaskedLMOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    mlm_loss: torch.FloatTensor | None = ...
    mep_loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    entity_logits: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    entity_hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...

@dataclass
@auto_docstring(
    custom_intro="""
    Outputs of entity classification models.
    """
)
class EntityClassificationOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    entity_hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class EntityPairClassificationOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    entity_hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class EntitySpanClassificationOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    entity_hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class LukeSequenceClassifierOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    entity_hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class LukeTokenClassifierOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    entity_hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...

@dataclass
@auto_docstring(
    custom_intro="""
    Outputs of question answering models.
    """
)
class LukeQuestionAnsweringModelOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    start_logits: torch.FloatTensor | None = ...
    end_logits: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    entity_hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...

@dataclass
@auto_docstring(
    custom_intro="""
    Outputs of multiple choice models.
    """
)
class LukeMultipleChoiceModelOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    entity_hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...

class LukeEmbeddings(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, input_ids=..., token_type_ids=..., position_ids=..., inputs_embeds=...
    ):  # -> Any:
        ...
    def create_position_ids_from_inputs_embeds(self, inputs_embeds):  # -> Tensor:
        ...

class LukeEntityEmbeddings(nn.Module):
    def __init__(self, config: LukeConfig) -> None: ...
    def forward(
        self,
        entity_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        token_type_ids: torch.LongTensor | None = ...,
    ):  # -> Any:
        ...

class LukeSelfAttention(nn.Module):
    def __init__(self, config) -> None: ...
    def transpose_for_scores(self, x): ...
    def forward(
        self,
        word_hidden_states,
        entity_hidden_states,
        attention_mask=...,
        head_mask=...,
        output_attentions=...,
    ):  # -> tuple[Tensor, Tensor | None, Any] | tuple[Tensor, Tensor | None]:
        ...

class LukeSelfOutput(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor: ...

class LukeAttention(nn.Module):
    def __init__(self, config) -> None: ...
    def prune_heads(self, heads): ...
    def forward(
        self,
        word_hidden_states,
        entity_hidden_states,
        attention_mask=...,
        head_mask=...,
        output_attentions=...,
    ):  # -> Any:
        ...

class LukeIntermediate(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class LukeOutput(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor: ...

class LukeLayer(GradientCheckpointingLayer):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        word_hidden_states,
        entity_hidden_states,
        attention_mask=...,
        head_mask=...,
        output_attentions=...,
    ):  # -> Any:
        ...
    def feed_forward_chunk(self, attention_output):  # -> Any:
        ...

class LukeEncoder(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        word_hidden_states,
        entity_hidden_states,
        attention_mask=...,
        head_mask=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
    ):  # -> tuple[Any | tuple[Any, ...] | tuple[()], ...] | BaseLukeModelOutput:
        ...

class LukePooler(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class EntityPredictionHeadTransform(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class EntityPredictionHead(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

@auto_docstring
class LukePreTrainedModel(PreTrainedModel):
    config: LukeConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...

@auto_docstring(custom_intro=...)
class LukeModel(LukePreTrainedModel):
    def __init__(self, config: LukeConfig, add_pooling_layer: bool = ...) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def get_entity_embeddings(self):  # -> Embedding:
        ...
    def set_entity_embeddings(self, value):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        entity_ids: torch.LongTensor | None = ...,
        entity_attention_mask: torch.FloatTensor | None = ...,
        entity_token_type_ids: torch.LongTensor | None = ...,
        entity_position_ids: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseLukeModelOutputWithPooling: ...
    def get_extended_attention_mask(
        self,
        word_attention_mask: torch.LongTensor,
        entity_attention_mask: torch.LongTensor | None,
    ):  # -> Tensor:
        ...

def create_position_ids_from_input_ids(input_ids, padding_idx): ...

class LukeLMHead(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, features, **kwargs):  # -> Any:
        ...

@auto_docstring(custom_intro=...)
class LukeForMaskedLM(LukePreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    def tie_weights(self):  # -> None:
        ...
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
        entity_ids: torch.LongTensor | None = ...,
        entity_attention_mask: torch.LongTensor | None = ...,
        entity_token_type_ids: torch.LongTensor | None = ...,
        entity_position_ids: torch.LongTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        entity_labels: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | LukeMaskedLMOutput: ...

@auto_docstring(custom_intro=...)
class LukeForEntityClassification(LukePreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        entity_ids: torch.LongTensor | None = ...,
        entity_attention_mask: torch.FloatTensor | None = ...,
        entity_token_type_ids: torch.LongTensor | None = ...,
        entity_position_ids: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | EntityClassificationOutput: ...

@auto_docstring(custom_intro=...)
class LukeForEntityPairClassification(LukePreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        entity_ids: torch.LongTensor | None = ...,
        entity_attention_mask: torch.FloatTensor | None = ...,
        entity_token_type_ids: torch.LongTensor | None = ...,
        entity_position_ids: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | EntityPairClassificationOutput: ...

@auto_docstring(custom_intro=...)
class LukeForEntitySpanClassification(LukePreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        entity_ids: torch.LongTensor | None = ...,
        entity_attention_mask: torch.LongTensor | None = ...,
        entity_token_type_ids: torch.LongTensor | None = ...,
        entity_position_ids: torch.LongTensor | None = ...,
        entity_start_positions: torch.LongTensor | None = ...,
        entity_end_positions: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | EntitySpanClassificationOutput: ...

@auto_docstring(custom_intro=...)
class LukeForSequenceClassification(LukePreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        entity_ids: torch.LongTensor | None = ...,
        entity_attention_mask: torch.FloatTensor | None = ...,
        entity_token_type_ids: torch.LongTensor | None = ...,
        entity_position_ids: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | LukeSequenceClassifierOutput: ...

@auto_docstring(custom_intro=...)
class LukeForTokenClassification(LukePreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        entity_ids: torch.LongTensor | None = ...,
        entity_attention_mask: torch.FloatTensor | None = ...,
        entity_token_type_ids: torch.LongTensor | None = ...,
        entity_position_ids: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | LukeTokenClassifierOutput: ...

@auto_docstring
class LukeForQuestionAnswering(LukePreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.FloatTensor | None = ...,
        entity_ids: torch.LongTensor | None = ...,
        entity_attention_mask: torch.FloatTensor | None = ...,
        entity_token_type_ids: torch.LongTensor | None = ...,
        entity_position_ids: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        start_positions: torch.LongTensor | None = ...,
        end_positions: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | LukeQuestionAnsweringModelOutput: ...

@auto_docstring
class LukeForMultipleChoice(LukePreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        entity_ids: torch.LongTensor | None = ...,
        entity_attention_mask: torch.FloatTensor | None = ...,
        entity_token_type_ids: torch.LongTensor | None = ...,
        entity_position_ids: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | LukeMultipleChoiceModelOutput: ...

__all__ = [
    "LukeForEntityClassification",
    "LukeForEntityPairClassification",
    "LukeForEntitySpanClassification",
    "LukeForMaskedLM",
    "LukeForMultipleChoice",
    "LukeForQuestionAnswering",
    "LukeForSequenceClassification",
    "LukeForTokenClassification",
    "LukeModel",
    "LukePreTrainedModel",
]
