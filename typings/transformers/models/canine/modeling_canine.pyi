from typing import Any
from dataclasses import dataclass

from torch import nn

import torch

from .configuration_canine import CanineConfig
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    ModelOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring

"""PyTorch CANINE model."""
logger = ...
_PRIMES = ...

@dataclass
@auto_docstring(custom_intro=...)
class CanineModelOutputWithPooling(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = ...
    pooler_output: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...

def load_tf_weights_in_canine(model, config, tf_checkpoint_path): ...

class CanineEmbeddings(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
    ) -> torch.FloatTensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.FloatTensor: ...

class CharactersToMolecules(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, char_encoding: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class ConvProjection(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        inputs: torch.Tensor,
        final_seq_char_positions: torch.Tensor | None = ...,
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class CanineSelfAttention(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        from_tensor: torch.Tensor,
        to_tensor: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class CanineSelfOutput(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states: tuple[torch.FloatTensor], input_tensor: torch.FloatTensor
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]: ...

class CanineAttention(nn.Module):
    def __init__(
        self,
        config,
        local=...,
        always_attend_to_first_position: bool = ...,
        first_position_attends_to_all: bool = ...,
        attend_from_chunk_width: int = ...,
        attend_from_chunk_stride: int = ...,
        attend_to_chunk_width: int = ...,
        attend_to_chunk_stride: int = ...,
    ) -> None: ...
    def prune_heads(self, heads):  # -> None:
        ...
    def forward(
        self,
        hidden_states: tuple[torch.FloatTensor],
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor | None]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.FloatTensor, torch.FloatTensor | None]: ...

class CanineIntermediate(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.FloatTensor: ...

class CanineOutput(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states: tuple[torch.FloatTensor], input_tensor: torch.FloatTensor
    ) -> torch.FloatTensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.FloatTensor: ...

class CanineLayer(GradientCheckpointingLayer):
    def __init__(
        self,
        config,
        local,
        always_attend_to_first_position,
        first_position_attends_to_all,
        attend_from_chunk_width,
        attend_from_chunk_stride,
        attend_to_chunk_width,
        attend_to_chunk_stride,
    ) -> None: ...
    def forward(
        self,
        hidden_states: tuple[torch.FloatTensor],
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor | None]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.FloatTensor, torch.FloatTensor | None]: ...
    def feed_forward_chunk(self, attention_output):  # -> Any:
        ...

class CanineEncoder(nn.Module):
    def __init__(
        self,
        config,
        local=...,
        always_attend_to_first_position=...,
        first_position_attends_to_all=...,
        attend_from_chunk_width=...,
        attend_from_chunk_stride=...,
        attend_to_chunk_width=...,
        attend_to_chunk_stride=...,
    ) -> None: ...
    def forward(
        self,
        hidden_states: tuple[torch.FloatTensor],
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutput: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple | BaseModelOutput: ...

class CaninePooler(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: tuple[torch.FloatTensor]) -> torch.FloatTensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.FloatTensor: ...

class CaninePredictionHeadTransform(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: tuple[torch.FloatTensor]) -> torch.FloatTensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.FloatTensor: ...

class CanineLMPredictionHead(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: tuple[torch.FloatTensor]) -> torch.FloatTensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.FloatTensor: ...

class CanineOnlyMLMHead(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, sequence_output: tuple[torch.Tensor]) -> tuple[torch.Tensor]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor]: ...

@auto_docstring
class CaninePreTrainedModel(PreTrainedModel):
    config: CanineConfig
    load_tf_weights = ...
    base_model_prefix = ...
    supports_gradient_checkpointing = ...

@auto_docstring
class CanineModel(CaninePreTrainedModel):
    def __init__(self, config, add_pooling_layer=...) -> None: ...
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
    ) -> tuple | CanineModelOutputWithPooling: ...

@auto_docstring(custom_intro=...)
class CanineForSequenceClassification(CaninePreTrainedModel):
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
    ) -> tuple | SequenceClassifierOutput: ...

@auto_docstring
class CanineForMultipleChoice(CaninePreTrainedModel):
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
    ) -> tuple | MultipleChoiceModelOutput: ...

@auto_docstring
class CanineForTokenClassification(CaninePreTrainedModel):
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
    ) -> tuple | TokenClassifierOutput: ...

@auto_docstring
class CanineForQuestionAnswering(CaninePreTrainedModel):
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
    ) -> tuple | QuestionAnsweringModelOutput: ...

__all__ = [
    "CanineForMultipleChoice",
    "CanineForQuestionAnswering",
    "CanineForSequenceClassification",
    "CanineForTokenClassification",
    "CanineLayer",
    "CanineModel",
    "CaninePreTrainedModel",
    "load_tf_weights_in_canine",
]
