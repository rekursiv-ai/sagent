from dataclasses import dataclass

import enum

from torch import nn

import torch

from .configuration_tapas import TapasConfig
from ...cache_utils import Cache
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutputWithPooling,
    MaskedLMOutput,
    SequenceClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring
from ...utils.deprecation import deprecate_kwarg

"""PyTorch TAPAS model."""
logger = ...
EPSILON_ZERO_DIVISION = ...
CLOSE_ENOUGH_TO_LOG_ZERO = ...

@dataclass
@auto_docstring(custom_intro=...)
class TableQuestionAnsweringOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    logits_aggregation: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...

def load_tf_weights_in_tapas(
    model, config, tf_checkpoint_path
):  # -> TapasForSequenceClassification | TapasModel | TapasForMaskedLM:
    ...

class TapasEmbeddings(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, input_ids=..., token_type_ids=..., position_ids=..., inputs_embeds=...
    ):  # -> Any:
        ...

class TapasSelfAttention(nn.Module):
    def __init__(self, config, layer_idx=...) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states,
        attention_mask=...,
        head_mask=...,
        encoder_hidden_states=...,
        past_key_values=...,
        output_attentions=...,
        cache_position=...,
    ):  # -> tuple[Tensor | Any, ...] | tuple[Tensor | EncoderDecoderCache | Any | None, ...] | tuple[Tensor, Any] | tuple[Tensor]:
        ...

class TapasSelfOutput(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor: ...

class TapasAttention(nn.Module):
    def __init__(self, config, layer_idx=...) -> None: ...
    def prune_heads(self, heads):  # -> None:
        ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        encoder_hidden_states: torch.FloatTensor | None = ...,
        past_key_values: Cache | None = ...,
        output_attentions: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple[torch.Tensor]: ...

class TapasIntermediate(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class TapasOutput(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor: ...

class TapasLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx=...) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        encoder_hidden_states: torch.FloatTensor | None = ...,
        encoder_attention_mask: torch.FloatTensor | None = ...,
        past_key_values: Cache | None = ...,
        output_attentions: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple[torch.Tensor]: ...
    def feed_forward_chunk(self, attention_output):  # -> Any:
        ...

class TapasEncoder(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states,
        attention_mask=...,
        head_mask=...,
        encoder_hidden_states=...,
        encoder_attention_mask=...,
        past_key_values=...,
        use_cache=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
        cache_position=...,
    ):  # -> tuple[Any | tuple[Any, ...] | tuple[()], ...] | BaseModelOutput:
        ...

class TapasPooler(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class TapasPredictionHeadTransform(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class TapasLMPredictionHead(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class TapasOnlyMLMHead(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor: ...

@auto_docstring
class TapasPreTrainedModel(PreTrainedModel):
    config: TapasConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _supports_param_buffer_assignment = ...

@auto_docstring
class TapasModel(TapasPreTrainedModel):
    def __init__(self, config, add_pooling_layer=...) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
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
        encoder_hidden_states: torch.FloatTensor | None = ...,
        encoder_attention_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutputWithPooling: ...

@auto_docstring
class TapasForMaskedLM(TapasPreTrainedModel):
    _tied_weights_keys = ...
    config: TapasConfig
    base_model_prefix = ...
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
        encoder_hidden_states: torch.FloatTensor | None = ...,
        encoder_attention_mask: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        **kwargs,
    ) -> tuple | MaskedLMOutput: ...

@auto_docstring(custom_intro=...)
class TapasForQuestionAnswering(TapasPreTrainedModel):
    def __init__(self, config: TapasConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        table_mask: torch.LongTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        aggregation_labels: torch.LongTensor | None = ...,
        float_answer: torch.FloatTensor | None = ...,
        numeric_values: torch.FloatTensor | None = ...,
        numeric_values_scale: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | TableQuestionAnsweringOutput: ...

@auto_docstring(custom_intro=...)
class TapasForSequenceClassification(TapasPreTrainedModel):
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
    ) -> tuple[torch.Tensor] | SequenceClassifierOutput: ...

class AverageApproximationFunction(enum.StrEnum):
    RATIO = ...
    FIRST_ORDER = ...
    SECOND_ORDER = ...

class IndexMap:
    def __init__(self, indices, num_segments, batch_dims=...) -> None: ...
    def batch_shape(self):  # -> Size:
        ...

class ProductIndexMap(IndexMap):
    def __init__(self, outer_index, inner_index) -> None: ...
    def project_outer(self, index):  # -> IndexMap:
        ...
    def project_inner(self, index):  # -> IndexMap:
        ...

def gather(values, index, name=...):  # -> Tensor:
    ...
def flatten(index, name=...):  # -> IndexMap:
    ...
def range_index_map(batch_shape, num_segments, name=...):  # -> IndexMap:
    ...
def reduce_sum(values, index, name=...):  # -> tuple[Tensor, IndexMap]:
    ...
def reduce_mean(values, index, name=...):  # -> tuple[Tensor, IndexMap]:
    ...
def reduce_max(values, index, name=...):  # -> tuple[Tensor, IndexMap]:
    ...
def reduce_min(values, index, name=...):  # -> tuple[Tensor, IndexMap]:
    ...
def compute_column_logits(
    sequence_output,
    column_output_weights,
    column_output_bias,
    cell_index,
    cell_mask,
    allow_empty_column_selection,
):  # -> Tensor:
    ...
def compute_token_logits(sequence_output, temperature, output_weights, output_bias): ...
def huber_loss(input, target, delta: float = ...):  # -> Tensor:
    ...

__all__ = [
    "TapasForMaskedLM",
    "TapasForQuestionAnswering",
    "TapasForSequenceClassification",
    "TapasModel",
    "TapasPreTrainedModel",
    "load_tf_weights_in_tapas",
]
