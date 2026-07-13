from torch import nn

import torch

from .configuration_deberta_v2 import DebertaV2Config
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring

"""PyTorch DeBERTa-v2 model."""
logger = ...

class DebertaV2SelfOutput(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states, input_tensor):  # -> Any:
        ...

@torch.jit.script
def make_log_bucket_position(
    relative_pos, bucket_size: int, max_position: int
):  # -> Tensor:
    ...
def build_relative_position(
    query_layer, key_layer, bucket_size: int = ..., max_position: int = ...
):  # -> Tensor:
    ...
@torch.jit.script
def c2p_dynamic_expand(c2p_pos, query_layer, relative_pos): ...
@torch.jit.script
def p2c_dynamic_expand(c2p_pos, query_layer, key_layer): ...
@torch.jit.script
def pos_dynamic_expand(pos_index, p2c_att, key_layer): ...
@torch.jit.script
def scaled_size_sqrt(query_layer: torch.Tensor, scale_factor: int):  # -> Tensor:
    ...
@torch.jit.script
def build_rpos(
    query_layer,
    key_layer,
    relative_pos,
    position_buckets: int,
    max_relative_positions: int,
):  # -> Tensor:
    ...

class DisentangledSelfAttention(nn.Module):
    def __init__(self, config) -> None: ...
    def transpose_for_scores(self, x, attention_heads) -> torch.Tensor: ...
    def forward(
        self,
        hidden_states,
        attention_mask,
        output_attentions=...,
        query_states=...,
        relative_pos=...,
        rel_embeddings=...,
    ):  # -> tuple[Tensor, None] | tuple[Tensor, Any]:
        ...
    def disentangled_attention_bias(
        self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor
    ):  # -> Literal[0]:
        ...

class DebertaV2Attention(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states,
        attention_mask,
        output_attentions: bool = ...,
        query_states=...,
        relative_pos=...,
        rel_embeddings=...,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class DebertaV2Intermediate(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class DebertaV2Output(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states, input_tensor):  # -> Any:
        ...

class DebertaV2Layer(GradientCheckpointingLayer):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states,
        attention_mask,
        query_states=...,
        relative_pos=...,
        rel_embeddings=...,
        output_attentions: bool = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class ConvLayer(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states, residual_states, input_mask):  # -> Any:
        ...

class DebertaV2Embeddings(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        input_ids=...,
        token_type_ids=...,
        position_ids=...,
        mask=...,
        inputs_embeds=...,
    ):  # -> Any:
        ...

class DebertaV2Encoder(nn.Module):
    def __init__(self, config) -> None: ...
    def get_rel_embedding(self):  # -> Any | Tensor | None:
        ...
    def get_attention_mask(self, attention_mask): ...
    def get_rel_pos(
        self, hidden_states, query_states=..., relative_pos=...
    ):  # -> Tensor | None:
        ...
    def forward(
        self,
        hidden_states,
        attention_mask,
        output_hidden_states=...,
        output_attentions=...,
        query_states=...,
        relative_pos=...,
        return_dict=...,
    ):  # -> tuple[Any | tuple[Tensor] | tuple[()] | tuple[Any, ...], ...] | BaseModelOutput:
        ...

@auto_docstring
class DebertaV2PreTrainedModel(PreTrainedModel):
    config: DebertaV2Config
    base_model_prefix = ...
    _keys_to_ignore_on_load_unexpected = ...
    supports_gradient_checkpointing = ...

@auto_docstring
class DebertaV2Model(DebertaV2PreTrainedModel):
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
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutput: ...

class LegacyDebertaV2PredictionHeadTransform(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class LegacyDebertaV2LMPredictionHead(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class LegacyDebertaV2OnlyMLMHead(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, sequence_output):  # -> Any:
        ...

class DebertaV2LMPredictionHead(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states, word_embeddings):  # -> Tensor:
        ...

class DebertaV2OnlyMLMHead(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, sequence_output, word_embeddings):  # -> Any:
        ...

@auto_docstring
class DebertaV2ForMaskedLM(DebertaV2PreTrainedModel):
    _tied_weights_keys = ...
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config) -> None: ...
    def get_output_embeddings(self):  # -> Linear:
        ...
    def set_output_embeddings(self, new_embeddings):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | MaskedLMOutput: ...

class ContextPooler(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states): ...
    @property
    def output_dim(self): ...

@auto_docstring(custom_intro=...)
class DebertaV2ForSequenceClassification(DebertaV2PreTrainedModel):
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
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | SequenceClassifierOutput: ...

@auto_docstring
class DebertaV2ForTokenClassification(DebertaV2PreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | TokenClassifierOutput: ...

@auto_docstring
class DebertaV2ForQuestionAnswering(DebertaV2PreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        start_positions: torch.Tensor | None = ...,
        end_positions: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | QuestionAnsweringModelOutput: ...

@auto_docstring
class DebertaV2ForMultipleChoice(DebertaV2PreTrainedModel):
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
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | MultipleChoiceModelOutput: ...

__all__ = [
    "DebertaV2ForMaskedLM",
    "DebertaV2ForMultipleChoice",
    "DebertaV2ForQuestionAnswering",
    "DebertaV2ForSequenceClassification",
    "DebertaV2ForTokenClassification",
    "DebertaV2Model",
    "DebertaV2PreTrainedModel",
]
