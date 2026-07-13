from torch import nn

import torch

from .configuration_deberta import DebertaConfig
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring

"""PyTorch DeBERTa model."""
logger = ...

class DebertaLayerNorm(nn.Module):
    def __init__(self, size, eps=...) -> None: ...
    def forward(self, hidden_states): ...

class DebertaSelfOutput(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states, input_tensor):  # -> Any:
        ...

@torch.jit.script
def build_relative_position(query_layer, key_layer):  # -> Tensor:
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
def build_rpos(query_layer: torch.Tensor, key_layer: torch.Tensor, relative_pos): ...
@torch.jit.script
def compute_attention_span(
    query_layer: torch.Tensor, key_layer: torch.Tensor, max_relative_positions: int
):  # -> Tensor:
    ...
@torch.jit.script
def uneven_size_corrected(
    p2c_att, query_layer: torch.Tensor, key_layer: torch.Tensor, relative_pos
):  # -> Tensor:
    ...

class DisentangledSelfAttention(nn.Module):
    def __init__(self, config) -> None: ...
    def transpose_for_scores(self, x): ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool = ...,
        query_states: torch.Tensor | None = ...,
        relative_pos: torch.Tensor | None = ...,
        rel_embeddings: torch.Tensor | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
    def disentangled_att_bias(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        relative_pos: torch.Tensor,
        rel_embeddings: torch.Tensor,
        scale_factor: int,
    ):  # -> Tensor | Literal[0]:
        ...

class DebertaEmbeddings(nn.Module):
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

class DebertaAttention(nn.Module):
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

class DebertaIntermediate(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class DebertaOutput(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states, input_tensor):  # -> Any:
        ...

class DebertaLayer(GradientCheckpointingLayer):
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

class DebertaEncoder(nn.Module):
    def __init__(self, config) -> None: ...
    def get_rel_embedding(self):  # -> Tensor | None:
        ...
    def get_attention_mask(self, attention_mask): ...
    def get_rel_pos(
        self, hidden_states, query_states=..., relative_pos=...
    ):  # -> None:
        ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool = ...,
        output_attentions: bool = ...,
        query_states=...,
        relative_pos=...,
        return_dict: bool = ...,
    ):  # -> tuple[Tensor | tuple[Tensor] | tuple[()] | tuple[Any, ...], ...] | BaseModelOutput:
        ...

@auto_docstring
class DebertaPreTrainedModel(PreTrainedModel):
    config: DebertaConfig
    base_model_prefix = ...
    _keys_to_ignore_on_load_unexpected = ...
    supports_gradient_checkpointing = ...

@auto_docstring
class DebertaModel(DebertaPreTrainedModel):
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

class LegacyDebertaPredictionHeadTransform(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class LegacyDebertaLMPredictionHead(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class LegacyDebertaOnlyMLMHead(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor: ...

class DebertaLMPredictionHead(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states, word_embeddings):  # -> Tensor:
        ...

class DebertaOnlyMLMHead(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, sequence_output, word_embeddings):  # -> Any:
        ...

@auto_docstring
class DebertaForMaskedLM(DebertaPreTrainedModel):
    _tied_weights_keys = ...
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
class DebertaForSequenceClassification(DebertaPreTrainedModel):
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
class DebertaForTokenClassification(DebertaPreTrainedModel):
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
class DebertaForQuestionAnswering(DebertaPreTrainedModel):
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

__all__ = [
    "DebertaForMaskedLM",
    "DebertaForQuestionAnswering",
    "DebertaForSequenceClassification",
    "DebertaForTokenClassification",
    "DebertaModel",
    "DebertaPreTrainedModel",
]
