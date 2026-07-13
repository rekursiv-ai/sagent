from dataclasses import dataclass

from torch import nn

import torch

from .configuration_realm import RealmConfig
from ....cache_utils import Cache
from ....modeling_layers import GradientCheckpointingLayer
from ....modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    MaskedLMOutput,
    ModelOutput,
)
from ....modeling_utils import PreTrainedModel
from ....utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ....utils.deprecation import deprecate_kwarg

"""PyTorch REALM model."""
logger = ...
_EMBEDDER_CHECKPOINT_FOR_DOC = ...
_ENCODER_CHECKPOINT_FOR_DOC = ...
_SCORER_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...

def load_tf_weights_in_realm(
    model, config, tf_checkpoint_path
):  # -> RealmReader | transformers.models.deprecated.realm.modeling_realm.<subclass of RealmReader and RealmEmbedder> | RealmEmbedder | RealmKnowledgeAugEncoder | RealmForOpenQA:
    ...

class RealmEmbeddings(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        past_key_values_length: int = ...,
    ) -> torch.Tensor: ...

class RealmSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=...) -> None: ...
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        encoder_hidden_states: torch.FloatTensor | None = ...,
        encoder_attention_mask: torch.FloatTensor | None = ...,
        past_key_values: Cache | None = ...,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.Tensor]: ...

class RealmSelfOutput(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor: ...

REALM_SELF_ATTENTION_CLASSES = ...

class RealmAttention(nn.Module):
    def __init__(self, config, position_embedding_type=...) -> None: ...
    def prune_heads(self, heads):  # -> None:
        ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        encoder_hidden_states: torch.FloatTensor | None = ...,
        encoder_attention_mask: torch.FloatTensor | None = ...,
        past_key_values: Cache | None = ...,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.Tensor]: ...

class RealmIntermediate(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class RealmOutput(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor: ...

class RealmLayer(GradientCheckpointingLayer):
    def __init__(self, config) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        encoder_hidden_states: torch.FloatTensor | None = ...,
        encoder_attention_mask: torch.FloatTensor | None = ...,
        past_key_values: Cache | None = ...,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.Tensor]: ...
    def feed_forward_chunk(self, attention_output):  # -> Any:
        ...

class RealmEncoder(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        encoder_hidden_states: torch.FloatTensor | None = ...,
        encoder_attention_mask: torch.FloatTensor | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor] | BaseModelOutputWithPastAndCrossAttentions: ...

class RealmPooler(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

@dataclass
class RealmEmbedderOutput(ModelOutput):
    projected_score: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...

@dataclass
class RealmScorerOutput(ModelOutput):
    relevance_score: torch.FloatTensor | None = ...
    query_score: torch.FloatTensor | None = ...
    candidate_score: torch.FloatTensor | None = ...

@dataclass
class RealmReaderOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    retriever_loss: torch.FloatTensor | None = ...
    reader_loss: torch.FloatTensor | None = ...
    retriever_correct: torch.BoolTensor | None = ...
    reader_correct: torch.BoolTensor | None = ...
    block_idx: torch.LongTensor | None = ...
    candidate: torch.LongTensor | None = ...
    start_pos: torch.IntTensor | None = ...
    end_pos: torch.IntTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...

@dataclass
class RealmForOpenQAOutput(ModelOutput):
    reader_output: dict | None = ...
    predicted_answer_ids: torch.LongTensor | None = ...

class RealmPredictionHeadTransform(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class RealmLMPredictionHead(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class RealmOnlyMLMHead(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, sequence_output):  # -> Any:
        ...

class RealmScorerProjection(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class RealmReaderProjection(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states, block_mask):  # -> tuple[Any, Tensor, Tensor]:
        ...

REALM_START_DOCSTRING = ...
REALM_INPUTS_DOCSTRING = ...

class RealmPreTrainedModel(PreTrainedModel):
    config: RealmConfig
    load_tf_weights = ...
    base_model_prefix = ...

class RealmBertModel(RealmPreTrainedModel):
    def __init__(self, config, add_pooling_layer=...) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def forward(
        self,
        input_ids=...,
        attention_mask=...,
        token_type_ids=...,
        position_ids=...,
        head_mask=...,
        inputs_embeds=...,
        encoder_hidden_states=...,
        encoder_attention_mask=...,
        past_key_values=...,
        use_cache=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
    ):  # -> Any | BaseModelOutputWithPoolingAndCrossAttentions:
        ...

@add_start_docstrings(
    ...,
    REALM_START_DOCSTRING,
)
class RealmEmbedder(RealmPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @add_start_docstrings_to_model_forward(
        REALM_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @replace_return_docstrings(
        output_type=RealmEmbedderOutput, config_class=_CONFIG_FOR_DOC
    )
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
    ) -> tuple | RealmEmbedderOutput: ...

@add_start_docstrings(
    ...,
    REALM_START_DOCSTRING,
)
class RealmScorer(RealmPreTrainedModel):
    def __init__(self, config, query_embedder=...) -> None: ...
    @add_start_docstrings_to_model_forward(
        REALM_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @replace_return_docstrings(
        output_type=RealmScorerOutput, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        candidate_input_ids: torch.LongTensor | None = ...,
        candidate_attention_mask: torch.FloatTensor | None = ...,
        candidate_token_type_ids: torch.LongTensor | None = ...,
        candidate_inputs_embeds: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | RealmScorerOutput: ...

@add_start_docstrings(
    ...,
    REALM_START_DOCSTRING,
)
class RealmKnowledgeAugEncoder(RealmPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def get_output_embeddings(self):  # -> Linear:
        ...
    def set_output_embeddings(self, new_embeddings):  # -> None:
        ...
    @add_start_docstrings_to_model_forward(
        REALM_INPUTS_DOCSTRING.format("batch_size, num_candidates, sequence_length")
    )
    @replace_return_docstrings(output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        relevance_score: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        mlm_mask: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | MaskedLMOutput: ...

@add_start_docstrings("The reader of REALM.", REALM_START_DOCSTRING)
class RealmReader(RealmPreTrainedModel):
    def __init__(self, config) -> None: ...
    @add_start_docstrings_to_model_forward(
        REALM_INPUTS_DOCSTRING.format("reader_beam_size, sequence_length")
    )
    @replace_return_docstrings(
        output_type=RealmReaderOutput, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        relevance_score: torch.FloatTensor | None = ...,
        block_mask: torch.BoolTensor | None = ...,
        start_positions: torch.LongTensor | None = ...,
        end_positions: torch.LongTensor | None = ...,
        has_answers: torch.BoolTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | RealmReaderOutput: ...

REALM_FOR_OPEN_QA_DOCSTRING = ...

@add_start_docstrings(
    ...,
    REALM_START_DOCSTRING,
)
class RealmForOpenQA(RealmPreTrainedModel):
    def __init__(self, config, retriever=...) -> None: ...
    @property
    def searcher_beam_size(self):  # -> int:
        ...
    def block_embedding_to(self, device):  # -> None:
        ...
    @add_start_docstrings_to_model_forward(
        REALM_FOR_OPEN_QA_DOCSTRING.format("1, sequence_length")
    )
    @replace_return_docstrings(
        output_type=RealmForOpenQAOutput, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: torch.LongTensor | None,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        answer_ids: torch.LongTensor | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | RealmForOpenQAOutput: ...

__all__ = [
    "RealmEmbedder",
    "RealmForOpenQA",
    "RealmKnowledgeAugEncoder",
    "RealmPreTrainedModel",
    "RealmReader",
    "RealmScorer",
    "load_tf_weights_in_realm",
]
