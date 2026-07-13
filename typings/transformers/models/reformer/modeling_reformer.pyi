from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from torch import nn
from torch.autograd.function import Function

import torch

from .configuration_reformer import ReformerConfig
from ...generation import GenerationMixin
from ...modeling_outputs import (
    CausalLMOutput,
    MaskedLMOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring

"""PyTorch REFORMER model."""
logger = ...
LSHSelfAttentionOutput = ...
LocalSelfAttentionOutput = ...
AttentionOutput = ...
ReformerOutput = ...
ReformerBackwardOutput = ...
ReformerEncoderOutput = ...

class ReformerDynamicCache:
    def __init__(self, _distributed_cache_data: Iterable | None = ...) -> None: ...
    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]: ...
    def __iter__(self):  # -> Generator[tuple[Tensor, Tensor], Any, None]:
        ...
    def __len__(self):  # -> int:
        ...
    def update(
        self,
        buckets: torch.Tensor,
        states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def get_seq_length(self, layer_idx: int | None = ...) -> int: ...
    def to_legacy_cache(self) -> tuple[tuple[torch.Tensor, torch.Tensor]]: ...
    @classmethod
    def from_legacy_cache(
        cls,
        past_buckets_states: tuple[tuple[torch.FloatTensor, torch.FloatTensor]]
        | None = ...,
    ) -> ReformerDynamicCache: ...

class AxialPositionEmbeddings(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, position_ids):  # -> Tensor:
        ...

class PositionEmbeddings(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, position_ids):  # -> Tensor:
        ...

class ReformerEmbeddings(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        input_ids=...,
        position_ids=...,
        inputs_embeds=...,
        start_idx_pos_encodings=...,
    ):  # -> Any:
        ...

class EfficientAttentionMixin: ...

class LSHSelfAttention(nn.Module, EfficientAttentionMixin):
    def __init__(self, config, layer_idx=...) -> None: ...
    def forward(
        self,
        hidden_states,
        attention_mask=...,
        head_mask=...,
        num_hashes=...,
        buckets=...,
        past_buckets_states=...,
        use_cache=...,
        output_attentions=...,
        cache_position=...,
        **kwargs,
    ): ...

class ReverseSort(Function):
    @staticmethod
    def forward(
        ctx, out_vectors, logits, sorted_bucket_idx, undo_sorted_bucket_idx
    ):  # -> tuple[Tensor, Tensor]:
        ...
    @staticmethod
    def backward(
        ctx, grad_out_vectors, grad_logits
    ):  # -> tuple[Tensor, Tensor, None, None]:
        ...

class LocalSelfAttention(nn.Module, EfficientAttentionMixin):
    def __init__(self, config, layer_idx=...) -> None: ...
    def forward(
        self,
        hidden_states,
        attention_mask=...,
        head_mask=...,
        past_buckets_states=...,
        use_cache=...,
        output_attentions=...,
        **kwargs,
    ):  # -> LocalSelfAttentionOutput:
        ...

class ReformerSelfOutput(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> Tensor:
        ...

class ReformerAttention(nn.Module):
    def __init__(self, config, layer_id=...) -> None: ...
    def forward(
        self,
        hidden_states,
        attention_mask=...,
        head_mask=...,
        num_hashes=...,
        past_buckets_states=...,
        use_cache=...,
        orig_sequence_length=...,
        output_attentions=...,
        buckets=...,
        cache_position=...,
    ):  # -> AttentionOutput:
        ...

class ReformerFeedForwardDense(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states): ...

class ReformerFeedForwardOutput(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> Tensor:
        ...

class ChunkReformerFeedForward(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, attention_output):  # -> Tensor:
        ...
    def forward_chunk(self, hidden_states):  # -> Any:
        ...

class ReformerLayer(nn.Module):
    def __init__(self, config, layer_id=...) -> None: ...
    def forward(
        self,
        prev_attn_output,
        hidden_states,
        attention_mask=...,
        head_mask=...,
        num_hashes=...,
        past_buckets_states=...,
        use_cache=...,
        orig_sequence_length=...,
        output_attentions=...,
    ):  # -> ReformerOutput:
        ...
    def backward_pass(
        self,
        next_attn_output,
        hidden_states,
        grad_attn_output,
        grad_hidden_states,
        attention_mask=...,
        head_mask=...,
        buckets=...,
    ):  # -> ReformerBackwardOutput:
        ...

class _ReversibleFunction(Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states,
        layers,
        attention_mask,
        head_mask,
        num_hashes,
        all_hidden_states,
        all_attentions,
        past_buckets_states,
        use_cache,
        orig_sequence_length,
        output_hidden_states,
        output_attentions,
    ):  # -> Tensor:
        ...
    @staticmethod
    def backward(
        ctx, grad_hidden_states
    ):  # -> tuple[Tensor, None, None, None, None, None, None, None, None, None, None, None]:
        ...

class ReformerEncoder(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states,
        attention_mask=...,
        head_mask=...,
        num_hashes=...,
        past_buckets_states=...,
        use_cache=...,
        orig_sequence_length=...,
        output_hidden_states=...,
        output_attentions=...,
    ):  # -> ReformerEncoderOutput:
        ...

class ReformerOnlyLMHead(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> Tensor:
        ...
    def forward_chunk(self, hidden_states):  # -> Any:
        ...

@auto_docstring
class ReformerPreTrainedModel(PreTrainedModel):
    config: ReformerConfig
    base_model_prefix = ...
    @property
    def dummy_inputs(self):  # -> dict[str, Tensor]:
        ...

@dataclass
@auto_docstring(
    custom_intro="""
    Output type of [`ReformerModel`].
    """
)
class ReformerModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor
    past_buckets_states: list[tuple[torch.LongTensor, torch.FloatTensor]] | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class ReformerModelWithLMHeadOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    past_buckets_states: list[tuple[torch.LongTensor, torch.FloatTensor]] | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...

@auto_docstring
class ReformerModel(ReformerPreTrainedModel):
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        num_hashes: int | None = ...,
        past_buckets_states: list[tuple[torch.Tensor]] | None = ...,
        use_cache: bool | None = ...,
        output_hidden_states: bool | None = ...,
        output_attentions: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | ReformerModelOutput: ...

@auto_docstring(custom_intro=...)
class ReformerModelWithLMHead(ReformerPreTrainedModel, GenerationMixin):
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
        position_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        num_hashes: int | None = ...,
        past_buckets_states: list[tuple[torch.Tensor]] | None = ...,
        use_cache: bool | None = ...,
        output_hidden_states: bool | None = ...,
        output_attentions: bool | None = ...,
        return_dict: bool | None = ...,
        labels: torch.Tensor | None = ...,
        **kwargs,
    ) -> tuple | CausalLMOutput: ...
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=..., use_cache=..., num_hashes=..., **kwargs
    ):  # -> dict[str, Any | None]:
        ...

@auto_docstring
class ReformerForMaskedLM(ReformerPreTrainedModel):
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
        position_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        num_hashes: int | None = ...,
        labels: torch.Tensor | None = ...,
        output_hidden_states: bool | None = ...,
        output_attentions: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | MaskedLMOutput: ...

@auto_docstring(custom_intro=...)
class ReformerForSequenceClassification(ReformerPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        num_hashes: int | None = ...,
        labels: torch.Tensor | None = ...,
        output_hidden_states: bool | None = ...,
        output_attentions: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | SequenceClassifierOutput: ...

class ReformerClassificationHead(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states, **kwargs):  # -> Any:
        ...

@auto_docstring
class ReformerForQuestionAnswering(ReformerPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        num_hashes: int | None = ...,
        start_positions: torch.Tensor | None = ...,
        end_positions: torch.Tensor | None = ...,
        output_hidden_states: bool | None = ...,
        output_attentions: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | QuestionAnsweringModelOutput: ...

__all__ = [
    "ReformerAttention",
    "ReformerForMaskedLM",
    "ReformerForQuestionAnswering",
    "ReformerForSequenceClassification",
    "ReformerLayer",
    "ReformerModel",
    "ReformerModelWithLMHead",
    "ReformerPreTrainedModel",
]
