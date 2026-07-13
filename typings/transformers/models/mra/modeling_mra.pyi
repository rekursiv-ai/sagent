from torch import nn

import torch

from .configuration_mra import MraConfig
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring

"""PyTorch MRA model."""
logger = ...
mra_cuda_kernel = ...

def load_cuda_kernels():  # -> None:
    ...
def sparse_max(
    sparse_qk_prod, indices, query_num_block, key_num_block
):  # -> tuple[Any, Any]:
    ...
def sparse_mask(mask, indices, block_size=...): ...
def mm_to_sparse(dense_query, dense_key, indices, block_size=...): ...
def sparse_dense_mm(
    sparse_query, indices, dense_key, query_num_block, block_size=...
): ...
def transpose_indices(indices, dim_1_block, dim_2_block): ...

class MraSampledDenseMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dense_query, dense_key, indices, block_size): ...
    @staticmethod
    def backward(ctx, grad):  # -> tuple[Any, Any, None, None]:
        ...
    @staticmethod
    def operator_call(
        dense_query, dense_key, indices, block_size=...
    ):  # -> Any | None:
        ...

class MraSparseDenseMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sparse_query, indices, dense_key, query_num_block): ...
    @staticmethod
    def backward(ctx, grad):  # -> tuple[Any, None, Any, None]:
        ...
    @staticmethod
    def operator_call(
        sparse_query, indices, dense_key, query_num_block
    ):  # -> Any | None:
        ...

class MraReduceSum:
    @staticmethod
    def operator_call(
        sparse_query, indices, query_num_block, key_num_block
    ):  # -> Tensor:
        ...

def get_low_resolution_logit(
    query, key, block_size, mask=..., value=...
):  # -> tuple[Any | Tensor, Any, Any, Any | None]:
    ...
def get_block_idxes(
    low_resolution_logit,
    num_blocks,
    approx_mode,
    initial_prior_first_n_blocks,
    initial_prior_diagonal_n_blocks,
):  # -> tuple[Tensor, Any | None]:
    ...
def mra2_attention(
    query,
    key,
    value,
    mask,
    num_blocks,
    approx_mode,
    block_size=...,
    initial_prior_first_n_blocks=...,
    initial_prior_diagonal_n_blocks=...,
):  # -> Tensor | Any:
    ...

class MraEmbeddings(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, input_ids=..., token_type_ids=..., position_ids=..., inputs_embeds=...
    ):  # -> Any:
        ...

class MraSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=...) -> None: ...
    def forward(self, hidden_states, attention_mask=...):  # -> tuple[Tensor | Any]:
        ...

class MraSelfOutput(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor: ...

class MraAttention(nn.Module):
    def __init__(self, config, position_embedding_type=...) -> None: ...
    def prune_heads(self, heads):  # -> None:
        ...
    def forward(self, hidden_states, attention_mask=...):  # -> Any:
        ...

class MraIntermediate(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class MraOutput(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor: ...

class MraLayer(GradientCheckpointingLayer):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states, attention_mask=...):  # -> Any:
        ...
    def feed_forward_chunk(self, attention_output):  # -> Any:
        ...

class MraEncoder(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states,
        attention_mask=...,
        head_mask=...,
        output_hidden_states=...,
        return_dict=...,
    ):  # -> tuple[Any | tuple[Any, ...] | tuple[()], ...] | BaseModelOutputWithCrossAttentions:
        ...

class MraPredictionHeadTransform(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class MraLMPredictionHead(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class MraOnlyMLMHead(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor: ...

@auto_docstring
class MraPreTrainedModel(PreTrainedModel):
    config: MraConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...

@auto_docstring
class MraModel(MraPreTrainedModel):
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
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutputWithCrossAttentions: ...

@auto_docstring
class MraForMaskedLM(MraPreTrainedModel):
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
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | MaskedLMOutput: ...

class MraClassificationHead(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, features, **kwargs):  # -> Any:
        ...

@auto_docstring(custom_intro=...)
class MraForSequenceClassification(MraPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | SequenceClassifierOutput: ...

@auto_docstring
class MraForMultipleChoice(MraPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | MultipleChoiceModelOutput: ...

@auto_docstring
class MraForTokenClassification(MraPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | TokenClassifierOutput: ...

@auto_docstring
class MraForQuestionAnswering(MraPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        start_positions: torch.Tensor | None = ...,
        end_positions: torch.Tensor | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | QuestionAnsweringModelOutput: ...

__all__ = [
    "MraForMaskedLM",
    "MraForMultipleChoice",
    "MraForQuestionAnswering",
    "MraForSequenceClassification",
    "MraForTokenClassification",
    "MraLayer",
    "MraModel",
    "MraPreTrainedModel",
]
