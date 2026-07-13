from torch import nn

import torch

from .configuration_distilbert import DistilBertConfig
from ...configuration_utils import PretrainedConfig
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

"""
PyTorch DistilBERT model adapted in part from Facebook, Inc XLM model (https://github.com/facebookresearch/XLM) and in
part from HuggingFace PyTorch version of Google AI Bert model (https://github.com/google-research/bert)
"""
logger = ...

def create_sinusoidal_embeddings(n_pos: int, dim: int, out: torch.Tensor):  # -> None:
    ...

class Embeddings(nn.Module):
    def __init__(self, config: PretrainedConfig) -> None: ...
    def forward(
        self, input_ids: torch.Tensor, input_embeds: torch.Tensor | None = ...
    ) -> torch.Tensor: ...

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: PretrainedConfig) -> None: ...
    def prune_heads(self, heads: list[int]):  # -> None:
        ...
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        head_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
    ) -> tuple[torch.Tensor, ...]: ...

class DistilBertFlashAttention2(MultiHeadSelfAttention):
    def __init__(self, *args, **kwargs) -> None: ...
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        head_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
    ) -> tuple[torch.Tensor, ...]: ...

class DistilBertSdpaAttention(MultiHeadSelfAttention):
    def __init__(self, config: PretrainedConfig) -> None: ...
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        head_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
    ) -> tuple[torch.Tensor, ...]: ...

class FFN(nn.Module):
    def __init__(self, config: PretrainedConfig) -> None: ...
    def forward(self, input: torch.Tensor) -> torch.Tensor: ...
    def ff_chunk(self, input: torch.Tensor) -> torch.Tensor: ...

DISTILBERT_ATTENTION_CLASSES = ...

class TransformerBlock(GradientCheckpointingLayer):
    def __init__(self, config: PretrainedConfig) -> None: ...
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
    ) -> tuple[torch.Tensor, ...]: ...

class Transformer(nn.Module):
    def __init__(self, config: PretrainedConfig) -> None: ...
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool | None = ...,
    ) -> BaseModelOutput | tuple[torch.Tensor, ...]: ...

@auto_docstring
class DistilBertPreTrainedModel(PreTrainedModel):
    config: DistilBertConfig
    load_tf_weights = ...
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...

@auto_docstring
class DistilBertModel(DistilBertPreTrainedModel):
    def __init__(self, config: PretrainedConfig) -> None: ...
    def get_position_embeddings(self) -> nn.Embedding: ...
    def resize_position_embeddings(self, new_num_position_embeddings: int):  # -> None:
        ...
    def get_input_embeddings(self) -> nn.Embedding: ...
    def set_input_embeddings(self, new_embeddings: nn.Embedding):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> BaseModelOutput | tuple[torch.Tensor, ...]: ...

@auto_docstring(custom_intro=...)
class DistilBertForMaskedLM(DistilBertPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config: PretrainedConfig) -> None: ...
    def get_position_embeddings(self) -> nn.Embedding: ...
    def resize_position_embeddings(self, new_num_position_embeddings: int):  # -> None:
        ...
    def get_output_embeddings(self) -> nn.Module: ...
    def set_output_embeddings(self, new_embeddings: nn.Module):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> MaskedLMOutput | tuple[torch.Tensor, ...]: ...

@auto_docstring(custom_intro=...)
class DistilBertForSequenceClassification(DistilBertPreTrainedModel):
    def __init__(self, config: PretrainedConfig) -> None: ...
    def get_position_embeddings(self) -> nn.Embedding: ...
    def resize_position_embeddings(self, new_num_position_embeddings: int):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> SequenceClassifierOutput | tuple[torch.Tensor, ...]: ...

@auto_docstring
class DistilBertForQuestionAnswering(DistilBertPreTrainedModel):
    def __init__(self, config: PretrainedConfig) -> None: ...
    def get_position_embeddings(self) -> nn.Embedding: ...
    def resize_position_embeddings(self, new_num_position_embeddings: int):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        start_positions: torch.Tensor | None = ...,
        end_positions: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> QuestionAnsweringModelOutput | tuple[torch.Tensor, ...]: ...

@auto_docstring
class DistilBertForTokenClassification(DistilBertPreTrainedModel):
    def __init__(self, config: PretrainedConfig) -> None: ...
    def get_position_embeddings(self) -> nn.Embedding: ...
    def resize_position_embeddings(self, new_num_position_embeddings: int):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> TokenClassifierOutput | tuple[torch.Tensor, ...]: ...

@auto_docstring
class DistilBertForMultipleChoice(DistilBertPreTrainedModel):
    def __init__(self, config: PretrainedConfig) -> None: ...
    def get_position_embeddings(self) -> nn.Embedding: ...
    def resize_position_embeddings(self, new_num_position_embeddings: int):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> MultipleChoiceModelOutput | tuple[torch.Tensor, ...]: ...

__all__ = [
    "DistilBertForMaskedLM",
    "DistilBertForMultipleChoice",
    "DistilBertForQuestionAnswering",
    "DistilBertForSequenceClassification",
    "DistilBertForTokenClassification",
    "DistilBertModel",
    "DistilBertPreTrainedModel",
]
