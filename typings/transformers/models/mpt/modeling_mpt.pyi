from torch import nn

import torch

from .configuration_mpt import MptConfig
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring
from ...utils.deprecation import deprecate_kwarg

"""PyTorch MPT model."""
logger = ...

def build_mpt_alibi_tensor(
    num_heads, sequence_length, alibi_bias_max=..., device=...
):  # -> Tensor:
    ...

class MptAttention(nn.Module):
    def __init__(self, config: MptConfig, layer_idx: int | None = ...) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_bias: torch.Tensor,
        past_key_values: Cache | None = ...,
        attention_mask: torch.Tensor | None = ...,
        cache_position: torch.Tensor | None = ...,
    ):  # -> tuple[Any, Tensor]:
        ...

class MptMLP(nn.Module):
    def __init__(self, config: MptConfig) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor: ...

class MptBlock(GradientCheckpointingLayer):
    def __init__(self, config: MptConfig, layer_idx: int | None = ...) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_bias: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Cache | None = ...,
        use_cache: bool = ...,
        output_attentions: bool = ...,
        cache_position: torch.Tensor | None = ...,
    ):  # -> tuple[Any, Any]:
        ...

@auto_docstring
class MptPreTrainedModel(PreTrainedModel):
    config: MptConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, *inputs, **kwargs) -> None: ...

@auto_docstring
class MptModel(MptPreTrainedModel):
    def __init__(self, config: MptConfig) -> None: ...
    def get_input_embeddings(self):  # -> Embedding | Tensor:
        ...
    def build_mpt_alibi_tensor(
        self, num_heads, sequence_length, alibi_bias_max=..., device=...
    ):  # -> Tensor:
        ...
    def set_input_embeddings(self, new_embeddings: torch.Tensor):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        attention_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor, ...] | BaseModelOutputWithPastAndCrossAttentions: ...

@auto_docstring(custom_intro=...)
class MptForCausalLM(MptPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    def __init__(self, config: MptConfig) -> None: ...
    def set_output_embeddings(self, new_embeddings: torch.Tensor):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        attention_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor] | CausalLMOutputWithCrossAttentions: ...

@auto_docstring(custom_intro=...)
class MptForSequenceClassification(MptPreTrainedModel):
    def __init__(self, config: MptConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        attention_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor] | SequenceClassifierOutputWithPast: ...

@auto_docstring
class MptForTokenClassification(MptPreTrainedModel):
    def __init__(self, config: MptConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        attention_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        **deprecated_arguments,
    ) -> tuple[torch.Tensor] | TokenClassifierOutput: ...

@auto_docstring
class MptForQuestionAnswering(MptPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        start_positions: torch.LongTensor | None = ...,
        end_positions: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | QuestionAnsweringModelOutput: ...

__all__ = [
    "MptForCausalLM",
    "MptForQuestionAnswering",
    "MptForSequenceClassification",
    "MptForTokenClassification",
    "MptModel",
    "MptPreTrainedModel",
]
