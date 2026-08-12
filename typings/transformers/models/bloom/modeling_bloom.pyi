from typing import Any
from torch import nn

import torch

from .configuration_bloom import BloomConfig
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

"""PyTorch BLOOM model."""
logger = ...

def build_alibi_tensor(
    attention_mask: torch.Tensor, num_heads: int, dtype: torch.dtype
) -> torch.Tensor: ...
def dropout_add(
    x: torch.Tensor, residual: torch.Tensor, prob: float, training: bool
) -> torch.Tensor: ...
def bloom_gelu_forward(x: torch.Tensor) -> torch.Tensor: ...
def bloom_gelu_back(g: torch.Tensor, x: torch.Tensor) -> torch.Tensor: ...

class GeLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor: ...
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor: ...

class BloomGelu(nn.Module):
    def __init__(self) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class BloomAttention(nn.Module):
    def __init__(self, config: BloomConfig, layer_idx: int | None = ...) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Cache | None = ...,
        head_mask: torch.Tensor | None = ...,
        use_cache: bool = ...,
        output_attentions: bool = ...,
        cache_position: torch.LongTensor | None = ...,
    ):  # -> tuple[Tensor, Any]:
        ...

class BloomMLP(nn.Module):
    def __init__(self, config: BloomConfig) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class BloomBlock(GradientCheckpointingLayer):
    def __init__(self, config: BloomConfig, layer_idx: int | None = ...) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Cache | None = ...,
        head_mask: torch.Tensor | None = ...,
        use_cache: bool = ...,
        output_attentions: bool = ...,
        cache_position: torch.LongTensor | None = ...,
    ):  # -> tuple[Any, Any]:
        ...

@auto_docstring
class BloomPreTrainedModel(PreTrainedModel):
    config: BloomConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _skip_keys_device_placement = ...
    _can_compile_fullgraph = ...
    def __init__(self, *inputs, **kwargs) -> None: ...

@auto_docstring
class BloomModel(BloomPreTrainedModel):
    def __init__(self, config: BloomConfig) -> None: ...
    def build_alibi_tensor(
        self, attention_mask: torch.Tensor, num_heads: int, dtype: torch.dtype
    ) -> torch.Tensor: ...
    def get_input_embeddings(self):  # -> Embedding | Tensor:
        ...
    def set_input_embeddings(self, new_embeddings: torch.Tensor):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        past_key_values: Cache
        | tuple[tuple[torch.Tensor, torch.Tensor], ...]
        | None = ...,
        attention_mask: torch.Tensor | None = ...,
        head_mask: torch.LongTensor | None = ...,
        inputs_embeds: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **deprecated_arguments,
    ) -> tuple[torch.Tensor, ...] | BaseModelOutputWithPastAndCrossAttentions: ...

@auto_docstring(custom_intro=...)
class BloomForCausalLM(BloomPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    def __init__(self, config: BloomConfig) -> None: ...
    def set_output_embeddings(self, new_embeddings: torch.Tensor):  # -> None:
        ...
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=...,
        attention_mask=...,
        inputs_embeds=...,
        cache_position=...,
        use_cache=...,
        **kwargs,
    ):  # -> dict[str, Any | None]:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        past_key_values: Cache
        | tuple[tuple[torch.Tensor, torch.Tensor], ...]
        | None = ...,
        attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **deprecated_arguments,
    ) -> tuple[torch.Tensor] | CausalLMOutputWithCrossAttentions: ...

@auto_docstring(custom_intro=...)
class BloomForSequenceClassification(BloomPreTrainedModel):
    def __init__(self, config: BloomConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        past_key_values: Cache
        | tuple[tuple[torch.Tensor, torch.Tensor], ...]
        | None = ...,
        attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        **deprecated_arguments,
    ) -> tuple[torch.Tensor] | SequenceClassifierOutputWithPast: ...

@auto_docstring
class BloomForTokenClassification(BloomPreTrainedModel):
    def __init__(self, config: BloomConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        past_key_values: Cache
        | tuple[tuple[torch.Tensor, torch.Tensor], ...]
        | None = ...,
        attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        **deprecated_arguments,
    ) -> tuple[torch.Tensor] | TokenClassifierOutput: ...

@auto_docstring
class BloomForQuestionAnswering(BloomPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
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
    "BloomForCausalLM",
    "BloomForQuestionAnswering",
    "BloomForSequenceClassification",
    "BloomForTokenClassification",
    "BloomModel",
    "BloomPreTrainedModel",
]
