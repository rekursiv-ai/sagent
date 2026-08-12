from typing import Any
from torch import nn

import torch

from .configuration_cpmant import CpmAntConfig
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring

"""PyTorch CPMAnt"""
logger = ...

class CpmAntLayerNorm(nn.Module):
    def __init__(self, config: CpmAntConfig) -> None: ...
    def forward(self, hidden_states: torch.Tensor):  # -> Tensor:
        ...

class CpmAntAttention(nn.Module):
    def __init__(self, config: CpmAntConfig, layer_idx=...) -> None: ...
    def forward(
        self,
        hidden_q: torch.Tensor,
        hidden_kv: torch.Tensor,
        attention_mask: torch.BoolTensor,
        position_bias: torch.Tensor,
        output_attentions: bool | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ):  # -> tuple[Any, Tensor | None]:
        ...

class CpmAntSelfAttentionBlock(nn.Module):
    def __init__(self, config: CpmAntConfig, layer_idx=...) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_bias: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ):  # -> tuple[Tensor, Any]:
        ...

class CpmAntDenseGatedACT(nn.Module):
    def __init__(self, config: CpmAntConfig) -> None: ...
    def forward(self, hidden_states: torch.Tensor):  # -> Tensor:
        ...

class CpmAntFeedForward(nn.Module):
    def __init__(self, config: CpmAntConfig) -> None: ...
    def forward(self, hidden_states: torch.Tensor):  # -> Tensor:
        ...

class CpmAntFFNBlock(nn.Module):
    def __init__(self, config: CpmAntConfig) -> None: ...
    def forward(self, hidden_states: torch.Tensor):  # -> Tensor:
        ...

class CpmAntTransformerBlock(nn.Module):
    def __init__(self, config: CpmAntConfig, layer_idx=...) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_bias: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ):  # -> tuple[Tensor, Any]:
        ...

class CpmAntEncoder(nn.Module):
    def __init__(self, config: CpmAntConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_bias: torch.Tensor,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ):  # -> tuple[Tensor, tuple[Tensor, ...] | Any | tuple[()] | None, tuple[()] | tuple[Any, ...] | None]:
        ...

class CpmAntIntermediate(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class CpmAntSegmentPositionEmbedding(nn.Module):
    def __init__(self, config: CpmAntConfig) -> None: ...
    def forward(
        self,
        key_pos: torch.Tensor,
        query_pos: torch.Tensor,
        key_segment: torch.Tensor,
        query_segment: torch.Tensor,
    ):  # -> Tensor:
        ...

class CpmAntOutput(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

@auto_docstring
class CpmAntPreTrainedModel(PreTrainedModel):
    config: CpmAntConfig
    base_model_prefix = ...

@auto_docstring
class CpmAntModel(CpmAntPreTrainedModel):
    def __init__(self, config: CpmAntConfig) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, embeddings, **kwargs):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor] | BaseModelOutputWithPast: ...

@auto_docstring(custom_intro=...)
class CpmAntForCausalLM(CpmAntPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    def __init__(self, config: CpmAntConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        labels: torch.Tensor | None = ...,
        return_dict: bool | None = ...,
        attention_mask: torch.Tensor | None = ...,
        cache_position: torch.Tensor | None = ...,
        **kwargs,
    ) -> tuple | CausalLMOutputWithPast: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, embeddings):  # -> None:
        ...

__all__ = ["CpmAntForCausalLM", "CpmAntModel", "CpmAntPreTrainedModel"]
