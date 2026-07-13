from typing import TypedDict

from torch import nn

import torch

from .configuration_granitemoeshared import GraniteMoeSharedConfig
from ..granitemoe.modeling_granitemoe import (
    GraniteMoeDecoderLayer,
    GraniteMoeForCausalLM,
    GraniteMoeModel,
    GraniteMoePreTrainedModel,
)
from ...cache_utils import Cache
from ...processing_utils import Unpack
from ...utils.deprecation import deprecate_kwarg

logger = ...

class GraniteFlashAttentionKwargs(TypedDict, total=False):
    cu_seq_lens_q: torch.LongTensor
    cu_seq_lens_k: torch.LongTensor
    max_length_q: int
    max_length_k: int
    seq_idx: torch.IntTensor

class GraniteMoeSharedMLP(nn.Module):
    def __init__(self, config: GraniteMoeSharedConfig) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class GraniteMoeSharedDecoderLayer(GraniteMoeDecoderLayer):
    def __init__(self, config: GraniteMoeSharedConfig, layer_idx: int) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        output_attentions: bool | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        output_router_logits: bool | None = ...,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = ...,
        **kwargs: Unpack[GraniteFlashAttentionKwargs],
    ) -> tuple[
        torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None
    ]: ...

class GraniteMoeSharedPreTrainedModel(GraniteMoePreTrainedModel):
    config: GraniteMoeSharedConfig
    _no_split_modules = ...

class GraniteMoeSharedModel(GraniteMoeModel):
    def __init__(self, config: GraniteMoeSharedConfig) -> None: ...

class GraniteMoeSharedForCausalLM(GraniteMoeForCausalLM):
    _tied_weights_keys = ...
    def __init__(self, config: GraniteMoeSharedConfig) -> None: ...

__all__ = [
    "GraniteMoeSharedForCausalLM",
    "GraniteMoeSharedModel",
    "GraniteMoeSharedPreTrainedModel",
]
