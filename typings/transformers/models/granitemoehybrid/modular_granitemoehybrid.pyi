import torch

from .configuration_granitemoehybrid import GraniteMoeHybridConfig
from ..bamba.modeling_bamba import BambaMixer, BambaRMSNormGated
from ..granitemoeshared.modeling_granitemoeshared import (
    GraniteFlashAttentionKwargs,
    GraniteMoeSharedAttention,
    GraniteMoeSharedDecoderLayer,
    GraniteMoeSharedForCausalLM,
    GraniteMoeSharedMLP,
    GraniteMoeSharedModel,
    GraniteMoeSharedPreTrainedModel,
)
from ...cache_utils import Cache
from ...modeling_outputs import BaseModelOutputWithPast
from ...processing_utils import Unpack
from ...utils import auto_docstring, can_return_tuple
from ...utils.deprecation import deprecate_kwarg

logger = ...

class GraniteMoeHybridAttention(GraniteMoeSharedAttention):
    def __init__(self, config: GraniteMoeHybridConfig, layer_idx: int) -> None: ...

class GraniteMoeHybridMambaLayer(BambaMixer):
    def __init__(self, config: GraniteMoeHybridConfig, layer_idx: int) -> None: ...

class GraniteMoeHybridRMSNormGated(BambaRMSNormGated):
    def __init__(self, hidden_size, eps=...) -> None: ...

class GraniteMoeHybridMLP(GraniteMoeSharedMLP):
    def __init__(self, config: GraniteMoeHybridConfig) -> None: ...

class GraniteMoeHybridDecoderLayer(GraniteMoeSharedDecoderLayer):
    def __init__(self, config: GraniteMoeHybridConfig, layer_idx: int) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
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

class GraniteMoeHybridPreTrainedModel(GraniteMoeSharedPreTrainedModel):
    config: GraniteMoeHybridConfig
    _no_split_modules = ...
    _is_stateful = ...

class GraniteMoeHybridModel(GraniteMoeSharedModel):
    def __init__(self, config: GraniteMoeHybridConfig) -> None: ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | list[torch.FloatTensor] | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        output_router_logits: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[GraniteFlashAttentionKwargs],
    ) -> tuple | BaseModelOutputWithPast: ...

class GraniteMoeHybridForCausalLM(GraniteMoeSharedForCausalLM):
    _tied_weights_keys = ...
    def __init__(self, config: GraniteMoeHybridConfig) -> None: ...
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=...,
        attention_mask=...,
        inputs_embeds=...,
        cache_position=...,
        position_ids=...,
        use_cache=...,
        **kwargs,
    ):  # -> dict[str, Any]:
        ...

__all__ = [
    "GraniteMoeHybridForCausalLM",
    "GraniteMoeHybridModel",
    "GraniteMoeHybridPreTrainedModel",
]
