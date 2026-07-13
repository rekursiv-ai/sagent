from torch import nn

import torch

from .configuration_phi3 import Phi3Config
from ..mistral.modeling_mistral import (
    MistralDecoderLayer,
    MistralForCausalLM,
    MistralForSequenceClassification,
    MistralForTokenClassification,
    MistralPreTrainedModel,
)
from ...cache_utils import Cache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...processing_utils import Unpack
from ...utils.deprecation import deprecate_kwarg

"""PyTorch Phi-3 model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...

class Phi3MLP(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor: ...

def apply_rotary_pos_emb(
    q, k, cos, sin, position_ids=..., unsqueeze_dim=...
):  # -> tuple[Tensor, Tensor]:
    ...

class Phi3Attention(nn.Module):
    def __init__(self, config: Phi3Config, layer_idx: int | None = ...) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class Phi3DecoderLayer(MistralDecoderLayer):
    def __init__(self, config: Phi3Config, layer_idx: int) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[
        torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None
    ]: ...

class Phi3PreTrainedModel(MistralPreTrainedModel):
    _version = ...

class Phi3ForCausalLM(MistralForCausalLM):
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=...,
        attention_mask=...,
        inputs_embeds=...,
        cache_position=...,
        position_ids=...,
        use_cache=...,
        logits_to_keep=...,
        **kwargs,
    ):  # -> dict[Any, Any]:
        ...

class Phi3ForSequenceClassification(MistralForSequenceClassification): ...
class Phi3ForTokenClassification(MistralForTokenClassification): ...

__all__ = [
    "Phi3ForCausalLM",
    "Phi3ForSequenceClassification",
    "Phi3ForTokenClassification",
    "Phi3Model",
    "Phi3PreTrainedModel",
]
