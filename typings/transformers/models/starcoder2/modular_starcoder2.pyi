from typing import Any

from torch import nn
from transformers.utils.generic import check_model_inputs

import torch

from .configuration_starcoder2 import Starcoder2Config
from ..mistral.modeling_mistral import (
    MistralAttention,
    MistralDecoderLayer,
    MistralForCausalLM,
    MistralForSequenceClassification,
    MistralForTokenClassification,
    MistralModel,
    MistralRotaryEmbedding,
)
from ...cache_utils import Cache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import BaseModelOutputWithPast
from ...processing_utils import Unpack
from ...utils import TransformersKwargs
from ...utils.deprecation import deprecate_kwarg

"""PyTorch Starcoder2 model."""
logger = ...

class Starcoder2MLP(nn.Module):
    def __init__(self, config: Starcoder2Config) -> None: ...
    def forward(
        self, hidden_states: tuple[torch.FloatTensor] | None
    ) -> torch.FloatTensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.FloatTensor: ...

class Starcoder2Attention(MistralAttention):
    def __init__(
        self, config: Starcoder2Config, layer_idx: int | None = ...
    ) -> None: ...
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

class Starcoder2DecoderLayer(MistralDecoderLayer):
    def __init__(self, config: Starcoder2Config, layer_idx: int) -> None: ...

class Starcoder2RotaryEmbedding(MistralRotaryEmbedding): ...

class Starcoder2Model(MistralModel):
    def __init__(self, config: Starcoder2Config) -> None: ...
    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | list[torch.FloatTensor] | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast: ...

class Starcoder2ForCausalLM(MistralForCausalLM): ...
class Starcoder2ForSequenceClassification(MistralForSequenceClassification): ...
class Starcoder2ForTokenClassification(MistralForTokenClassification): ...

__all__ = [
    "Starcoder2ForCausalLM",
    "Starcoder2ForSequenceClassification",
    "Starcoder2ForTokenClassification",
    "Starcoder2Model",
    "Starcoder2PreTrainedModel",
]
