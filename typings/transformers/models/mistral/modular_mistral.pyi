from transformers.utils.generic import check_model_inputs

import torch

from .configuration_mistral import MistralConfig
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaForSequenceClassification,
    LlamaForTokenClassification,
    LlamaMLP,
    LlamaModel,
    LlamaPreTrainedModel,
)
from ...cache_utils import Cache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GenericForQuestionAnswering
from ...modeling_outputs import BaseModelOutputWithPast
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.deprecation import deprecate_kwarg

logger = ...

class MistralMLP(LlamaMLP):
    def __init__(self, config) -> None: ...

class MistralAttention(LlamaAttention):
    def __init__(self, config: MistralConfig, layer_idx: int) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class MistralDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: MistralConfig, layer_idx: int) -> None: ...

class MistralPreTrainedModel(LlamaPreTrainedModel):
    _can_record_outputs = ...

class MistralModel(LlamaModel):
    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast: ...

class MistralForCausalLM(LlamaForCausalLM): ...
class MistralForTokenClassification(LlamaForTokenClassification): ...
class MistralForSequenceClassification(LlamaForSequenceClassification): ...
class MistralForQuestionAnswering(
    GenericForQuestionAnswering, MistralPreTrainedModel
): ...

__all__ = [
    "MistralForCausalLM",
    "MistralForQuestionAnswering",
    "MistralForSequenceClassification",
    "MistralForTokenClassification",
    "MistralModel",
    "MistralPreTrainedModel",
]
