from typing import Any

from torch import nn

import torch

from ..gemma.modeling_gemma import (
    GemmaAttention,
    GemmaForCausalLM,
    GemmaForSequenceClassification,
    GemmaForTokenClassification,
    GemmaMLP,
    GemmaModel,
    GemmaPreTrainedModel,
    GemmaRMSNorm,
    GemmaRotaryEmbedding,
)
from ...cache_utils import Cache
from ...configuration_utils import PretrainedConfig
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...processing_utils import Unpack
from ...utils import TransformersKwargs
from ...utils.deprecation import deprecate_kwarg

logger = ...

class Gemma2Config(PretrainedConfig):
    keys_to_ignore_at_inference = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        intermediate_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        num_key_value_heads=...,
        head_dim=...,
        hidden_activation=...,
        max_position_embeddings=...,
        initializer_range=...,
        rms_norm_eps=...,
        use_cache=...,
        pad_token_id=...,
        eos_token_id=...,
        bos_token_id=...,
        tie_word_embeddings=...,
        rope_theta=...,
        attention_bias=...,
        attention_dropout=...,
        query_pre_attn_scalar=...,
        sliding_window=...,
        layer_types=...,
        final_logit_softcapping=...,
        attn_logit_softcapping=...,
        **kwargs,
    ) -> None: ...

class Gemma2RMSNorm(GemmaRMSNorm): ...

class Gemma2MLP(GemmaMLP):
    def __init__(self, config) -> None: ...

class Gemma2RotaryEmbedding(GemmaRotaryEmbedding): ...

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = ...,
    scaling: float | None = ...,
    softcap: float | None = ...,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]: ...

class Gemma2Attention(GemmaAttention):
    def __init__(self, config: Gemma2Config, layer_idx: int) -> None: ...
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

class Gemma2DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Gemma2Config, layer_idx: int) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        output_attentions: bool | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs,
    ) -> tuple[
        torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None
    ]: ...

class Gemma2PreTrainedModel(GemmaPreTrainedModel): ...

class Gemma2Model(GemmaModel):
    def __init__(self, config: Gemma2Config) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast: ...
    def __call__(self, *args: Any, **kwargs: Any) -> BaseModelOutputWithPast: ...

class Gemma2ForCausalLM(GemmaForCausalLM):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **kwargs,
    ) -> CausalLMOutputWithPast: ...
    def __call__(self, *args: Any, **kwargs: Any) -> CausalLMOutputWithPast: ...

class Gemma2ForSequenceClassification(GemmaForSequenceClassification): ...
class Gemma2ForTokenClassification(GemmaForTokenClassification): ...

__all__ = [
    "Gemma2Config",
    "Gemma2ForCausalLM",
    "Gemma2ForSequenceClassification",
    "Gemma2ForTokenClassification",
    "Gemma2Model",
    "Gemma2PreTrainedModel",
]
