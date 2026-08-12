from typing import Any
import torch

from ..cohere.modeling_cohere import (
    CohereAttention,
    CohereDecoderLayer,
    CohereForCausalLM,
    CohereLayerNorm,
    CoherePreTrainedModel,
    CohereRotaryEmbedding,
)
from ..gemma2.modeling_gemma2 import Gemma2Model
from ...cache_utils import Cache
from ...configuration_utils import PretrainedConfig
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import BaseModelOutputWithPast
from ...processing_utils import Unpack
from ...utils import TransformersKwargs
from ...utils.deprecation import deprecate_kwarg

logger = ...

class Cohere2Config(PretrainedConfig):
    keys_to_ignore_at_inference = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        intermediate_size=...,
        logit_scale=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        num_key_value_heads=...,
        hidden_act=...,
        max_position_embeddings=...,
        initializer_range=...,
        layer_norm_eps=...,
        use_cache=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        tie_word_embeddings=...,
        rope_theta=...,
        rope_scaling=...,
        attention_bias=...,
        attention_dropout=...,
        sliding_window=...,
        layer_types=...,
        **kwargs,
    ) -> None: ...

class Cohere2RotaryEmbedding(CohereRotaryEmbedding): ...
class Cohere2LayerNorm(CohereLayerNorm): ...

class Cohere2Attention(CohereAttention):
    def __init__(self, config: Cohere2Config, layer_idx: int | None = ...) -> None: ...
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

class Cohere2DecoderLayer(CohereDecoderLayer):
    def __init__(self, config: Cohere2Config, layer_idx: int) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[
        torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None
    ]: ...

class Cohere2PreTrainedModel(CoherePreTrainedModel):
    config: Cohere2Config

class Cohere2Model(Gemma2Model):
    def __init__(self, config: Cohere2Config) -> None: ...
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
    def __call__(self, *args: Any, **kwargs: Any) -> BaseModelOutputWithPast: ...

class Cohere2ForCausalLM(CohereForCausalLM): ...

__all__ = [
    "Cohere2Config",
    "Cohere2ForCausalLM",
    "Cohere2Model",
    "Cohere2PreTrainedModel",
]
