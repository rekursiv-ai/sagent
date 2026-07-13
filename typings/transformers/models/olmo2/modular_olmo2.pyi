from transformers.utils.generic import TransformersKwargs

import torch

from ..llama.modeling_llama import LlamaPreTrainedModel, LlamaRMSNorm
from ..olmo.configuration_olmo import OlmoConfig
from ..olmo.modeling_olmo import (
    OlmoAttention,
    OlmoDecoderLayer,
    OlmoForCausalLM,
    OlmoModel,
    OlmoRotaryEmbedding,
)
from ...cache_utils import Cache
from ...processing_utils import Unpack
from ...utils.deprecation import deprecate_kwarg

logger = ...

class Olmo2Config(OlmoConfig):
    model_type = ...
    base_model_tp_plan = ...
    base_model_pp_plan = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        intermediate_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        num_key_value_heads=...,
        hidden_act=...,
        max_position_embeddings=...,
        initializer_range=...,
        use_cache=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        tie_word_embeddings=...,
        rope_theta=...,
        rope_scaling=...,
        attention_bias=...,
        attention_dropout=...,
        rms_norm_eps=...,
        **kwargs,
    ) -> None: ...

class Olmo2RMSNorm(LlamaRMSNorm):
    def forward(self, hidden_states): ...

def rotate_half(x):  # -> Tensor:
    ...

class Olmo2Attention(OlmoAttention):
    def __init__(self, config: Olmo2Config, layer_idx: int | None = ...) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class Olmo2DecoderLayer(OlmoDecoderLayer):
    def __init__(self, config: Olmo2Config, layer_idx: int) -> None: ...
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
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor: ...

class Olmo2RotaryEmbedding(OlmoRotaryEmbedding): ...
class Olmo2PreTrainedModel(LlamaPreTrainedModel): ...

class Olmo2Model(OlmoModel):
    def __init__(self, config: Olmo2Config) -> None: ...

class Olmo2ForCausalLM(OlmoForCausalLM): ...

__all__ = ["Olmo2Config", "Olmo2ForCausalLM", "Olmo2Model", "Olmo2PreTrainedModel"]
