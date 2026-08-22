from typing import Any

from transformers.utils.generic import TransformersKwargs

import torch

from ..olmo2.configuration_olmo2 import Olmo2Config
from ..olmo2.modeling_olmo2 import (
    Olmo2Attention,
    Olmo2DecoderLayer,
    Olmo2ForCausalLM,
    Olmo2Model,
    Olmo2PreTrainedModel,
    Olmo2RMSNorm,
    Olmo2RotaryEmbedding,
)
from ...cache_utils import Cache
from ...modeling_outputs import BaseModelOutputWithPast
from ...processing_utils import Unpack

class Olmo3Config(Olmo2Config):
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
        sliding_window=...,
        layer_types=...,
        **kwargs,
    ) -> None: ...

class Olmo3RMSNorm(Olmo2RMSNorm): ...

class Olmo3Attention(Olmo2Attention):
    def __init__(self, config: Olmo3Config, layer_idx: int) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class Olmo3DecoderLayer(Olmo2DecoderLayer): ...

class Olmo3RotaryEmbedding(Olmo2RotaryEmbedding):
    def __init__(
        self, config: Olmo3Config, device=..., rope_type: str | None = ...
    ) -> None: ...

class Olmo3PreTrainedModel(Olmo2PreTrainedModel): ...

class Olmo3Model(Olmo2Model):
    def __init__(self, config: Olmo3Config) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        cache_position: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast: ...
    def __call__(self, *args: Any, **kwargs: Any) -> BaseModelOutputWithPast: ...

class Olmo3ForCausalLM(Olmo2ForCausalLM): ...

__all__ = ["Olmo3Config", "Olmo3ForCausalLM", "Olmo3Model", "Olmo3PreTrainedModel"]
