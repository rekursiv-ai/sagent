from typing import Any
from torch import nn
from transformers.utils.generic import check_model_inputs

import torch

from ..llama.modeling_llama import (
    LlamaForCausalLM,
    LlamaForQuestionAnswering,
    LlamaForSequenceClassification,
    LlamaForTokenClassification,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from ..olmo2.modeling_olmo2 import Olmo2DecoderLayer, Olmo2MLP
from ...cache_utils import Cache
from ...configuration_utils import PretrainedConfig
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...processing_utils import Unpack
from ...utils import TransformersKwargs
from ...utils.deprecation import deprecate_kwarg

"""LG AI Research EXAONE Lab"""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...

class Exaone4Config(PretrainedConfig):
    keys_to_ignore_at_inference = ...
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
        rms_norm_eps=...,
        use_cache=...,
        bos_token_id=...,
        eos_token_id=...,
        tie_word_embeddings=...,
        rope_theta=...,
        rope_scaling=...,
        attention_dropout=...,
        sliding_window=...,
        sliding_window_pattern=...,
        layer_types=...,
        **kwargs,
    ) -> None: ...

class Exaone4RMSNorm(LlamaRMSNorm): ...
class Exaone4RotaryEmbedding(LlamaRotaryEmbedding): ...

class Exaone4Attention(nn.Module):
    def __init__(self, config: Exaone4Config, layer_idx: int) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class Exaone4MLP(Olmo2MLP): ...
class Exaone4DecoderLayer(Olmo2DecoderLayer): ...

class Exaone4PreTrainedModel(LlamaPreTrainedModel):
    config_class = Exaone4Config
    _no_split_modules = ...

class Exaone4Model(Exaone4PreTrainedModel, LlamaModel):
    def __init__(self, config: Exaone4Config) -> None: ...
    @check_model_inputs
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
    ) -> tuple | BaseModelOutputWithPast: ...

class Exaone4ForCausalLM(LlamaForCausalLM):
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast: ...
    def __call__(self, *args: Any, **kwargs: Any) -> CausalLMOutputWithPast: ...

class Exaone4ForSequenceClassification(LlamaForSequenceClassification): ...
class Exaone4ForTokenClassification(LlamaForTokenClassification): ...
class Exaone4ForQuestionAnswering(LlamaForQuestionAnswering): ...

__all__ = [
    "Exaone4Config",
    "Exaone4ForCausalLM",
    "Exaone4ForQuestionAnswering",
    "Exaone4ForSequenceClassification",
    "Exaone4ForTokenClassification",
    "Exaone4Model",
    "Exaone4PreTrainedModel",
]
