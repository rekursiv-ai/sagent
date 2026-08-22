from typing import Any

import torch

from ..mixtral.modeling_mixtral import MixtralModel, MixtralPreTrainedModel
from ..olmo2.modeling_olmo2 import Olmo2Attention, Olmo2RMSNorm, Olmo2RotaryEmbedding
from ..olmoe.configuration_olmoe import OlmoeConfig
from ..olmoe.modeling_olmoe import (
    OlmoeDecoderLayer,
    OlmoeForCausalLM,
    OlmoeMLP,
    OlmoeSparseMoeBlock,
)
from ...cache_utils import Cache
from ...modeling_outputs import MoeModelOutputWithPast
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import check_model_inputs

class FlexOlmoConfig(OlmoeConfig):
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
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        tie_word_embeddings=...,
        rope_theta=...,
        rope_scaling=...,
        attention_bias=...,
        attention_dropout=...,
        num_experts_per_tok=...,
        num_experts=...,
        output_router_logits=...,
        router_aux_loss_coef=...,
        norm_topk_prob=...,
        **kwargs,
    ) -> None: ...

class FlexOlmoRMSNorm(Olmo2RMSNorm): ...
class FlexOlmoRotaryEmbedding(Olmo2RotaryEmbedding): ...
class FlexOlmoMLP(OlmoeMLP): ...
class FlexOlmoAttention(Olmo2Attention): ...
class FlexOlmoSparseMoeBlock(OlmoeSparseMoeBlock): ...

class FlexOlmoDecoderLayer(OlmoeDecoderLayer):
    def __init__(self, config: FlexOlmoConfig, layer_idx: int) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = ...,
        **kwargs,
    ) -> torch.FloatTensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.FloatTensor: ...

class FlexOlmoPreTrainedModel(MixtralPreTrainedModel): ...

class FlexOlmoModel(MixtralModel):
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
    ) -> MoeModelOutputWithPast: ...

class FlexOlmoForCausalLM(OlmoeForCausalLM): ...

__all__ = [
    "FlexOlmoConfig",
    "FlexOlmoForCausalLM",
    "FlexOlmoModel",
    "FlexOlmoPreTrainedModel",
]
