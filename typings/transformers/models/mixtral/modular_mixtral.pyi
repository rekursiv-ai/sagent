from torch import nn

import torch

from .configuration_mixtral import MixtralConfig
from ..mistral.modeling_mistral import (
    MistralAttention,
    MistralForCausalLM,
    MistralForQuestionAnswering,
    MistralForSequenceClassification,
    MistralForTokenClassification,
    MistralModel,
    MistralPreTrainedModel,
    MistralRMSNorm,
    MistralRotaryEmbedding,
)
from ...cache_utils import Cache
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from ...processing_utils import Unpack
from ...utils import TransformersKwargs
from ...utils.deprecation import deprecate_kwarg

"""PyTorch Mixtral model."""
logger = ...

def load_balancing_loss_func(
    gate_logits: torch.Tensor | tuple[torch.Tensor] | None,
    num_experts: int | None = ...,
    top_k=...,
    attention_mask: torch.Tensor | None = ...,
) -> torch.Tensor | int: ...

class MixtralBlockSparseTop2MLP(nn.Module):
    def __init__(self, config: MixtralConfig) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class MixtralSparseMoeBlock(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class MixtralRMSNorm(MistralRMSNorm): ...
class MixtralAttention(MistralAttention): ...

class MixtralDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: MixtralConfig, layer_idx: int) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.FloatTensor: ...

class MixtralRotaryEmbedding(MistralRotaryEmbedding): ...

class MixtralPreTrainedModel(MistralPreTrainedModel):
    _can_compile_fullgraph = ...
    _can_record_outputs = ...

class MixtralModel(MistralModel):
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

class MixtralForCausalLM(MistralForCausalLM):
    _tied_weights_keys = ...
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
        output_router_logits: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeCausalLMOutputWithPast: ...

class MixtralForSequenceClassification(MistralForSequenceClassification): ...
class MixtralForTokenClassification(MistralForTokenClassification): ...
class MixtralForQuestionAnswering(MistralForQuestionAnswering): ...

__all__ = [
    "MixtralForCausalLM",
    "MixtralForQuestionAnswering",
    "MixtralForSequenceClassification",
    "MixtralForTokenClassification",
    "MixtralModel",
    "MixtralPreTrainedModel",
]
