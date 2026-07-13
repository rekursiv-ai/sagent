import torch

from .configuration_phi import PhiConfig
from ..clip.modeling_clip import CLIPMLP
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaForCausalLM,
    LlamaForSequenceClassification,
    LlamaForTokenClassification,
    LlamaModel,
    LlamaRotaryEmbedding,
)
from ...cache_utils import Cache
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast
from ...processing_utils import Unpack
from ...utils import TransformersKwargs
from ...utils.deprecation import deprecate_kwarg

logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...

class PhiAttention(LlamaAttention):
    def __init__(self, config: PhiConfig, layer_idx: int) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class PhiMLP(CLIPMLP): ...

class PhiDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: PhiConfig, layer_idx: int) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        output_attentions: bool | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = ...,
        **kwargs,
    ) -> tuple[
        torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None
    ]: ...

class PhiRotaryEmbedding(LlamaRotaryEmbedding): ...

class PhiModel(LlamaModel):
    def __init__(self, config: PhiConfig) -> None: ...
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

class PhiForCausalLM(LlamaForCausalLM):
    def __init__(self, config) -> None: ...

class PhiForSequenceClassification(LlamaForSequenceClassification): ...
class PhiForTokenClassification(LlamaForTokenClassification): ...

__all__ = [
    "PhiForCausalLM",
    "PhiForSequenceClassification",
    "PhiForTokenClassification",
    "PhiModel",
    "PhiPreTrainedModel",
]
