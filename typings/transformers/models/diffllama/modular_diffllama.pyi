from torch import nn

import torch

from .configuration_diffllama import DiffLlamaConfig
from ..gemma.modeling_gemma import GemmaForCausalLM
from ..llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaForQuestionAnswering,
    LlamaForSequenceClassification,
    LlamaForTokenClassification,
    LlamaModel,
    LlamaPreTrainedModel,
)
from ..mistral.modeling_mistral import MistralMLP
from ...cache_utils import Cache
from ...utils.deprecation import deprecate_kwarg

logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...

class DiffLlamaMLP(MistralMLP): ...

def lambda_init_fn(layer_idx):  # -> float:
    ...

class DiffLlamaAttention(nn.Module):
    def __init__(
        self, config: DiffLlamaConfig, layer_idx: int | None = ...
    ) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class DiffLlamaFlashAttention2(DiffLlamaAttention):
    def __init__(self, *args, **kwargs) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool = ...,
        cache_position: torch.LongTensor | None = ...,
    ) -> tuple[torch.Tensor, None]: ...

class DiffLlamaSdpaAttention(DiffLlamaAttention):
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

DIFFLLAMA_ATTENTION_CLASSES = ...

class DiffLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: DiffLlamaConfig, layer_idx: int) -> None: ...

class DiffLlamaPreTrainedModel(LlamaPreTrainedModel):
    _supports_flex_attn = ...
    _supports_attention_backend = ...

class DiffLlamaModel(LlamaModel): ...
class DiffLlamaForCausalLM(GemmaForCausalLM): ...
class DiffLlamaForSequenceClassification(LlamaForSequenceClassification): ...
class DiffLlamaForQuestionAnswering(LlamaForQuestionAnswering): ...
class DiffLlamaForTokenClassification(LlamaForTokenClassification): ...

__all__ = [
    "DiffLlamaForCausalLM",
    "DiffLlamaForQuestionAnswering",
    "DiffLlamaForSequenceClassification",
    "DiffLlamaForTokenClassification",
    "DiffLlamaModel",
    "DiffLlamaPreTrainedModel",
]
