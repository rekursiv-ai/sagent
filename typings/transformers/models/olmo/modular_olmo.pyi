from torch import nn

import torch

from .configuration_olmo import OlmoConfig
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaModel,
    LlamaRotaryEmbedding,
)
from ...cache_utils import Cache
from ...utils.deprecation import deprecate_kwarg

logger = ...

class OlmoLayerNorm(nn.Module):
    def __init__(self, hidden_size: int) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class OlmoMLP(LlamaMLP):
    def __init__(self, config) -> None: ...

def apply_rotary_pos_emb(
    q, k, cos, sin, position_ids=..., unsqueeze_dim=...
):  # -> tuple[Any, Any]:
    ...

class OlmoAttention(LlamaAttention):
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

class OlmoDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: OlmoConfig, layer_idx: int) -> None: ...

class OlmoRotaryEmbedding(LlamaRotaryEmbedding):
    def forward(self, x, position_ids):  # -> tuple[Tensor, Tensor]:
        ...

class OlmoModel(LlamaModel):
    def __init__(self, config: OlmoConfig) -> None: ...

class OlmoForCausalLM(LlamaForCausalLM): ...

__all__ = ["OlmoForCausalLM", "OlmoModel", "OlmoPreTrainedModel"]
