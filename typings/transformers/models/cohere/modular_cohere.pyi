from typing import Any

from torch import nn

import torch

from .configuration_cohere import CohereConfig
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaModel,
    LlamaRotaryEmbedding,
)
from ...cache_utils import Cache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import CausalLMOutputWithPast
from ...modeling_rope_utils import dynamic_rope_update
from ...processing_utils import Unpack
from ...utils import TransformersKwargs
from ...utils.deprecation import deprecate_kwarg

"""PyTorch Cohere model."""
logger = ...

class CohereLayerNorm(nn.Module):
    def __init__(self, hidden_size=..., eps=..., bias=...) -> None: ...
    def forward(self, hidden_states): ...

class CohereRotaryEmbedding(LlamaRotaryEmbedding):
    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):  # -> tuple[Tensor, Tensor]:
        ...

def rotate_half(x):  # -> Tensor:
    ...
def apply_rotary_pos_emb(
    q, k, cos, sin, position_ids=..., unsqueeze_dim=...
):  # -> tuple[Any, Any]:
    ...

class CohereMLP(LlamaMLP):
    def __init__(self, config) -> None: ...

class CohereAttention(LlamaAttention):
    def __init__(self, config: CohereConfig, layer_idx: int | None = ...) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class CohereDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: CohereConfig, layer_idx: int) -> None: ...
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
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[
        torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None
    ]: ...

class CohereModel(LlamaModel):
    def __init__(self, config: CohereConfig) -> None: ...

class CohereForCausalLM(LlamaForCausalLM):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | list[torch.FloatTensor] | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast: ...
    def __call__(self, *args: Any, **kwargs: Any) -> CausalLMOutputWithPast: ...

__all__ = ["CohereForCausalLM", "CohereModel", "CoherePreTrainedModel"]
