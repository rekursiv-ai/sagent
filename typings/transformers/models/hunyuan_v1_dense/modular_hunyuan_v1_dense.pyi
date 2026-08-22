from typing import Any

from torch import nn
from transformers.cache_utils import Cache

import torch

from .configuration_hunyuan_v1_dense import HunYuanDenseV1Config
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaForSequenceClassification,
    LlamaMLP,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
)
from ...modeling_rope_utils import dynamic_rope_update
from ...processing_utils import Unpack
from ...utils import TransformersKwargs

"""PyTorch HunYuanDenseV1 model."""
logger = ...

class HunYuanDenseV1RMSNorm(LlamaRMSNorm): ...

class HunYuanDenseV1MLP(LlamaMLP):
    def __init__(
        self, config: HunYuanDenseV1Config, layer_idx=..., is_shared_mlp=...
    ) -> None: ...

class HunYuanDenseV1Attention(LlamaAttention):
    def __init__(self, config: HunYuanDenseV1Config, layer_idx: int) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

class HunYuanDenseV1DecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: HunYuanDenseV1Config, layer_idx: int) -> None: ...

class HunYuanDenseV1PreTrainedModel(LlamaPreTrainedModel): ...

class HunYuanDenseV1RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor
    def __init__(self, config: HunYuanDenseV1Config, device=...) -> None: ...
    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):  # -> tuple[Tensor, Tensor]:
        ...

class HunYuanDenseV1Model(LlamaModel): ...
class HunYuanDenseV1ForCausalLM(LlamaForCausalLM): ...
class HunYuanDenseV1ForSequenceClassification(LlamaForSequenceClassification): ...

__all__ = [
    "HunYuanDenseV1ForCausalLM",
    "HunYuanDenseV1ForSequenceClassification",
    "HunYuanDenseV1Model",
    "HunYuanDenseV1PreTrainedModel",
]
