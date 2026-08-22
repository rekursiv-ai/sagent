from typing import Any

from torch import nn
from transformers.cache_utils import Cache

import torch

from .configuration_hunyuan_v1_moe import HunYuanMoEV1Config
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

"""PyTorch HunYuanMoEV1 model."""
logger = ...

class HunYuanMoEV1RMSNorm(LlamaRMSNorm): ...

class HunYuanMoEV1MLP(LlamaMLP):
    def __init__(
        self, config: HunYuanMoEV1Config, layer_idx=..., is_shared_mlp=...
    ) -> None: ...

class HunYuanMoEV1Attention(LlamaAttention):
    def __init__(self, config: HunYuanMoEV1Config, layer_idx: int) -> None: ...
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

class HunYuanMoEV1Gate(nn.Module):
    def __init__(
        self, config: HunYuanMoEV1Config, layer_idx: int | None = ...
    ) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class HunYuanMoEV1Moe(nn.Module):
    def __init__(
        self, config: HunYuanMoEV1Config, layer_idx: int | None = ...
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class HunYuanMoEV1DecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: HunYuanMoEV1Config, layer_idx: int) -> None: ...

class HunYuanMoEV1PreTrainedModel(LlamaPreTrainedModel):
    _can_compile_fullgraph = ...

class HunYuanMoEV1RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor
    def __init__(self, config: HunYuanMoEV1Config, device=...) -> None: ...
    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):  # -> tuple[Tensor, Tensor]:
        ...

class HunYuanMoEV1Model(LlamaModel): ...
class HunYuanMoEV1ForCausalLM(LlamaForCausalLM): ...
class HunYuanMoEV1ForSequenceClassification(LlamaForSequenceClassification): ...

__all__ = [
    "HunYuanMoEV1ForCausalLM",
    "HunYuanMoEV1ForSequenceClassification",
    "HunYuanMoEV1Model",
    "HunYuanMoEV1PreTrainedModel",
]
