from torch import nn

import torch

from .configuration_deepseek_v3 import DeepseekV3Config
from ..llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from ...cache_utils import Cache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import (
    GenericForSequenceClassification,
    GenericForTokenClassification,
)
from ...processing_utils import Unpack
from ...utils.deprecation import deprecate_kwarg

logger = ...

class DeepseekV3RMSNorm(LlamaRMSNorm): ...
class DeepseekV3RotaryEmbedding(LlamaRotaryEmbedding): ...

def apply_rotary_pos_emb_interleave(
    q, k, cos, sin, position_ids=..., unsqueeze_dim=...
):  # -> tuple[Any, Any]:
    ...
def yarn_get_mscale(scale=..., mscale=...):  # -> float:
    ...

class DeepseekV3MLP(nn.Module):
    def __init__(self, config, hidden_size=..., intermediate_size=...) -> None: ...
    def forward(self, x):  # -> Any:
        ...

class DeepseekV3TopkRouter(nn.Module):
    def __init__(self, config) -> None: ...
    @torch.no_grad()
    def get_topk_indices(self, scores):  # -> Tensor:
        ...
    def forward(self, hidden_states):  # -> tuple[Tensor, Any]:
        ...

class DeepseekV3MoE(nn.Module):
    def __init__(self, config) -> None: ...
    def moe(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ):  # -> Tensor:
        ...
    def forward(self, hidden_states):  # -> Any:
        ...

class DeepseekV3Attention(nn.Module):
    def __init__(self, config: DeepseekV3Config, layer_idx: int) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class DeepseekV3DecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: DeepseekV3Config, layer_idx: int) -> None: ...

class DeepseekV3PreTrainedModel(LlamaPreTrainedModel):
    _can_compile_fullgraph = ...

class DeepseekV3Model(LlamaModel):
    _keys_to_ignore_on_load_unexpected = ...

class DeepseekV3ForCausalLM(LlamaForCausalLM): ...
class DeepseekV3ForSequenceClassification(
    GenericForSequenceClassification, DeepseekV3PreTrainedModel
): ...
class DeepseekV3ForTokenClassification(
    GenericForTokenClassification, DeepseekV3PreTrainedModel
): ...

__all__ = [
    "DeepseekV3ForCausalLM",
    "DeepseekV3ForSequenceClassification",
    "DeepseekV3ForTokenClassification",
    "DeepseekV3Model",
    "DeepseekV3PreTrainedModel",
]
