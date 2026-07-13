from torch import nn

import torch

from .configuration_qwen3_moe import Qwen3MoeConfig
from ..llama.modeling_llama import (
    LlamaForQuestionAnswering,
    LlamaForSequenceClassification,
    LlamaForTokenClassification,
    LlamaRMSNorm,
)
from ..mixtral.modeling_mixtral import MixtralForCausalLM, MixtralModel
from ..qwen2_moe.modeling_qwen2_moe import Qwen2MoeDecoderLayer
from ..qwen3.modeling_qwen3 import Qwen3Attention
from ...cache_utils import Cache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import MoeCausalLMOutputWithPast
from ...processing_utils import Unpack
from ...utils import TransformersKwargs
from ...utils.deprecation import deprecate_kwarg

"""PyTorch Qwen3 model."""
logger = ...

class Qwen3MoeAttention(Qwen3Attention):
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int) -> None: ...

class Qwen3MoeMLP(nn.Module):
    def __init__(self, config, intermediate_size=...) -> None: ...
    def forward(self, x):  # -> Any:
        ...

class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class Qwen3MoeRMSNorm(LlamaRMSNorm): ...

class Qwen3MoeDecoderLayer(Qwen2MoeDecoderLayer):
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.FloatTensor: ...

class Qwen3MoeModel(MixtralModel): ...

class Qwen3MoeForCausalLM(MixtralForCausalLM):
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

class Qwen3MoeForSequenceClassification(LlamaForSequenceClassification): ...
class Qwen3MoeForTokenClassification(LlamaForTokenClassification): ...
class Qwen3MoeForQuestionAnswering(LlamaForQuestionAnswering): ...

__all__ = [
    "Qwen3MoeForCausalLM",
    "Qwen3MoeForQuestionAnswering",
    "Qwen3MoeForSequenceClassification",
    "Qwen3MoeForTokenClassification",
    "Qwen3MoeModel",
    "Qwen3MoePreTrainedModel",
]
