from torch import nn

import torch

from .configuration_seed_oss import SeedOssConfig
from ..llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaForQuestionAnswering,
    LlamaForSequenceClassification,
    LlamaForTokenClassification,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
)
from ...cache_utils import Cache
from ...modeling_outputs import CausalLMOutputWithPast
from ...processing_utils import Unpack
from ...utils import TransformersKwargs
from ...utils.deprecation import deprecate_kwarg

"""PyTorch SeedOss model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...

class SeedOssRMSNorm(LlamaRMSNorm): ...

class SeedOssMLP(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, x):  # -> Tensor:
        ...

class SeedOssAttention(nn.Module):
    def __init__(self, config: SeedOssConfig, layer_idx: int) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

class SeedOssDecoderLayer(LlamaDecoderLayer): ...
class SeedOssPreTrainedModel(LlamaPreTrainedModel): ...
class SeedOssModel(LlamaModel): ...

class SeedOssForCausalLM(LlamaForCausalLM):
    def forward(
        self, **super_kwargs: Unpack[TransformersKwargs]
    ) -> CausalLMOutputWithPast: ...

class SeedOssForSequenceClassification(LlamaForSequenceClassification): ...
class SeedOssForTokenClassification(LlamaForTokenClassification): ...
class SeedOssForQuestionAnswering(LlamaForQuestionAnswering): ...

__all__ = [
    "SeedOssForCausalLM",
    "SeedOssForQuestionAnswering",
    "SeedOssForSequenceClassification",
    "SeedOssForTokenClassification",
    "SeedOssModel",
    "SeedOssPreTrainedModel",
]
