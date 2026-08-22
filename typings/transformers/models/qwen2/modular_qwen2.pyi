from typing import Any

from packaging import version
from torch import nn

import torch

from .configuration_qwen2 import Qwen2Config
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaForQuestionAnswering,
    LlamaForSequenceClassification,
    LlamaForTokenClassification,
    LlamaMLP,
    LlamaPreTrainedModel,
)
from ..mistral.modeling_mistral import MistralModel
from ...cache_utils import Cache
from ...integrations import use_kernel_forward_from_hub
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import BaseModelOutputWithPast
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import check_model_inputs
from ...utils.import_utils import get_torch_version

logger = ...

class Qwen2MLP(LlamaMLP):
    def __init__(self, config) -> None: ...

class Qwen2Attention(LlamaAttention):
    def __init__(self, config: Qwen2Config, layer_idx: int) -> None: ...
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

if version.parse(get_torch_version()) >= version.parse("2.3.0"):
    class Qwen2RMSNorm(nn.RMSNorm):
        def __init__(self, hidden_size, eps: float = ...) -> None: ...

else:
    @use_kernel_forward_from_hub("RMSNorm")
    class Qwen2RMSNorm(nn.Module):
        def __init__(self, hidden_size, eps: float = ...) -> None: ...
        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
        def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...
        def extra_repr(self):  # -> str:
            ...

class Qwen2DecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: Qwen2Config, layer_idx: int) -> None: ...

class Qwen2PreTrainedModel(LlamaPreTrainedModel): ...

class Qwen2Model(MistralModel):
    def __init__(self, config: Qwen2Config) -> None: ...
    @check_model_inputs
    @auto_docstring
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
    ) -> BaseModelOutputWithPast: ...

class Qwen2ForCausalLM(LlamaForCausalLM): ...
class Qwen2ForSequenceClassification(LlamaForSequenceClassification): ...
class Qwen2ForTokenClassification(LlamaForTokenClassification): ...
class Qwen2ForQuestionAnswering(LlamaForQuestionAnswering): ...

__all__ = [
    "Qwen2ForCausalLM",
    "Qwen2ForQuestionAnswering",
    "Qwen2ForSequenceClassification",
    "Qwen2ForTokenClassification",
    "Qwen2Model",
    "Qwen2PreTrainedModel",
    "Qwen2RMSNorm",
]
