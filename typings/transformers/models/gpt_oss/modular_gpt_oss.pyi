from typing import Any

from torch import nn

import torch

from .configuration_gpt_oss import GptOssConfig
from ..llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from ..mixtral.modeling_mixtral import (
    MixtralForCausalLM,
    MixtralForSequenceClassification,
    MixtralForTokenClassification,
    MixtralModel,
)
from ..qwen2.modeling_qwen2 import Qwen2Attention
from ...cache_utils import Cache
from ...integrations.hub_kernels import use_kernel_forward_from_hub
from ...modeling_outputs import MoeModelOutputWithPast
from ...modeling_rope_utils import dynamic_rope_update
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import check_model_inputs

logger = ...

class GptOssRMSNorm(LlamaRMSNorm):
    def forward(self, hidden_states): ...

class GptOssExperts(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, router_indices=..., routing_weights=...
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class GptOssTopKRouter(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> tuple[Tensor, Tensor]:
        ...

@use_kernel_forward_from_hub("MegaBlocksMoeMLP")
class GptOssMLP(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> tuple[Any, Any]:
        ...

class GptOssRotaryEmbedding(LlamaRotaryEmbedding):
    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):  # -> tuple[Any, Any]:
        ...

def apply_rotary_pos_emb(
    q, k, cos, sin, position_ids=..., unsqueeze_dim=...
):  # -> tuple[Tensor, Tensor]:
    ...
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = ...,
    **kwargs,
):  # -> tuple[Tensor, Tensor]:
    ...

class GptOssAttention(Qwen2Attention):
    def __init__(self, config: GptOssConfig, layer_idx: int) -> None: ...
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

class GptOssDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: GptOssConfig, layer_idx: int) -> None: ...
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
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor: ...

class GptOssPreTrainedModel(LlamaPreTrainedModel):
    _keep_in_fp32_modules = ...
    _supports_sdpa = ...
    _supports_flash_attention = ...
    _supports_flex_attention = ...
    _can_record_outputs = ...

class GptOssModel(MixtralModel):
    _no_split_modules = ...
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
    ) -> MoeModelOutputWithPast: ...

class GptOssForCausalLM(MixtralForCausalLM): ...
class GptOssForSequenceClassification(MixtralForSequenceClassification): ...
class GptOssForTokenClassification(MixtralForTokenClassification): ...

__all__ = [
    "GptOssForCausalLM",
    "GptOssForSequenceClassification",
    "GptOssForTokenClassification",
    "GptOssModel",
    "GptOssPreTrainedModel",
]
