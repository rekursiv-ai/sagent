from typing import Any

from fla.modules import FusedRMSNormGated
from torch import nn

import torch

from .configuration_qwen3_next import Qwen3NextConfig
from ..gemma3.modeling_gemma3 import Gemma3RMSNorm
from ..llama.modeling_llama import (
    LlamaForQuestionAnswering,
    LlamaForSequenceClassification,
    LlamaForTokenClassification,
)
from ..mixtral.modeling_mixtral import MixtralForCausalLM
from ..qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock
from ..qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeDecoderLayer,
    Qwen3MoeMLP,
    Qwen3MoeRotaryEmbedding,
)
from ...cache_utils import Cache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import check_model_inputs
from ...utils.import_utils import (
    is_causal_conv1d_available,
    is_flash_linear_attention_available,
)

"""PyTorch Qwen3-Next model."""
if is_causal_conv1d_available(): ...
else: ...
if is_flash_linear_attention_available(): ...
else:
    FusedRMSNormGated = ...
is_fast_path_available = ...
logger = ...

class Qwen3NextRMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps=..., **kwargs) -> None: ...
    def forward(self, hidden_states, gate=...): ...

class Qwen3NextDynamicCache:
    is_compileable = ...
    def __init__(self, config: Qwen3NextConfig) -> None: ...
    def __len__(self):  # -> int:
        ...
    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]: ...
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def reorder_cache(self, beam_idx: torch.LongTensor):  # -> None:
        ...
    def get_seq_length(self, layer_idx: int | None = ...) -> int: ...
    def get_mask_sizes(
        self, cache_position: torch.Tensor, layer_idx: int
    ) -> tuple[int, int]: ...
    @property
    def has_previous_state(self):  # -> bool:
        ...

class Qwen3NextRotaryEmbedding(Qwen3MoeRotaryEmbedding): ...
class Qwen3NextRMSNorm(Gemma3RMSNorm): ...

class Qwen3NextAttention(Qwen3MoeAttention):
    def __init__(self, config: Qwen3NextConfig, layer_idx: int) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

def torch_causal_conv1d_update(
    hidden_states, conv_state, weight, bias=..., activation=...
):  # -> Tensor:
    ...
def l2norm(x: torch.FloatTensor, dim: int = ..., eps: float = ...):  # -> Tensor:
    ...
def torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=...,
    initial_state=...,
    output_final_state=...,
    use_qk_l2norm_in_kernel=...,
):  # -> tuple[Tensor, Tensor | Any | None]:
    ...
def torch_recurrent_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    initial_state,
    output_final_state,
    use_qk_l2norm_in_kernel=...,
):  # -> tuple[Tensor, Tensor | Any | None]:
    ...

class Qwen3NextGatedDeltaNet(nn.Module):
    def __init__(self, config: Qwen3NextConfig, layer_idx: int) -> None: ...
    def fix_query_key_value_ordering(
        self, mixed_qkvz, mixed_ba
    ):  # -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Qwen3NextDynamicCache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
    ):  # -> Any:
        ...

class Qwen3NextMLP(Qwen3MoeMLP): ...
class Qwen3NextSparseMoeBlock(Qwen2MoeSparseMoeBlock): ...

class Qwen3NextDecoderLayer(Qwen3MoeDecoderLayer):
    def __init__(self, config: Qwen3NextConfig, layer_idx: int) -> None: ...
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
    def __call__(self, *args: Any, **kwargs: Any) -> torch.FloatTensor: ...

class Qwen3NextPreTrainedModel(PreTrainedModel):
    config: Qwen3NextConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _skip_keys_device_placement = ...
    _supports_flash_attn_2 = ...
    _supports_sdpa = ...
    _keys_to_ignore_on_load_unexpected = ...
    _can_record_outputs = ...
    _is_stateful = ...

class Qwen3NextModel(Qwen3NextPreTrainedModel):
    def __init__(self, config: Qwen3NextConfig) -> None: ...
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

class Qwen3NextForCausalLM(MixtralForCausalLM):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Qwen3NextDynamicCache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_router_logits: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeCausalLMOutputWithPast: ...
    def __call__(self, *args: Any, **kwargs: Any) -> MoeCausalLMOutputWithPast: ...

class Qwen3NextForSequenceClassification(LlamaForSequenceClassification): ...
class Qwen3NextForTokenClassification(LlamaForTokenClassification): ...
class Qwen3NextForQuestionAnswering(LlamaForQuestionAnswering): ...

__all__ = [
    "Qwen3NextForCausalLM",
    "Qwen3NextForQuestionAnswering",
    "Qwen3NextForSequenceClassification",
    "Qwen3NextForTokenClassification",
    "Qwen3NextModel",
    "Qwen3NextPreTrainedModel",
]
