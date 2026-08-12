from typing import Any

from fla.modules import FusedRMSNormGated
from torch import nn

import torch

from .configuration_qwen3_next import Qwen3NextConfig
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from ...modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from ...modeling_rope_utils import dynamic_rope_update
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import check_model_inputs
from ...utils.import_utils import (
    is_causal_conv1d_available,
    is_flash_linear_attention_available,
)

if is_causal_conv1d_available(): ...
else: ...
if is_flash_linear_attention_available(): ...
else:
    FusedRMSNormGated = ...
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

class Qwen3NextRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor
    def __init__(self, config: Qwen3NextConfig, device=...) -> None: ...
    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):  # -> tuple[Tensor, Tensor]:
        ...

class Qwen3NextRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = ...) -> None: ...
    def forward(self, x): ...
    def extra_repr(self):  # -> str:
        ...

def rotate_half(x):  # -> Tensor:
    ...
def apply_rotary_pos_emb(
    q, k, cos, sin, position_ids=..., unsqueeze_dim=...
):  # -> tuple[Tensor, Tensor]:
    ...
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor: ...
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = ...,
    **kwargs: Unpack[TransformersKwargs],
):  # -> tuple[Tensor, Tensor]:
    ...

class Qwen3NextAttention(nn.Module):
    def __init__(self, config: Qwen3NextConfig, layer_idx: int) -> None: ...
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

def apply_mask_to_padding_states(hidden_states, attention_mask): ...

is_fast_path_available = ...

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

class Qwen3NextMLP(nn.Module):
    def __init__(self, config, intermediate_size=...) -> None: ...
    def forward(self, x):  # -> Any:
        ...

class Qwen3NextSparseMoeBlock(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class Qwen3NextDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3NextConfig, layer_idx: int) -> None: ...
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

def load_balancing_loss_func(
    gate_logits: torch.Tensor | tuple[torch.Tensor] | None,
    num_experts: int | None = ...,
    top_k=...,
    attention_mask: torch.Tensor | None = ...,
) -> torch.Tensor | int: ...

@auto_docstring
class Qwen3NextForCausalLM(Qwen3NextPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    _tp_plan = ...
    _pp_plan = ...
    def __init__(self, config) -> None: ...
    @can_return_tuple
    @auto_docstring
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

class Qwen3NextForSequenceClassification(
    GenericForSequenceClassification, Qwen3NextPreTrainedModel
): ...
class Qwen3NextForTokenClassification(
    GenericForTokenClassification, Qwen3NextPreTrainedModel
): ...

class Qwen3NextForQuestionAnswering(
    GenericForQuestionAnswering, Qwen3NextPreTrainedModel
):
    base_model_prefix = ...

__all__ = [
    "Qwen3NextForCausalLM",
    "Qwen3NextForQuestionAnswering",
    "Qwen3NextForSequenceClassification",
    "Qwen3NextForTokenClassification",
    "Qwen3NextModel",
    "Qwen3NextPreTrainedModel",
]
