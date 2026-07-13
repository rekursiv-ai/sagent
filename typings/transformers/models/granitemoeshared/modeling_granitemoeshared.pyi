from typing import TypedDict

from torch import nn

import torch

from .configuration_granitemoeshared import GraniteMoeSharedConfig
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, MoeCausalLMOutputWithPast
from ...modeling_rope_utils import dynamic_rope_update
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import auto_docstring
from ...utils.deprecation import deprecate_kwarg

logger = ...

class GraniteFlashAttentionKwargs(TypedDict, total=False):
    cu_seq_lens_q: torch.LongTensor
    cu_seq_lens_k: torch.LongTensor
    max_length_q: int
    max_length_k: int
    seq_idx: torch.IntTensor

class GraniteMoeSharedMLP(nn.Module):
    def __init__(self, config: GraniteMoeSharedConfig) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class GraniteMoeSharedRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=...) -> None: ...
    def forward(self, hidden_states): ...
    def extra_repr(self):  # -> str:
        ...

class GraniteMoeSharedParallelExperts(nn.Module):
    def __init__(self, num_experts: int, input_size: int, output_size: int) -> None: ...
    def forward(self, inputs, expert_size):  # -> Tensor:
        ...

class GraniteMoeSharedTopKGating(nn.Module):
    def __init__(self, input_size: int, num_experts: int, top_k: int) -> None: ...
    def forward(self, hidden_states):  # -> tuple[Any, Any, Tensor, list[Any], Any]:
        ...

class GraniteMoeSharedMoE(nn.Module):
    def __init__(self, config: GraniteMoeSharedConfig) -> None: ...
    def forward(self, layer_input):  # -> tuple[Tensor, Any]:
        ...

def rotate_half(x):  # -> Tensor:
    ...
def apply_rotary_pos_emb(
    q, k, cos, sin, position_ids=..., unsqueeze_dim=...
):  # -> tuple[Any, Any]:
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
    **kwargs,
):  # -> tuple[Tensor, Tensor]:
    ...

class GraniteMoeSharedAttention(nn.Module):
    def __init__(
        self, config: GraniteMoeSharedConfig, layer_idx: int | None = ...
    ) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool = ...,
        cache_position: torch.LongTensor | None = ...,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class GraniteMoeSharedDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: GraniteMoeSharedConfig, layer_idx: int) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        output_attentions: bool | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        output_router_logits: bool | None = ...,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = ...,
        **kwargs: Unpack[GraniteFlashAttentionKwargs],
    ) -> tuple[
        torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None
    ]: ...

@auto_docstring
class GraniteMoeSharedPreTrainedModel(PreTrainedModel):
    config: GraniteMoeSharedConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _skip_keys_device_placement = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _can_compile_fullgraph = ...

class GraniteMoeSharedRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor
    def __init__(self, config: GraniteMoeSharedConfig, device=...) -> None: ...
    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):  # -> tuple[Tensor, Tensor]:
        ...

@auto_docstring
class GraniteMoeSharedModel(GraniteMoeSharedPreTrainedModel):
    def __init__(self, config: GraniteMoeSharedConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | list[torch.FloatTensor] | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        output_router_logits: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs,
    ) -> tuple | BaseModelOutputWithPast: ...

def load_balancing_loss_func(
    gate_logits: torch.Tensor | tuple[torch.Tensor] | None,
    num_experts: int | None = ...,
    top_k=...,
    attention_mask: torch.Tensor | None = ...,
) -> torch.Tensor | int: ...

class GraniteMoeSharedForCausalLM(GraniteMoeSharedPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    def __init__(self, config: GraniteMoeSharedConfig) -> None: ...
    @auto_docstring
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
        output_router_logits: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **kwargs,
    ) -> tuple | MoeCausalLMOutputWithPast: ...

__all__ = [
    "GraniteMoeSharedForCausalLM",
    "GraniteMoeSharedModel",
    "GraniteMoeSharedPreTrainedModel",
]
