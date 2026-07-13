from torch import nn

import torch

from .configuration_jetmoe import JetMoeConfig
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_layers import (
    GenericForSequenceClassification,
    GradientCheckpointingLayer,
)
from ...modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from ...modeling_rope_utils import dynamic_rope_update
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring, can_return_tuple
from ...utils.deprecation import deprecate_kwarg

"""PyTorch JetMoe model."""
logger = ...

def load_balancing_loss_func(
    gate_logits: torch.Tensor | tuple[torch.Tensor] | None,
    num_experts: int | None = ...,
    top_k=...,
    attention_mask: torch.Tensor | None = ...,
) -> torch.Tensor | int: ...

class JetMoeParallelExperts(nn.Module):
    def __init__(self, num_experts: int, input_size: int, output_size: int) -> None: ...
    def forward(self, inputs, expert_size):  # -> Tensor:
        ...

class JetMoeTopKGating(nn.Module):
    def __init__(self, input_size: int, num_experts: int, top_k: int) -> None: ...
    def forward(self, hidden_states):  # -> tuple[Any, Any, Tensor, list[Any], Any]:
        ...

class JetMoeMoE(nn.Module):
    def __init__(self, config: JetMoeConfig) -> None: ...
    def forward(self, layer_input):  # -> tuple[Tensor, Any]:
        ...

class JetMoeMoA(nn.Module):
    def __init__(self, config: JetMoeConfig) -> None: ...
    def map(self, layer_input):  # -> tuple[Tensor, Any, tuple[Any, Any, Any, Any]]:
        ...
    def reduce(self, layer_input, topo_info):  # -> Tensor:
        ...
    def forward(self, layer_input): ...

class JetMoeRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=...) -> None: ...
    def forward(self, hidden_states): ...
    def extra_repr(self):  # -> str:
        ...

class JetMoeRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor
    def __init__(self, config: JetMoeConfig, device=...) -> None: ...
    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):  # -> tuple[Tensor, Tensor]:
        ...

def rotate_half(x):  # -> Tensor:
    ...
def apply_rotary_pos_emb(
    q, k, cos, sin, position_ids=..., unsqueeze_dim=...
):  # -> tuple[Any, Any]:
    ...

class JetMoeAttention(nn.Module):
    def __init__(self, config: JetMoeConfig, layer_idx: int | None = ...) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        output_attentions: bool = ...,
        use_cache: bool = ...,
        cache_position: torch.LongTensor | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class JetMoeSdpaAttention(JetMoeAttention):
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        output_attentions: bool = ...,
        use_cache: bool = ...,
        cache_position: torch.LongTensor | None = ...,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        tuple[torch.Tensor] | None,
        torch.Tensor | None,
    ]: ...

class JetMoeFlashAttention2(JetMoeAttention):
    def __init__(self, *args, **kwargs) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.FloatTensor | None,
        attention_mask: torch.FloatTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
    ) -> (
        tuple[torch.Tensor, tuple[torch.Tensor]]
        | tuple[torch.Tensor, tuple[torch.Tensor], tuple[torch.Tensor, ...]]
        | None
    ): ...

JETMOE_ATTENTION_CLASSES = ...

class JetMoeBlock(GradientCheckpointingLayer):
    def __init__(self, config: JetMoeConfig, layer_idx: int | None = ...) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.FloatTensor | None,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_router_logits: bool | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
    ) -> (
        tuple[torch.Tensor] | tuple[torch.Tensor, tuple[torch.FloatTensor, ...]] | None
    ): ...

@auto_docstring
class JetMoePreTrainedModel(PreTrainedModel):
    config: JetMoeConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _skip_keys_device_placement = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...

@auto_docstring
class JetMoeModel(JetMoePreTrainedModel):
    def __init__(self, config: JetMoeConfig) -> None: ...
    @can_return_tuple
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
        cache_position: torch.LongTensor | None = ...,
    ) -> MoeModelOutputWithPast: ...

class JetMoeForCausalLM(JetMoePreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        output_router_logits: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **kwargs,
    ) -> MoeCausalLMOutputWithPast: ...

class JetMoeForSequenceClassification(
    GenericForSequenceClassification, JetMoePreTrainedModel
): ...

__all__ = [
    "JetMoeForCausalLM",
    "JetMoeForSequenceClassification",
    "JetMoeModel",
    "JetMoePreTrainedModel",
]
