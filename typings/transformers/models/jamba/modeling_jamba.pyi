from typing import Any

from torch import nn

import torch

from .configuration_jamba import JambaConfig
from ...generation import GenerationMixin
from ...modeling_layers import (
    GenericForSequenceClassification,
    GradientCheckpointingLayer,
)
from ...modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.deprecation import deprecate_kwarg
from ...utils.import_utils import is_causal_conv1d_available, is_mamba_ssm_available

"""PyTorch Jamba model."""
if is_mamba_ssm_available(): ...
else: ...
if is_causal_conv1d_available(): ...
else: ...
is_fast_path_available = ...
logger = ...

def load_balancing_loss_func(
    router_logits: torch.Tensor | tuple[torch.Tensor] | None,
    num_experts: int | None = ...,
    top_k=...,
    attention_mask: torch.Tensor | None = ...,
) -> torch.Tensor | int: ...

class JambaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=...) -> None: ...
    def forward(self, hidden_states): ...
    def extra_repr(self):  # -> str:
        ...

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor: ...

class HybridMambaAttentionDynamicCache:
    is_compileable = ...
    def __init__(self, config, batch_size, dtype=..., device=...) -> None: ...
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

class JambaAttention(nn.Module):
    def __init__(self, config: JambaConfig, layer_idx: int | None = ...) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: HybridMambaAttentionDynamicCache | None = ...,
        output_attentions: bool = ...,
        use_cache: bool = ...,
        cache_position: torch.LongTensor | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class JambaFlashAttention2(JambaAttention):
    def __init__(self, *args, **kwargs) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: HybridMambaAttentionDynamicCache | None = ...,
        output_attentions: bool = ...,
        use_cache: bool = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs,
    ):  # -> tuple[Any, Any | None, HybridMambaAttentionDynamicCache | None]:
        ...

class JambaSdpaAttention(JambaAttention):
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: HybridMambaAttentionDynamicCache | None = ...,
        output_attentions: bool = ...,
        use_cache: bool = ...,
        cache_position: torch.LongTensor | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

JAMBA_ATTENTION_CLASSES = ...

class JambaMambaMixer(nn.Module):
    def __init__(self, config: JambaConfig, layer_idx) -> None: ...
    def cuda_kernels_forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: HybridMambaAttentionDynamicCache | None = ...,
        attention_mask: torch.LongTensor | None = ...,
    ):  # -> Any:
        ...
    def slow_forward(
        self,
        input_states,
        cache_params: HybridMambaAttentionDynamicCache | None = ...,
        attention_mask: torch.LongTensor | None = ...,
    ):  # -> Any:
        ...
    def forward(
        self,
        hidden_states,
        cache_params: HybridMambaAttentionDynamicCache | None = ...,
        attention_mask: torch.LongTensor | None = ...,
    ):  # -> Any:
        ...

class JambaMLP(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, x):  # -> Any:
        ...

class JambaSparseMoeBlock(nn.Module):
    def __init__(self, config: JambaConfig) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

class JambaAttentionDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: JambaConfig, layer_idx: int) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: HybridMambaAttentionDynamicCache | None = ...,
        output_attentions: bool | None = ...,
        output_router_logits: bool | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
    ) -> tuple[
        torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None
    ]: ...

class JambaMambaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: JambaConfig, layer_idx: int) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: HybridMambaAttentionDynamicCache | None = ...,
        output_attentions: bool | None = ...,
        output_router_logits: bool | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
    ) -> tuple[
        torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None
    ]: ...

@auto_docstring
class JambaPreTrainedModel(PreTrainedModel):
    config: JambaConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _skip_keys_device_placement = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _is_stateful = ...

ALL_DECODER_LAYER_TYPES = ...

@auto_docstring
class JambaModel(JambaPreTrainedModel):
    def __init__(self, config: JambaConfig) -> None: ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: HybridMambaAttentionDynamicCache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        output_router_logits: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast: ...

class JambaForCausalLM(JambaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    def __init__(self, config: JambaConfig) -> None: ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: HybridMambaAttentionDynamicCache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        output_router_logits: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeCausalLMOutputWithPast: ...
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=...,
        attention_mask=...,
        inputs_embeds=...,
        output_router_logits=...,
        cache_position=...,
        position_ids=...,
        use_cache=...,
        **kwargs,
    ):  # -> dict[str, Any]:
        ...

class JambaForSequenceClassification(
    GenericForSequenceClassification, JambaPreTrainedModel
): ...

__all__ = [
    "JambaForCausalLM",
    "JambaForSequenceClassification",
    "JambaModel",
    "JambaPreTrainedModel",
]
