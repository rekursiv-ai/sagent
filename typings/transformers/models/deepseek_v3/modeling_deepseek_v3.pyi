from torch import nn

import torch

from .configuration_deepseek_v3 import DeepseekV3Config
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...integrations import use_kernel_forward_from_hub
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import (
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_rope_utils import dynamic_rope_update
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import check_model_inputs

@use_kernel_forward_from_hub("RMSNorm")
class DeepseekV3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=...) -> None: ...
    def forward(self, hidden_states): ...
    def extra_repr(self):  # -> str:
        ...

class DeepseekV3RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor
    def __init__(self, config: DeepseekV3Config, device=...) -> None: ...
    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):  # -> tuple[Tensor, Tensor]:
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
    **kwargs: Unpack[TransformersKwargs],
):  # -> tuple[Tensor, Tensor]:
    ...
def apply_rotary_pos_emb_interleave(
    q, k, cos, sin, position_ids=..., unsqueeze_dim=...
):  # -> tuple[Any, Any]:
    ...
def yarn_get_mscale(scale=..., mscale=...):  # -> float:
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

class DeepseekV3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: DeepseekV3Config, layer_idx: int) -> None: ...
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

@auto_docstring
class DeepseekV3PreTrainedModel(PreTrainedModel):
    config: DeepseekV3Config
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _skip_keys_device_placement = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _supports_flex_attn = ...
    _can_compile_fullgraph = ...
    _supports_attention_backend = ...
    _can_record_outputs = ...

@auto_docstring
class DeepseekV3Model(DeepseekV3PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config: DeepseekV3Config) -> None: ...
    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        cache_position: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast: ...

@auto_docstring
class DeepseekV3ForCausalLM(DeepseekV3PreTrainedModel, GenerationMixin):
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
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast: ...

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
