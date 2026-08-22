from typing import Any

from torch import nn

import torch

from .configuration_stablelm import StableLmConfig
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_layers import (
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_rope_utils import dynamic_rope_update
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring, can_return_tuple
from ...utils.deprecation import deprecate_kwarg

"""PyTorch StableLM model."""
logger = ...

class StableLmRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor
    def __init__(self, config: StableLmConfig, device=...) -> None: ...
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

class StableLmMLP(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, x):  # -> Any:
        ...

class StableLmLayerNormPerHead(nn.Module):
    def __init__(self, dim, num_heads, eps=..., bias=...) -> None: ...
    def forward(self, hidden_states: torch.Tensor):  # -> Tensor:
        ...

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor: ...

class StableLmAttention(nn.Module):
    def __init__(self, config: StableLmConfig, layer_idx: int | None = ...) -> None: ...
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
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class StableLmSdpaAttention(StableLmAttention):
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
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class StableLmFlashAttention2(StableLmAttention):
    def __init__(self, *args, **kwargs) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        output_attentions: bool = ...,
        use_cache: bool = ...,
        cache_position: torch.LongTensor | None = ...,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

ATTENTION_CLASSES = ...

class StableLmDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: StableLmConfig, layer_idx: int) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        output_attentions: bool | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = ...,
    ) -> tuple[
        torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None
    ]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[
        torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None
    ]: ...

@auto_docstring
class StableLmPreTrainedModel(PreTrainedModel):
    config: StableLmConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _skip_keys_device_placement = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _can_compile_fullgraph = ...

@auto_docstring
class StableLmModel(StableLmPreTrainedModel):
    def __init__(self, config: StableLmConfig) -> None: ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
    ) -> BaseModelOutputWithPast: ...

class StableLmForCausalLM(StableLmPreTrainedModel, GenerationMixin):
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
        cache_position: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **kwargs,
    ) -> CausalLMOutputWithPast: ...

class StableLmForSequenceClassification(
    GenericForSequenceClassification, StableLmPreTrainedModel
): ...
class StableLmForTokenClassification(
    GenericForTokenClassification, StableLmPreTrainedModel
): ...

__all__ = [
    "StableLmForCausalLM",
    "StableLmForSequenceClassification",
    "StableLmForTokenClassification",
    "StableLmModel",
    "StableLmPreTrainedModel",
]
