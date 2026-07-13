from torch import Size, Tensor, nn

import torch

from .configuration_nemotron import NemotronConfig
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_rope_utils import dynamic_rope_update
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring, can_return_tuple
from ...utils.deprecation import deprecate_kwarg

"""PyTorch Nemotron model."""
logger = ...

class NemotronLayerNorm1P(nn.LayerNorm):
    def __init__(
        self,
        normalized_shape: int | list[int] | Size,
        eps: float = ...,
        elementwise_affine: bool = ...,
        bias: bool = ...,
        device=...,
        dtype=...,
    ) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class NemotronRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor
    def __init__(self, config: NemotronConfig, device=...) -> None: ...
    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):  # -> tuple[Tensor, Tensor]:
        ...

def rotate_half(x):  # -> Tensor:
    ...
def apply_rotary_pos_emb(
    q, k, cos, sin, position_ids=..., unsqueeze_dim=...
):  # -> tuple[Tensor, Tensor]:
    ...

class NemotronMLP(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, x):  # -> Any:
        ...

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor: ...

class NemotronAttention(nn.Module):
    def __init__(self, config: NemotronConfig, layer_idx: int | None = ...) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        output_attentions: bool = ...,
        use_cache: bool = ...,
        cache_position: torch.LongTensor | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class NemotronFlashAttention2(NemotronAttention):
    def __init__(self, *args, **kwargs) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        output_attentions: bool = ...,
        use_cache: bool = ...,
        cache_position: torch.LongTensor | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class NemotronSdpaAttention(NemotronAttention):
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        output_attentions: bool = ...,
        use_cache: bool = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

NEMOTRON_ATTENTION_CLASSES = ...

class NemotronDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: NemotronConfig, layer_idx: int) -> None: ...
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
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = ...,
        **kwargs,
    ) -> tuple[
        torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None
    ]: ...

@auto_docstring
class NemotronPreTrainedModel(PreTrainedModel):
    config: NemotronConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _skip_keys_device_placement = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _can_compile_fullgraph = ...

@auto_docstring
class NemotronModel(NemotronPreTrainedModel):
    def __init__(self, config: NemotronConfig) -> None: ...
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
        cache_position: torch.LongTensor | None = ...,
    ) -> BaseModelOutputWithPast: ...

class NemotronForCausalLM(NemotronPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    @can_return_tuple
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
        cache_position: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **kwargs,
    ) -> CausalLMOutputWithPast: ...

class NemotronForSequenceClassification(
    GenericForSequenceClassification, NemotronPreTrainedModel
): ...

class NemotronForQuestionAnswering(
    GenericForQuestionAnswering, NemotronPreTrainedModel
):
    base_model_prefix = ...

class NemotronForTokenClassification(
    GenericForTokenClassification, NemotronPreTrainedModel
): ...

__all__ = [
    "NemotronForCausalLM",
    "NemotronForQuestionAnswering",
    "NemotronForSequenceClassification",
    "NemotronForTokenClassification",
    "NemotronModel",
    "NemotronPreTrainedModel",
]
