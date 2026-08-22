from typing import Any

from torch import nn

import torch

from .configuration_falcon import FalconConfig
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ...modeling_rope_utils import dynamic_rope_update
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring

"""PyTorch Falcon model."""
logger = ...

class FalconLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

def rotate_half(x):  # -> Tensor:
    ...
def apply_rotary_pos_emb(
    q, k, cos, sin, position_ids=..., unsqueeze_dim=...
):  # -> tuple[Any, Any]:
    ...

class FalconRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor
    def __init__(self, config: FalconConfig, device=...) -> None: ...
    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):  # -> tuple[Tensor, Tensor]:
        ...

def build_alibi_tensor(
    attention_mask: torch.Tensor, num_heads: int, dtype: torch.dtype
) -> torch.Tensor: ...
def dropout_add(
    x: torch.Tensor, residual: torch.Tensor, prob: float, training: bool
) -> torch.Tensor: ...

class FalconAttention(nn.Module):
    def __init__(self, config: FalconConfig, layer_idx=...) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        alibi: torch.Tensor | None,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor | None = ...,
        layer_past: Cache | None = ...,
        head_mask: torch.Tensor | None = ...,
        use_cache: bool = ...,
        output_attentions: bool = ...,
        cache_position: torch.LongTensor | None = ...,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = ...,
    ):  # -> tuple[Any, Tensor | None] | tuple[Any, Any | None]:
        ...

class FalconFlashAttention2(FalconAttention):
    def __init__(self, *args, **kwargs) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        alibi: torch.Tensor | None,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor | None = ...,
        layer_past: Cache | None = ...,
        head_mask: torch.Tensor | None = ...,
        use_cache: bool = ...,
        output_attentions: bool = ...,
        cache_position: torch.LongTensor | None = ...,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = ...,
    ):  # -> tuple[Any, Tensor | Any | None]:
        ...

class FalconMLP(nn.Module):
    def __init__(self, config: FalconConfig) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

FALCON_ATTENTION_CLASSES = ...

class FalconDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: FalconConfig, layer_idx=...) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        alibi: torch.Tensor | None,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor | None = ...,
        layer_past: Cache | tuple[torch.Tensor, torch.Tensor] | None = ...,
        head_mask: torch.Tensor | None = ...,
        use_cache: bool = ...,
        output_attentions: bool = ...,
        cache_position: torch.LongTensor | None = ...,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = ...,
        **kwargs,
    ):  # -> tuple[Tensor, Any]:
        ...

@auto_docstring
class FalconPreTrainedModel(PreTrainedModel):
    config: FalconConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _can_compile_fullgraph = ...
    def __init__(self, *inputs, **kwargs) -> None: ...

@auto_docstring
class FalconModel(FalconPreTrainedModel):
    def __init__(self, config: FalconConfig) -> None: ...
    def get_input_embeddings(self):  # -> Embedding | Tensor:
        ...
    def set_input_embeddings(self, new_embeddings: torch.Tensor):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        past_key_values: Cache
        | tuple[tuple[torch.Tensor, torch.Tensor], ...]
        | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        head_mask: torch.LongTensor | None = ...,
        inputs_embeds: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
    ) -> tuple[torch.Tensor, ...] | BaseModelOutputWithPastAndCrossAttentions: ...

@auto_docstring(custom_intro=...)
class FalconForCausalLM(FalconPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    def __init__(self, config: FalconConfig) -> None: ...
    def set_output_embeddings(self, new_embeddings: torch.Tensor):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        past_key_values: Cache
        | tuple[tuple[torch.Tensor, torch.Tensor], ...]
        | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **kwargs,
    ) -> tuple[torch.Tensor] | CausalLMOutputWithCrossAttentions: ...

@auto_docstring(custom_intro=...)
class FalconForSequenceClassification(FalconPreTrainedModel):
    def __init__(self, config: FalconConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor] | SequenceClassifierOutputWithPast: ...

@auto_docstring
class FalconForTokenClassification(FalconPreTrainedModel):
    def __init__(self, config: FalconConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor] | TokenClassifierOutput: ...

@auto_docstring
class FalconForQuestionAnswering(FalconPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        start_positions: torch.LongTensor | None = ...,
        end_positions: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | QuestionAnsweringModelOutput: ...

__all__ = [
    "FalconForCausalLM",
    "FalconForQuestionAnswering",
    "FalconForSequenceClassification",
    "FalconForTokenClassification",
    "FalconModel",
    "FalconPreTrainedModel",
]
