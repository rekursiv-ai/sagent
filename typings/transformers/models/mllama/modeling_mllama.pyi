from typing import Any

from torch import nn

import torch

from .configuration_mllama import MllamaConfig, MllamaTextConfig, MllamaVisionConfig
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from ...modeling_rope_utils import dynamic_rope_update
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
)
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import check_model_inputs

"""PyTorch Mllama model."""
logger = ...

class MllamaPrecomputedAspectRatioEmbedding(nn.Module):
    def __init__(self, config: MllamaVisionConfig, is_gated: bool = ...) -> None: ...
    def forward(
        self, hidden_state: torch.Tensor, aspect_ratio_ids: torch.Tensor
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class MllamaPrecomputedPositionEmbedding(nn.Module):
    def __init__(self, config: MllamaVisionConfig) -> None: ...
    def forward(
        self, hidden_state: torch.Tensor, aspect_ratio_ids: torch.Tensor
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class MllamaVisionMLP(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

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

class MllamaVisionAttention(nn.Module):
    def __init__(self, config: MllamaVisionConfig) -> None: ...
    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class MllamaVisionEncoderLayer(nn.Module):
    def __init__(self, config: MllamaVisionConfig, is_gated: bool = ...) -> None: ...
    def forward(
        self, hidden_state: torch.Tensor, attention_mask: torch.Tensor | None = ...
    ):  # -> Tensor:
        ...

class MllamaVisionEncoder(nn.Module):
    def __init__(
        self, config: MllamaVisionConfig, num_layers=..., is_gated=...
    ) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = ...
    ) -> BaseModelOutput: ...
    def __call__(self, *args: Any, **kwargs: Any) -> BaseModelOutput: ...

class MllamaTextRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=...) -> None: ...
    def forward(self, hidden_states): ...
    def extra_repr(self):  # -> str:
        ...

class MllamaTextCrossAttention(nn.Module):
    def __init__(
        self, config: MllamaTextConfig | None = ..., layer_idx: int | None = ...
    ) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        attention_mask: torch.Tensor | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

def rotate_half(x):  # -> Tensor:
    ...
def apply_rotary_pos_emb(
    q, k, cos, sin, position_ids=..., unsqueeze_dim=...
):  # -> tuple[Any, Any]:
    ...

class MllamaTextSelfAttention(nn.Module):
    def __init__(self, config: MllamaTextConfig, layer_idx: int) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: torch.Tensor,
        use_cache: bool = ...,
        past_key_values=...,
        cache_position=...,
        **kwargs,
    ):  # -> tuple[Any, Any]:
        ...

class MllamaTextMLP(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, x):  # -> Any:
        ...

class MllamaSelfAttentionDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: MllamaTextConfig, layer_idx: int) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: torch.Tensor | None = ...,
        cross_attention_mask: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        full_text_row_masked_out_mask: tuple[torch.Tensor, torch.Tensor] | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[
        torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None
    ]: ...

class MllamaCrossAttentionDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: MllamaTextConfig, layer_idx: int) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: torch.Tensor,
        cross_attention_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        full_text_row_masked_out_mask: tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        position_embeddings: torch.Tensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor]: ...

class MllamaRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor
    def __init__(self, config: MllamaTextConfig, device=...) -> None: ...
    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):  # -> tuple[Tensor, Tensor]:
        ...

@auto_docstring
class MllamaPreTrainedModel(PreTrainedModel):
    config: MllamaConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _can_compile_fullgraph = ...
    _supports_sdpa = ...
    _supports_flash_attn = ...
    _supports_flex_attn = ...
    _supports_attention_backend = ...
    _can_record_outputs = ...

@auto_docstring(custom_intro=...)
class MllamaVisionModel(MllamaPreTrainedModel):
    config: MllamaVisionConfig
    base_model_prefix = ...
    def __init__(self, config: MllamaVisionConfig) -> None: ...
    def get_input_embeddings(self):  # -> Conv2d:
        ...
    def apply_class_embedding(self, hidden_state: torch.Tensor) -> torch.Tensor: ...
    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor,
        aspect_ratio_ids: torch.Tensor,
        aspect_ratio_mask: torch.Tensor,
        **kwargs,
    ) -> BaseModelOutput: ...

@auto_docstring(custom_intro=...)
class MllamaTextModel(MllamaPreTrainedModel):
    config: MllamaTextConfig
    base_model_prefix = ...
    def __init__(self, config: MllamaTextConfig) -> None: ...
    @check_model_inputs
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        cross_attention_states: torch.FloatTensor | None = ...,
        cross_attention_mask: torch.Tensor | None = ...,
        full_text_row_masked_out_mask: tuple[torch.Tensor, torch.Tensor] | None = ...,
        past_key_values: Cache | list[torch.FloatTensor] | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast: ...

@auto_docstring(custom_intro=...)
class MllamaForCausalLM(MllamaPreTrainedModel, GenerationMixin):
    config: MllamaTextConfig
    _can_compile_fullgraph = ...
    base_model_prefix = ...
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        cross_attention_states: torch.LongTensor | None = ...,
        cross_attention_mask: torch.LongTensor | None = ...,
        full_text_row_masked_out_mask: tuple[torch.Tensor, torch.Tensor] | None = ...,
        past_key_values: Cache | list[torch.FloatTensor] | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | CausalLMOutputWithPast: ...

@auto_docstring(custom_intro=...)
class MllamaModel(MllamaPreTrainedModel):
    _checkpoint_conversion_mapping = ...
    def __init__(self, config: MllamaConfig) -> None: ...
    def get_input_embeddings(self):  # -> Module:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def set_decoder(self, decoder):  # -> None:
        ...
    def get_decoder(self):  # -> MllamaTextModel:
        ...
    @check_model_inputs
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        aspect_ratio_mask: torch.Tensor | None = ...,
        aspect_ratio_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        cross_attention_mask: torch.Tensor | None = ...,
        cross_attention_states: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast: ...

@auto_docstring(custom_intro=...)
class MllamaForConditionalGeneration(MllamaPreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = ...
    _tied_weights_keys = ...
    def __init__(self, config: MllamaConfig) -> None: ...
    def get_input_embeddings(self):  # -> Module:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def set_decoder(self, decoder):  # -> None:
        ...
    def get_decoder(self):  # -> MllamaTextModel:
        ...
    @property
    def language_model(self):  # -> MllamaTextModel:
        ...
    @property
    def vision_model(self):  # -> MllamaVisionModel:
        ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        aspect_ratio_mask: torch.Tensor | None = ...,
        aspect_ratio_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        cross_attention_mask: torch.Tensor | None = ...,
        cross_attention_states: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | list[torch.FloatTensor] | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | CausalLMOutputWithPast: ...
    def prepare_inputs_for_generation(
        self,
        input_ids=...,
        inputs_embeds=...,
        attention_mask=...,
        position_ids=...,
        pixel_values=...,
        aspect_ratio_ids=...,
        aspect_ratio_mask=...,
        cross_attention_mask=...,
        past_key_values=...,
        use_cache=...,
        cache_position=...,
        logits_to_keep=...,
        **kwargs,
    ):  # -> dict[Any, Any]:
        ...

__all__ = [
    "MllamaForCausalLM",
    "MllamaForConditionalGeneration",
    "MllamaModel",
    "MllamaPreTrainedModel",
    "MllamaTextModel",
    "MllamaVisionModel",
]
