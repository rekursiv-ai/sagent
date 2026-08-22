from dataclasses import dataclass
from typing import Any

from torch import nn

import torch

from .configuration_aria import AriaConfig, AriaTextConfig
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...integrations import use_kernel_forward_from_hub
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    ModelOutput,
)
from ...modeling_rope_utils import dynamic_rope_update
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import check_model_inputs

@use_kernel_forward_from_hub("RMSNorm")
class AriaTextRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=...) -> None: ...
    def forward(self, hidden_states): ...
    def extra_repr(self):  # -> str:
        ...

class AriaProjectorMLP(nn.Module):
    def __init__(self, in_features, hidden_features, output_dim) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class AriaCrossAttention(nn.Module):
    def __init__(self, config: AriaConfig, dropout_rate: float = ...) -> None: ...
    def forward(self, key_value_states, hidden_states, attn_mask=...):  # -> Any:
        ...

class AriaProjector(nn.Module):
    def __init__(self, config: AriaConfig) -> None: ...
    def forward(
        self, key_value_states: torch.Tensor, attn_mask: torch.Tensor | None = ...
    ):  # -> Any:
        ...

class AriaSharedExpertsMLP(nn.Module):
    def __init__(self, config: AriaTextConfig) -> None: ...
    def forward(self, x):  # -> Any:
        ...

def sequential_experts_gemm(
    token_states, expert_weights, tokens_per_expert
):  # -> Tensor:
    ...

class AriaGroupedExpertsGemm(nn.Module):
    def __init__(self, in_features, out_features, groups) -> None: ...
    def forward(self, input, tokens_per_expert):  # -> Tensor:
        ...

class AriaGroupedExpertsMLP(nn.Module):
    def __init__(self, config: AriaTextConfig) -> None: ...
    def forward(self, permuted_tokens, tokens_per_expert):  # -> Any:
        ...

class AriaTextMoELayer(nn.Module):
    def __init__(self, config: AriaTextConfig) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

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

class AriaTextAttention(nn.Module):
    def __init__(self, config: AriaTextConfig, layer_idx: int) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

class AriaTextDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: AriaTextConfig, layer_idx: int) -> None: ...
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
class AriaTextPreTrainedModel(PreTrainedModel):
    config: AriaTextConfig
    base_model_prefix = ...
    _no_split_modules = ...
    supports_gradient_checkpointing = ...
    _skip_keys_device_placement = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _supports_attention_backend = ...
    _can_record_outputs = ...

@auto_docstring
class AriaPreTrainedModel(PreTrainedModel):
    config: AriaConfig
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

class AriaTextRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor
    def __init__(self, config: AriaTextConfig, device=...) -> None: ...
    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):  # -> tuple[Tensor, Tensor]:
        ...

@auto_docstring
class AriaTextModel(AriaTextPreTrainedModel):
    def __init__(self, config: AriaTextConfig) -> None: ...
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
class AriaTextForCausalLM(AriaTextPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    _tp_plan = ...
    _pp_plan = ...
    def __init__(self, config: AriaTextConfig) -> None: ...
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

@dataclass
@auto_docstring(custom_intro=...)
class AriaCausalLMOutputWithPast(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    past_key_values: Cache | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...
    image_hidden_states: torch.FloatTensor | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class AriaModelOutputWithPast(BaseModelOutputWithPast):
    image_hidden_states: torch.FloatTensor | None = ...

@auto_docstring(custom_intro=...)
class AriaModel(AriaPreTrainedModel):
    _checkpoint_conversion_mapping = ...
    def __init__(self, config: AriaConfig) -> None: ...
    def get_input_embeddings(self):  # -> Any:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def set_decoder(self, decoder):  # -> None:
        ...
    def get_decoder(self):  # -> Any:
        ...
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: torch.FloatTensor | None = ...,
        vision_feature_layer: int = ...,
    ):  # -> Any:
        ...
    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor,
    ):  # -> Any:
        ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        pixel_mask: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple | AriaModelOutputWithPast: ...

@auto_docstring(custom_intro=...)
class AriaForConditionalGeneration(AriaPreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = ...
    _tied_weights_keys = ...
    def __init__(self, config: AriaConfig) -> None: ...
    def get_input_embeddings(self):  # -> Any:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def get_output_embeddings(self) -> nn.Module: ...
    def set_decoder(self, decoder):  # -> None:
        ...
    def get_decoder(self):  # -> Any:
        ...
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: torch.FloatTensor | None = ...,
        vision_feature_layer: int = ...,
    ):  # -> Any:
        ...
    @property
    def language_model(self):  # -> Any:
        ...
    @property
    def vision_tower(self):  # -> Any:
        ...
    @property
    def multi_modal_projector(self):  # -> AriaProjector:
        ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        pixel_mask: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | AriaCausalLMOutputWithPast: ...
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=...,
        inputs_embeds=...,
        pixel_values=...,
        pixel_mask=...,
        attention_mask=...,
        cache_position=...,
        logits_to_keep=...,
        **kwargs,
    ):  # -> dict[Any, Any]:
        ...

__all__ = [
    "AriaForConditionalGeneration",
    "AriaModel",
    "AriaPreTrainedModel",
    "AriaTextForCausalLM",
    "AriaTextModel",
    "AriaTextPreTrainedModel",
]
