from typing import Any
from torch import nn

import torch

from .configuration_pix2struct import (
    Pix2StructConfig,
    Pix2StructTextConfig,
    Pix2StructVisionConfig,
)
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqModelOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring
from ...utils.deprecation import deprecate_kwarg

"""Pix2Struct modeling file"""
logger = ...

class Pix2StructLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=...) -> None: ...
    def forward(self, hidden_states): ...

Pix2StructLayerNorm = ...

class Pix2StructVisionEmbeddings(nn.Module):
    def __init__(self, config: Pix2StructConfig) -> None: ...
    def forward(self, flattened_patches: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class Pix2StructVisionAttention(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states,
        attention_mask=...,
        position_bias=...,
        layer_head_mask=...,
        output_attentions=...,
    ):  # -> tuple[Any, Any | Tensor, Any | Tensor] | tuple[Any, Any | Tensor]:
        ...

class Pix2StructVisionMlp(nn.Module):
    def __init__(self, config: Pix2StructVisionConfig) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class Pix2StructVisionLayer(GradientCheckpointingLayer):
    def __init__(self, config: Pix2StructConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]: ...

class Pix2StructVisionEncoder(nn.Module):
    def __init__(self, config: Pix2StructVisionConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ) -> tuple | BaseModelOutput: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple | BaseModelOutput: ...

@auto_docstring
class Pix2StructPreTrainedModel(PreTrainedModel):
    config: Pix2StructConfig
    _can_compile_fullgraph = ...
    @property
    def dummy_inputs(self):  # -> dict[str, Tensor]:
        ...

@auto_docstring
class Pix2StructVisionModel(Pix2StructPreTrainedModel):
    config: Pix2StructVisionConfig
    main_input_name = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    def __init__(self, config: Pix2StructVisionConfig) -> None: ...
    def get_input_embeddings(self):  # -> Linear:
        ...
    @auto_docstring
    def forward(
        self,
        flattened_patches: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutputWithPooling: ...

class Pix2StructTextDenseGatedActDense(nn.Module):
    def __init__(self, config: Pix2StructTextConfig) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class Pix2StructTextLayerFF(nn.Module):
    def __init__(self, config: Pix2StructTextConfig) -> None: ...
    def forward(self, hidden_states): ...

class Pix2StructTextAttention(nn.Module):
    def __init__(
        self,
        config: Pix2StructTextConfig,
        has_relative_attention_bias=...,
        layer_idx: int | None = ...,
    ) -> None: ...
    def compute_bias(
        self, query_length, key_length, device=..., cache_position=...
    ):  # -> Any:
        ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states,
        mask=...,
        key_value_states=...,
        position_bias=...,
        past_key_values=...,
        layer_head_mask=...,
        query_length=...,
        use_cache=...,
        output_attentions=...,
        cache_position=...,
    ):  # -> tuple[Any, Any | Tensor, Any | Tensor] | tuple[Any, Any | Tensor]:
        ...

class Pix2StructTextLayerSelfAttention(nn.Module):
    def __init__(
        self, config, has_relative_attention_bias=..., layer_idx: int | None = ...
    ) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states,
        attention_mask=...,
        position_bias=...,
        layer_head_mask=...,
        past_key_values=...,
        use_cache=...,
        output_attentions=...,
        cache_position=...,
    ):  # -> Any:
        ...

class Pix2StructTextLayerCrossAttention(nn.Module):
    def __init__(self, config, layer_idx: int | None = ...) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=...,
        position_bias=...,
        layer_head_mask=...,
        past_key_values=...,
        use_cache=...,
        query_length=...,
        output_attentions=...,
        cache_position=...,
    ):  # -> Any:
        ...

class Pix2StructTextBlock(GradientCheckpointingLayer):
    def __init__(
        self, config, has_relative_attention_bias=..., layer_idx: int | None = ...
    ) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states,
        attention_mask=...,
        position_bias=...,
        encoder_hidden_states=...,
        encoder_attention_mask=...,
        encoder_decoder_position_bias=...,
        layer_head_mask=...,
        cross_attn_layer_head_mask=...,
        past_key_values=...,
        use_cache=...,
        output_attentions=...,
        return_dict=...,
        cache_position=...,
    ):  # -> Any:
        ...

@auto_docstring(custom_intro=...)
class Pix2StructTextModel(Pix2StructPreTrainedModel):
    config: Pix2StructTextConfig
    _no_split_modules = ...
    _tied_weights_keys = ...
    supports_gradient_checkpointing = ...
    def __init__(self, config) -> None: ...
    def set_input_embeddings(self, new_embeddings):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        encoder_hidden_states: torch.FloatTensor | None = ...,
        encoder_attention_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        cross_attn_head_mask: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        labels: torch.LongTensor | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs,
    ) -> tuple[torch.FloatTensor, ...] | CausalLMOutputWithCrossAttentions: ...

@auto_docstring(custom_intro=...)
class Pix2StructForConditionalGeneration(Pix2StructPreTrainedModel, GenerationMixin):
    config: Pix2StructConfig
    main_input_name = ...
    _tied_weights_keys = ...
    def __init__(self, config: Pix2StructConfig) -> None: ...
    def get_input_embeddings(self):  # -> Module:
        ...
    def set_input_embeddings(self, new_embeddings):  # -> None:
        ...
    def get_output_embeddings(self) -> nn.Module: ...
    def set_output_embeddings(self, new_embeddings):  # -> None:
        ...
    def get_encoder(self):  # -> Pix2StructVisionModel:
        ...
    @auto_docstring
    def forward(
        self,
        flattened_patches: torch.FloatTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        decoder_input_ids: torch.LongTensor | None = ...,
        decoder_attention_mask: torch.BoolTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        decoder_head_mask: torch.FloatTensor | None = ...,
        cross_attn_head_mask: torch.Tensor | None = ...,
        encoder_outputs: tuple[tuple[torch.FloatTensor]] | None = ...,
        past_key_values: Cache | None = ...,
        labels: torch.LongTensor | None = ...,
        decoder_inputs_embeds: torch.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
    ) -> tuple[torch.FloatTensor] | Seq2SeqModelOutput: ...

__all__ = [
    "Pix2StructForConditionalGeneration",
    "Pix2StructPreTrainedModel",
    "Pix2StructTextModel",
    "Pix2StructVisionModel",
]
