from dataclasses import dataclass
from typing import Any

from torch import nn

import torch

from .configuration_kosmos2 import Kosmos2Config, Kosmos2TextConfig, Kosmos2VisionConfig
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPooling,
    CausalLMOutputWithCrossAttentions,
)
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.deprecation import deprecate_kwarg

"""PyTorch KOSMOS-2 model."""
logger = ...

def create_position_ids_from_input_ids(
    input_ids, padding_idx, past_key_values_length=...
): ...

@dataclass
@auto_docstring(custom_intro=...)
class Kosmos2ModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = ...
    past_key_values: Cache | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...
    image_embeds: torch.FloatTensor | None = ...
    projection_attentions: tuple[torch.FloatTensor] | None = ...
    vision_model_output: BaseModelOutputWithPooling = ...
    def to_tuple(self) -> tuple[Any]: ...

@dataclass
@auto_docstring(custom_intro=...)
class Kosmos2ForConditionalGenerationModelOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    past_key_values: Cache | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...
    image_embeds: torch.FloatTensor | None = ...
    projection_attentions: tuple[torch.FloatTensor] | None = ...
    vision_model_output: BaseModelOutputWithPooling = ...
    def to_tuple(self) -> tuple[Any]: ...

class Kosmos2VisionEmbeddings(nn.Module):
    def __init__(self, config: Kosmos2VisionConfig) -> None: ...
    def interpolate_pos_encoding(
        self, embeddings: torch.Tensor, height: int, width: int
    ) -> torch.Tensor: ...
    def forward(
        self, pixel_values: torch.FloatTensor, interpolate_pos_encoding=...
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

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

class Kosmos2VisionAttention(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        causal_attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class Kosmos2VisionMLP(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class Kosmos2VisionEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Kosmos2VisionConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.FloatTensor]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple[torch.FloatTensor]: ...

class Kosmos2VisionEncoder(nn.Module):
    def __init__(self, config: Kosmos2VisionConfig) -> None: ...
    @can_return_tuple
    def forward(
        self,
        inputs_embeds,
        attention_mask: torch.Tensor | None = ...,
        causal_attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutput: ...

class Kosmos2VisionTransformer(nn.Module):
    def __init__(self, config: Kosmos2VisionConfig) -> None: ...
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        interpolate_pos_encoding: bool = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutputWithPooling: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple | BaseModelOutputWithPooling: ...

class Kosmos2TextSinusoidalPositionalEmbedding(nn.Module):
    def __init__(
        self, num_positions: int, embedding_dim: int, padding_idx: int | None = ...
    ) -> None: ...
    def make_weights(
        self, num_embeddings: int, embedding_dim: int, padding_idx: int | None = ...
    ):  # -> None:
        ...
    @staticmethod
    def get_embedding(
        num_embeddings: int, embedding_dim: int, padding_idx: int | None = ...
    ):  # -> Tensor:
        ...
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        past_key_values_length: int = ...,
        position_ids: torch.Tensor | None = ...,
    ):  # -> Tensor | Any:
        ...
    def create_position_ids_from_inputs_embeds(
        self, inputs_embeds, past_key_values_length
    ): ...

class KosmosTextAttention(nn.Module):
    def __init__(
        self,
        config,
        embed_dim: int,
        num_heads: int,
        dropout: float = ...,
        is_decoder: bool | None = ...,
        add_inner_attn_layernorm: bool | None = ...,
        bias: bool | None = ...,
        layer_idx: bool | None = ...,
    ) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        attention_mask: torch.Tensor | None = ...,
        layer_head_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
        cache_position: torch.Tensor | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]: ...

class Kosmos2TextFFN(nn.Module):
    def __init__(self, config: Kosmos2TextConfig) -> None: ...
    def forward(self, hidden_states):  # -> Tensor:
        ...

class Kosmos2TextBlock(GradientCheckpointingLayer):
    def __init__(self, config: Kosmos2TextConfig, layer_idx=...) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        encoder_hidden_states: torch.Tensor | None = ...,
        encoder_attention_mask: torch.Tensor | None = ...,
        layer_head_mask: torch.Tensor | None = ...,
        cross_attn_layer_head_mask: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        output_attentions: bool | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
        **kwargs,
    ) -> tuple[
        torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None
    ]: ...

class Kosmos2TextTransformer(nn.Module):
    def __init__(self, config: Kosmos2TextConfig) -> None: ...
    def forward_embedding(
        self,
        input_ids,
        inputs_embeds: torch.Tensor | None = ...,
        image_embeds: torch.Tensor | None = ...,
        img_input_mask: torch.Tensor | None = ...,
        past_key_values_length: int = ...,
        position_ids: torch.Tensor | None = ...,
    ):  # -> Tensor:
        ...
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        image_embeds: torch.Tensor | None = ...,
        image_embeds_position_mask: torch.Tensor | None = ...,
        encoder_hidden_states: torch.Tensor | None = ...,
        encoder_attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        cross_attn_head_mask: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple | BaseModelOutputWithPastAndCrossAttentions: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple | BaseModelOutputWithPastAndCrossAttentions: ...

@auto_docstring
class Kosmos2PreTrainedModel(PreTrainedModel):
    config: Kosmos2Config
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _supports_attention_backend = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...

class Kosmos2VisionModel(Kosmos2PreTrainedModel):
    config: Kosmos2VisionConfig
    main_input_name = ...
    def __init__(self, config: Kosmos2VisionConfig) -> None: ...
    def get_input_embeddings(self) -> nn.Module: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        interpolate_pos_encoding: bool = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutputWithPooling: ...

class Kosmos2TextModel(Kosmos2PreTrainedModel):
    config: Kosmos2TextConfig
    def __init__(self, config: Kosmos2TextConfig) -> None: ...
    def get_input_embeddings(self) -> nn.Module: ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        image_embeds: torch.Tensor | None = ...,
        image_embeds_position_mask: torch.Tensor | None = ...,
        encoder_hidden_states: torch.Tensor | None = ...,
        encoder_attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        cross_attn_head_mask: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple | BaseModelOutputWithPastAndCrossAttentions: ...

@auto_docstring(custom_intro=...)
class Kosmos2TextForCausalLM(Kosmos2PreTrainedModel, GenerationMixin):
    config: Kosmos2TextConfig
    _tied_weights_keys = ...
    def __init__(self, config: Kosmos2TextConfig) -> None: ...
    def get_input_embeddings(self) -> nn.Module: ...
    def get_output_embeddings(self) -> nn.Module: ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        image_embeds: torch.Tensor | None = ...,
        image_embeds_position_mask: torch.Tensor | None = ...,
        encoder_hidden_states: torch.Tensor | None = ...,
        encoder_attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        cross_attn_head_mask: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | CausalLMOutputWithCrossAttentions: ...
    def prepare_inputs_for_generation(
        self,
        input_ids,
        image_embeds=...,
        image_embeds_position_mask=...,
        past_key_values=...,
        attention_mask=...,
        inputs_embeds=...,
        use_cache=...,
        cache_position=...,
        **model_kwargs,
    ):  # -> dict[Any, Any]:
        ...

class Kosmos2ImageToTextProjection(nn.Module):
    def __init__(self, config: Kosmos2Config) -> None: ...
    def forward(self, features):  # -> tuple[Any, Any]:
        ...

@auto_docstring(custom_intro=...)
class Kosmos2Model(Kosmos2PreTrainedModel):
    config: Kosmos2Config
    main_input_name = ...
    def __init__(self, config: Kosmos2Config) -> None: ...
    def get_input_embeddings(self) -> nn.Module: ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        return_attentions: bool | None = ...,
        interpolate_pos_encoding: bool | None = ...,
    ):  # -> tuple[Any, Any] | Any:
        ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor | None = ...,
        input_ids: torch.Tensor | None = ...,
        image_embeds_position_mask: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        image_embeds: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        interpolate_pos_encoding: bool = ...,
        return_dict: bool | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple | Kosmos2ModelOutput: ...

@auto_docstring(custom_intro=...)
class Kosmos2ForConditionalGeneration(Kosmos2PreTrainedModel, GenerationMixin):
    config: Kosmos2Config
    main_input_name = ...
    _tied_weights_keys = ...
    def __init__(self, config: Kosmos2Config) -> None: ...
    def get_input_embeddings(self) -> nn.Module: ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def get_output_embeddings(self) -> nn.Module: ...
    def set_output_embeddings(self, new_embeddings):  # -> None:
        ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor | None = ...,
        input_ids: torch.Tensor | None = ...,
        image_embeds_position_mask: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        image_embeds: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Kosmos2ForConditionalGenerationModelOutput: ...
    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor | None = ...,
        image_embeds_position_mask: torch.Tensor | None = ...,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        image_embeds: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        **kwargs,
    ):  # -> GenerateOutput | LongTensor:
        ...

__all__ = ["Kosmos2ForConditionalGeneration", "Kosmos2Model", "Kosmos2PreTrainedModel"]
