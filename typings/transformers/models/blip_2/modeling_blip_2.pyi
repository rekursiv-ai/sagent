from dataclasses import dataclass
from typing import Any

from torch import nn

import torch

from .configuration_blip_2 import Blip2Config, Blip2QFormerConfig, Blip2VisionConfig
from ...generation import GenerationMixin
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithPast,
)
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import (
    ModelOutput,
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    filter_out_non_signature_kwargs,
)
from ...utils.generic import check_model_inputs

"""PyTorch BLIP-2 model."""
logger = ...

@dataclass
@auto_docstring(custom_intro=...)
class Blip2ForConditionalGenerationModelOutput(ModelOutput):
    loss: tuple[torch.FloatTensor] | None = ...
    logits: tuple[torch.FloatTensor] | None = ...
    vision_outputs: torch.FloatTensor | None = ...
    qformer_outputs: tuple[torch.FloatTensor] | None = ...
    language_model_outputs: tuple[torch.FloatTensor] | None = ...
    def to_tuple(self) -> tuple[Any]: ...

@dataclass
@auto_docstring
class Blip2ImageTextMatchingModelOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits_per_image: torch.FloatTensor | None = ...
    logits_per_text: torch.FloatTensor | None = ...
    text_embeds: torch.FloatTensor | None = ...
    image_embeds: torch.FloatTensor | None = ...
    text_model_output: BaseModelOutputWithPooling = ...
    vision_model_output: BaseModelOutputWithPooling = ...
    def to_tuple(self) -> tuple[Any]: ...

@dataclass
@auto_docstring(custom_intro=...)
class Blip2TextModelOutput(ModelOutput):
    text_embeds: torch.FloatTensor | None = ...
    last_hidden_state: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class Blip2VisionModelOutput(ModelOutput):
    image_embeds: torch.FloatTensor | None = ...
    last_hidden_state: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...

class Blip2VisionEmbeddings(nn.Module):
    def __init__(self, config: Blip2VisionConfig) -> None: ...
    def interpolate_pos_encoding(
        self, embeddings: torch.Tensor, height: int, width: int
    ) -> torch.Tensor: ...
    def forward(
        self, pixel_values: torch.FloatTensor, interpolate_pos_encoding: bool = ...
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

class Blip2Attention(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: torch.Tensor | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class Blip2MLP(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class Blip2EncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Blip2Config) -> None: ...
    @auto_docstring
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.FloatTensor: ...

@auto_docstring
class Blip2PreTrainedModel(PreTrainedModel):
    config: Blip2Config
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _supports_attention_backend = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _supports_flex_attn = ...
    _no_split_modules = ...
    _skip_keys_device_placement = ...

class Blip2Encoder(nn.Module):
    def __init__(self, config: Blip2Config) -> None: ...
    @auto_docstring
    def forward(
        self,
        inputs_embeds,
        attention_mask: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutput: ...

@auto_docstring
class Blip2VisionModel(Blip2PreTrainedModel):
    main_input_name = ...
    config: Blip2VisionConfig
    _can_record_outputs = ...
    def __init__(self, config: Blip2VisionConfig) -> None: ...
    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        interpolate_pos_encoding: bool = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling: ...
    def get_input_embeddings(self):  # -> Blip2VisionEmbeddings:
        ...

class Blip2QFormerMultiHeadAttention(nn.Module):
    def __init__(self, config, is_cross_attention=...) -> None: ...
    def save_attn_gradients(self, attn_gradients):  # -> None:
        ...
    def get_attn_gradients(self): ...
    def save_attention_map(self, attention_map):  # -> None:
        ...
    def get_attention_map(self): ...
    def transpose_for_scores(self, x): ...
    def forward(
        self,
        hidden_states,
        attention_mask=...,
        head_mask=...,
        encoder_hidden_states=...,
        encoder_attention_mask=...,
        **kwargs: Unpack[TransformersKwargs],
    ):  # -> tuple[Tensor, Any]:
        ...

class Blip2QFormerSelfOutput(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class Blip2QFormerAttention(nn.Module):
    def __init__(self, config, is_cross_attention=...) -> None: ...
    def prune_heads(self, heads):  # -> None:
        ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        encoder_hidden_states: torch.FloatTensor | None = ...,
        encoder_attention_mask: torch.FloatTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class Blip2QFormerIntermediate(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class Blip2QFormerOutput(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class Blip2QFormerLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx) -> None: ...
    def forward(
        self,
        hidden_states,
        attention_mask=...,
        head_mask=...,
        encoder_hidden_states=...,
        encoder_attention_mask=...,
        query_length=...,
        **kwargs: Unpack[TransformersKwargs],
    ):  # -> Tensor:
        ...
    def feed_forward_chunk(self, attention_output):  # -> Any:
        ...
    def feed_forward_chunk_query(self, attention_output):  # -> Any:
        ...

class Blip2QFormerEncoder(nn.Module):
    def __init__(self, config) -> None: ...
    @can_return_tuple
    def forward(
        self,
        hidden_states,
        attention_mask=...,
        head_mask=...,
        encoder_hidden_states=...,
        encoder_attention_mask=...,
        query_length=...,
        **kwargs: Unpack[TransformersKwargs],
    ):  # -> BaseModelOutputWithPastAndCrossAttentions:
        ...

class Blip2TextEmbeddings(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        input_ids: torch.FloatTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        query_embeds: torch.FloatTensor | None = ...,
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

@auto_docstring(
    custom_intro="""
    BLIP-2 Querying Transformer (Q-Former).
    """
)
class Blip2QFormerModel(Blip2PreTrainedModel):
    _supports_attention_backend = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _supports_flex_attn = ...
    _can_record_outputs = ...
    def __init__(self, config: Blip2QFormerConfig) -> None: ...
    def get_input_embeddings(self):  # -> Tensor | Module:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def get_extended_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: tuple[int],
        device: torch.device,
        has_query: bool = ...,
    ) -> torch.Tensor: ...
    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        query_embeds: torch.FloatTensor,
        query_length: int | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        encoder_hidden_states: torch.FloatTensor | None = ...,
        encoder_attention_mask: torch.FloatTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor] | BaseModelOutputWithPoolingAndCrossAttentions: ...

@auto_docstring(custom_intro=...)
class Blip2Model(Blip2PreTrainedModel):
    config: Blip2Config
    main_input_name = ...
    _keep_in_fp32_modules = ...
    _supports_flash_attn = ...
    def __init__(self, config: Blip2Config) -> None: ...
    def get_input_embeddings(self):  # -> Any:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def set_output_embeddings(self, new_embeddings):  # -> None:
        ...
    def get_output_embeddings(self) -> nn.Module: ...
    def get_encoder(self):  # -> Any:
        ...
    def get_decoder(self):  # -> Any:
        ...
    @filter_out_non_signature_kwargs()
    @auto_docstring
    def get_text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        decoder_input_ids: torch.Tensor | None = ...,
        decoder_attention_mask: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        legacy_output: bool = ...,
    ) -> torch.FloatTensor | CausalLMOutputWithPast: ...
    @filter_out_non_signature_kwargs()
    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        interpolate_pos_encoding: bool = ...,
        legacy_output: bool = ...,
    ) -> torch.FloatTensor | CausalLMOutputWithPast: ...
    @filter_out_non_signature_kwargs()
    @auto_docstring
    def get_qformer_features(
        self,
        pixel_values: torch.FloatTensor,
        interpolate_pos_encoding: bool = ...,
        legacy_output: bool = ...,
    ) -> torch.FloatTensor | BaseModelOutputWithPooling: ...
    def get_placeholder_mask(
        self, input_ids: torch.LongTensor, inputs_embeds: torch.FloatTensor
    ):  # -> Any:
        ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.FloatTensor,
        attention_mask: torch.LongTensor | None = ...,
        decoder_input_ids: torch.LongTensor | None = ...,
        decoder_attention_mask: torch.LongTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        interpolate_pos_encoding: bool = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Blip2ForConditionalGenerationModelOutput: ...

@auto_docstring
class Blip2TextModelWithProjection(Blip2PreTrainedModel):
    supports_gradient_checkpointing = ...
    _keep_in_fp32_modules = ...
    _supports_flash_attn = ...
    def __init__(self, config: Blip2Config) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Blip2TextModelOutput: ...

@auto_docstring
class Blip2VisionModelWithProjection(Blip2PreTrainedModel):
    main_input_name = ...
    _keep_in_fp32_modules = ...
    _supports_flash_attn = ...
    def __init__(self, config: Blip2Config) -> None: ...
    def get_input_embeddings(self) -> nn.Module: ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Blip2VisionModelOutput: ...

@auto_docstring(custom_intro=...)
class Blip2ForConditionalGeneration(Blip2PreTrainedModel, GenerationMixin):
    config: Blip2Config
    main_input_name = ...
    _can_compile_fullgraph = ...
    _keep_in_fp32_modules = ...
    _supports_flash_attn = ...
    def __init__(self, config: Blip2Config) -> None: ...
    def get_input_embeddings(self):  # -> Any:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def set_output_embeddings(self, new_embeddings):  # -> None:
        ...
    def get_output_embeddings(self) -> nn.Module: ...
    def get_encoder(self):  # -> Any:
        ...
    def get_decoder(self):  # -> Any:
        ...
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        interpolate_pos_encoding: bool | None = ...,
        return_dict: bool | None = ...,
    ):  # -> tuple[Any, Any, Any] | Any:
        ...
    def get_placeholder_mask(
        self, input_ids: torch.LongTensor, inputs_embeds: torch.FloatTensor
    ):  # -> Any:
        ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor | None = ...,
        decoder_input_ids: torch.LongTensor | None = ...,
        decoder_attention_mask: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        interpolate_pos_encoding: bool = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Blip2ForConditionalGenerationModelOutput: ...
    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        interpolate_pos_encoding: bool = ...,
        **generate_kwargs,
    ) -> torch.LongTensor: ...

@auto_docstring(custom_intro=...)
class Blip2ForImageTextRetrieval(Blip2PreTrainedModel):
    main_input_name = ...
    _keep_in_fp32_modules = ...
    _supports_flash_attn = ...
    def __init__(self, config: Blip2Config) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor | None = ...,
        use_image_text_matching_head: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | Blip2ImageTextMatchingModelOutput: ...

__all__ = [
    "Blip2ForConditionalGeneration",
    "Blip2ForImageTextRetrieval",
    "Blip2Model",
    "Blip2PreTrainedModel",
    "Blip2QFormerModel",
    "Blip2TextModelWithProjection",
    "Blip2VisionModel",
    "Blip2VisionModelWithProjection",
]
