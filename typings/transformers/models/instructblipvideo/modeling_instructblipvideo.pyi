from dataclasses import dataclass
from typing import Any

from torch import nn

import torch

from .configuration_instructblipvideo import (
    InstructBlipVideoConfig,
    InstructBlipVideoQFormerConfig,
    InstructBlipVideoVisionConfig,
)
from ...generation import GenerationMixin
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import check_model_inputs

logger = ...

class InstructBlipVideoVisionEmbeddings(nn.Module):
    def __init__(self, config: InstructBlipVideoVisionConfig) -> None: ...
    def interpolate_pos_encoding(
        self, embeddings: torch.Tensor, height: int, width: int
    ) -> torch.Tensor: ...
    def forward(
        self, pixel_values: torch.FloatTensor, interpolate_pos_encoding: bool = ...
    ) -> torch.Tensor: ...

@auto_docstring
class InstructBlipVideoPreTrainedModel(PreTrainedModel):
    config: InstructBlipVideoConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _supports_attention_backend = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _supports_flex_attn = ...
    _can_compile_fullgraph = ...
    _no_split_modules = ...

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

class InstructBlipVideoAttention(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: torch.Tensor | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class InstructBlipVideoMLP(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class InstructBlipVideoEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: InstructBlipVideoConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.FloatTensor: ...

class InstructBlipVideoEncoder(nn.Module):
    def __init__(self, config: InstructBlipVideoConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        inputs_embeds,
        attention_mask: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutput: ...

class InstructBlipVideoVisionModel(InstructBlipVideoPreTrainedModel):
    main_input_name = ...
    config: InstructBlipVideoVisionConfig
    _can_record_outputs = ...
    def __init__(self, config: InstructBlipVideoVisionConfig) -> None: ...
    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        interpolate_pos_encoding: bool = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling: ...
    def get_input_embeddings(self):  # -> InstructBlipVideoVisionEmbeddings:
        ...

class InstructBlipVideoQFormerMultiHeadAttention(nn.Module):
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

class InstructBlipVideoQFormerSelfOutput(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor: ...

class InstructBlipVideoQFormerAttention(nn.Module):
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

class InstructBlipVideoQFormerIntermediate(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class InstructBlipVideoQFormerOutput(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor: ...

class InstructBlipVideoQFormerLayer(GradientCheckpointingLayer):
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

class InstructBlipVideoQFormerEncoder(nn.Module):
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

class InstructBlipVideoQFormerEmbeddings(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        input_ids=...,
        position_ids=...,
        query_embeds=...,
        past_key_values_length=...,
    ):  # -> Any:
        ...

class InstructBlipVideoQFormerModel(InstructBlipVideoPreTrainedModel):
    _supports_attention_backend = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _supports_flex_attn = ...
    _can_record_outputs = ...
    def __init__(self, config: InstructBlipVideoQFormerConfig) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
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
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        query_embeds: torch.Tensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        encoder_hidden_states: torch.FloatTensor | None = ...,
        encoder_attention_mask: torch.FloatTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.FloatTensor] | BaseModelOutputWithPoolingAndCrossAttentions: ...

@dataclass
@auto_docstring(custom_intro=...)
class InstructBlipVideoForConditionalGenerationModelOutput(ModelOutput):
    loss: tuple[torch.FloatTensor] | None = ...
    logits: tuple[torch.FloatTensor] | None = ...
    vision_outputs: torch.FloatTensor | None = ...
    qformer_outputs: tuple[torch.FloatTensor] | None = ...
    language_model_outputs: tuple[torch.FloatTensor] | None = ...
    def to_tuple(self) -> tuple[Any]: ...

@auto_docstring(custom_intro=...)
class InstructBlipVideoModel(InstructBlipVideoPreTrainedModel):
    main_input_name = ...
    _keep_in_fp32_modules = ...
    def __init__(self, config: InstructBlipVideoConfig) -> None: ...
    def get_input_embeddings(self):  # -> Any:
        ...
    def set_input_embeddings(self, value):  # -> None:
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
        qformer_input_ids: torch.FloatTensor,
        qformer_attention_mask: torch.LongTensor | None = ...,
        input_ids: torch.FloatTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        decoder_input_ids: torch.LongTensor | None = ...,
        decoder_attention_mask: torch.LongTensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        interpolate_pos_encoding: bool = ...,
        use_cache: bool | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple | InstructBlipVideoForConditionalGenerationModelOutput: ...

@auto_docstring(custom_intro=...)
class InstructBlipVideoForConditionalGeneration(
    InstructBlipVideoPreTrainedModel, GenerationMixin
):
    config: InstructBlipVideoConfig
    main_input_name = ...
    _can_compile_fullgraph = ...
    _keep_in_fp32_modules = ...
    def __init__(self, config: InstructBlipVideoConfig) -> None: ...
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
        qformer_input_ids: torch.LongTensor,
        qformer_attention_mask: torch.LongTensor | None = ...,
        interpolate_pos_encoding: bool | None = ...,
        return_dict: bool | None = ...,
    ):  # -> None:
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
        qformer_input_ids: torch.FloatTensor,
        qformer_attention_mask: torch.LongTensor | None = ...,
        input_ids: torch.FloatTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        decoder_input_ids: torch.LongTensor | None = ...,
        decoder_attention_mask: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        labels: torch.LongTensor | None = ...,
        return_dict: bool | None = ...,
        interpolate_pos_encoding: bool = ...,
        use_cache: bool | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | InstructBlipVideoForConditionalGenerationModelOutput: ...
    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.FloatTensor,
        qformer_input_ids: torch.LongTensor | None = ...,
        qformer_attention_mask: torch.LongTensor | None = ...,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        interpolate_pos_encoding: bool = ...,
        **generate_kwargs,
    ) -> torch.LongTensor: ...
    def get_video_features(
        self,
        pixel_values: torch.FloatTensor,
        qformer_input_ids: torch.LongTensor,
        qformer_attention_mask: torch.LongTensor | None = ...,
        interpolate_pos_encoding: bool | None = ...,
        return_dict: bool | None = ...,
    ):  # -> tuple[Any, Any, Any] | Any:
        ...

__all__ = [
    "InstructBlipVideoForConditionalGeneration",
    "InstructBlipVideoModel",
    "InstructBlipVideoPreTrainedModel",
    "InstructBlipVideoQFormerModel",
    "InstructBlipVideoVisionModel",
]
