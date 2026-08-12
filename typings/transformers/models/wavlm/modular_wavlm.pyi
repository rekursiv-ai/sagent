from typing import Any
from torch import nn

import torch

from .configuration_wavlm import WavLMConfig
from ..wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2FeatureProjection,
    Wav2Vec2FeedForward,
    Wav2Vec2ForAudioFrameClassification,
    Wav2Vec2ForCTC,
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2ForXVector,
    Wav2Vec2Model,
    Wav2Vec2PositionalConvEmbedding,
    Wav2Vec2PreTrainedModel,
)
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import Wav2Vec2BaseModelOutput
from ...modeling_utils import PreTrainedModel

logger = ...

class WavLMPositionalConvEmbedding(Wav2Vec2PositionalConvEmbedding): ...
class WavLMFeatureProjection(Wav2Vec2FeatureProjection): ...

class WavLMAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = ...,
        num_buckets: int = ...,
        max_distance: int = ...,
        has_relative_position_bias: bool = ...,
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        position_bias: torch.Tensor | None = ...,
        output_attentions: bool = ...,
        index=...,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...
    def torch_multi_head_self_attention(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.LongTensor | torch.BoolTensor,
        gated_position_bias: torch.FloatTensor,
        output_attentions: bool,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]: ...
    def compute_bias(self, query_length: int, key_length: int) -> torch.FloatTensor: ...

class WavLMFeedForward(Wav2Vec2FeedForward): ...

class WavLMEncoderLayer(GradientCheckpointingLayer):
    def __init__(
        self, config: WavLMConfig, has_relative_position_bias: bool = ...
    ) -> None: ...
    def forward(
        self,
        hidden_states,
        attention_mask=...,
        position_bias=...,
        output_attentions=...,
        index=...,
    ):  # -> tuple[Any, Any, Any] | tuple[Any, Any]:
        ...

class WavLMEncoderLayerStableLayerNorm(GradientCheckpointingLayer):
    def __init__(
        self, config: WavLMConfig, has_relative_position_bias: bool = ...
    ) -> None: ...
    def forward(
        self,
        hidden_states,
        attention_mask=...,
        position_bias=...,
        output_attentions=...,
    ):  # -> tuple[Any, Any, Any] | tuple[Any, Any]:
        ...

class WavLMEncoder(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states,
        attention_mask=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
    ):  # -> tuple[Any | tuple[Any, ...] | tuple[()] | tuple[Any | None, ...], ...] | BaseModelOutput:
        ...

class WavLMEncoderStableLayerNorm(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states,
        attention_mask=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
    ):  # -> tuple[Any | tuple[Any, ...] | tuple[()] | tuple[Any | None, ...], ...] | BaseModelOutput:
        ...

class WavLMGumbelVectorQuantizer(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> tuple[Tensor | Any, Tensor]:
        ...

class WavLMPreTrainedModel(PreTrainedModel, Wav2Vec2PreTrainedModel):
    config: WavLMConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _supports_flex_attn = ...
    def init_adapter_layers(self): ...
    def load_adapter(self): ...

WavLMBaseModelOutput = Wav2Vec2BaseModelOutput

class WavLMModel(Wav2Vec2Model): ...
class WavLMForCTC(Wav2Vec2ForCTC): ...
class WavLMForSequenceClassification(Wav2Vec2ForSequenceClassification): ...
class WavLMForAudioFrameClassification(Wav2Vec2ForAudioFrameClassification): ...
class WavLMForXVector(Wav2Vec2ForXVector): ...

__all__ = [
    "WavLMForAudioFrameClassification",
    "WavLMForCTC",
    "WavLMForSequenceClassification",
    "WavLMForXVector",
    "WavLMModel",
    "WavLMPreTrainedModel",
]
