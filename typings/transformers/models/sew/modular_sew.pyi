from torch import nn

import torch

from .configuration_sew import SEWConfig
from ..wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Attention,
    Wav2Vec2EncoderLayer,
    Wav2Vec2FeatureEncoder,
    Wav2Vec2FeedForward,
    Wav2Vec2ForCTC,
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2GroupNormConvLayer,
    Wav2Vec2LayerNormConvLayer,
    Wav2Vec2NoLayerNormConvLayer,
    Wav2Vec2SamePadLayer,
)
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring

"""PyTorch SEW model."""
_HIDDEN_STATES_START_POSITION = ...

class SEWNoLayerNormConvLayer(Wav2Vec2NoLayerNormConvLayer): ...
class SEWLayerNormConvLayer(Wav2Vec2LayerNormConvLayer): ...
class SEWGroupNormConvLayer(Wav2Vec2GroupNormConvLayer): ...

class SEWPositionalConvEmbedding(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states): ...

class SEWSamePadLayer(Wav2Vec2SamePadLayer): ...

class SEWUpsampling(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states): ...

class SEWFeatureEncoder(Wav2Vec2FeatureEncoder): ...

class SEWFeatureExtractor(SEWFeatureEncoder):
    def __init__(self, config) -> None: ...

class SEWAttention(Wav2Vec2Attention): ...
class SEWFeedForward(Wav2Vec2FeedForward): ...
class SEWEncoderLayer(Wav2Vec2EncoderLayer): ...

class SEWEncoder(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states,
        attention_mask=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
    ):  # -> tuple[Tensor | Any | tuple[Any, ...] | tuple[()] | tuple[Any | None, ...], ...] | BaseModelOutput:
        ...

@auto_docstring
class SEWPreTrainedModel(PreTrainedModel):
    config: SEWConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _supports_flex_attn = ...

@auto_docstring
class SEWModel(SEWPreTrainedModel):
    def __init__(self, config: SEWConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_values: torch.Tensor | None,
        attention_mask: torch.Tensor | None = ...,
        mask_time_indices: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutput: ...

class SEWForCTC(Wav2Vec2ForCTC): ...
class SEWForSequenceClassification(Wav2Vec2ForSequenceClassification): ...

__all__ = [
    "SEWForCTC",
    "SEWForSequenceClassification",
    "SEWModel",
    "SEWPreTrainedModel",
]
