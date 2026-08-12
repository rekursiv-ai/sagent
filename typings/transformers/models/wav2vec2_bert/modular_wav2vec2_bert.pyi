from typing import Any
from torch import nn

import torch

from .configuration_wav2vec2_bert import Wav2Vec2BertConfig
from ..wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2FeedForward,
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2Model,
)
from ..wav2vec2_conformer.modeling_wav2vec2_conformer import (
    Wav2Vec2ConformerForAudioFrameClassification,
    Wav2Vec2ConformerForCTC,
    Wav2Vec2ConformerForXVector,
    Wav2Vec2ConformerRelPositionalEmbedding,
    Wav2Vec2ConformerRotaryPositionalEmbedding,
    Wav2Vec2ConformerSelfAttention,
)
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    CausalLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    Wav2Vec2BaseModelOutput,
    XVectorOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring

logger = ...
_HIDDEN_STATES_START_POSITION = ...

class Wav2Vec2BertRotaryPositionalEmbedding(Wav2Vec2ConformerRotaryPositionalEmbedding):
    def __init__(self, config) -> None: ...

class Wav2Vec2BertRelPositionalEmbedding(Wav2Vec2ConformerRelPositionalEmbedding): ...

class Wav2Vec2BertFeatureProjection(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> tuple[Any, Any]:
        ...

class Wav2Vec2BertFeedForward(Wav2Vec2FeedForward):
    def __init__(self, config, act_fn=..., hidden_size=...) -> None: ...

class Wav2Vec2BertConvolutionModule(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states, attention_mask=...):  # -> Any:
        ...

class Wav2Vec2BertSelfAttention(Wav2Vec2ConformerSelfAttention, nn.Module):
    def __init__(self, config, is_adapter_attention=...) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        relative_position_embeddings: torch.Tensor | None = ...,
        output_attentions: bool = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class Wav2Vec2BertEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states,
        attention_mask: torch.Tensor | None = ...,
        relative_position_embeddings: torch.Tensor | None = ...,
        output_attentions: bool = ...,
        conv_attention_mask: torch.Tensor | None = ...,
    ):  # -> tuple[Any, Any]:
        ...

class Wav2Vec2BertEncoder(nn.Module):
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

class Wav2Vec2BertAdapter(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states, attention_mask=...):  # -> Any:
        ...

class Wav2Vec2BertAdapterLayer(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states,
        attention_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
        sub_sampled_lengths: torch.Tensor | None = ...,
    ):  # -> Any:
        ...

@auto_docstring
class Wav2Vec2BertPreTrainedModel(PreTrainedModel):
    config: Wav2Vec2BertConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...

Wav2Vec2BertBaseModelOutput = Wav2Vec2BaseModelOutput

class Wav2Vec2BertModel(Wav2Vec2Model, Wav2Vec2BertPreTrainedModel):
    def __init__(self, config: Wav2Vec2BertConfig) -> None: ...
    def freeze_feature_extractor(self): ...
    def freeze_feature_encoder(self): ...
    def forward(
        self,
        input_features: torch.Tensor | None,
        attention_mask: torch.Tensor | None = ...,
        mask_time_indices: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | Wav2Vec2BertBaseModelOutput: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple | Wav2Vec2BertBaseModelOutput: ...

class Wav2Vec2BertForCTC(Wav2Vec2ConformerForCTC):
    def __init__(self, config, target_lang: str | None = ...) -> None: ...
    def freeze_feature_encoder(self): ...
    def forward(
        self,
        input_features: torch.Tensor | None,
        attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: torch.Tensor | None = ...,
    ) -> tuple | CausalLMOutput: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple | CausalLMOutput: ...

class Wav2Vec2BertForSequenceClassification(Wav2Vec2ForSequenceClassification):
    def __init__(self, config) -> None: ...
    def freeze_feature_extractor(self): ...
    def freeze_feature_encoder(self): ...
    def freeze_base_model(self):  # -> None:
        ...
    def forward(
        self,
        input_features: torch.Tensor | None,
        attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: torch.Tensor | None = ...,
    ) -> tuple | SequenceClassifierOutput: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple | SequenceClassifierOutput: ...

class Wav2Vec2BertForAudioFrameClassification(
    Wav2Vec2ConformerForAudioFrameClassification
):
    def __init__(self, config) -> None: ...
    def freeze_feature_encoder(self): ...
    def forward(
        self,
        input_features: torch.Tensor | None,
        attention_mask: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | TokenClassifierOutput: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple | TokenClassifierOutput: ...

class Wav2Vec2BertForXVector(Wav2Vec2ConformerForXVector):
    def __init__(self, config) -> None: ...
    def freeze_feature_encoder(self): ...
    def forward(
        self,
        input_features: torch.Tensor | None,
        attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: torch.Tensor | None = ...,
    ) -> tuple | XVectorOutput: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple | XVectorOutput: ...

__all__ = [
    "Wav2Vec2BertForAudioFrameClassification",
    "Wav2Vec2BertForCTC",
    "Wav2Vec2BertForSequenceClassification",
    "Wav2Vec2BertForXVector",
    "Wav2Vec2BertModel",
    "Wav2Vec2BertPreTrainedModel",
]
