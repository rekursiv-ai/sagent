from typing import Any
from torch import nn

import torch

from .configuration_wav2vec2_bert import Wav2Vec2BertConfig
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

class Wav2Vec2BertRotaryPositionalEmbedding(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> Tensor:
        ...

class Wav2Vec2BertRelPositionalEmbedding(nn.Module):
    def __init__(self, config) -> None: ...
    def extend_pe(self, x):  # -> None:
        ...
    def forward(self, hidden_states: torch.Tensor):  # -> Tensor:
        ...

class Wav2Vec2BertFeatureProjection(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> tuple[Any, Any]:
        ...

class Wav2Vec2BertFeedForward(nn.Module):
    def __init__(self, config, act_fn=..., hidden_size=...) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class Wav2Vec2BertConvolutionModule(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states, attention_mask=...):  # -> Any:
        ...

class Wav2Vec2BertSelfAttention(nn.Module):
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

@auto_docstring
class Wav2Vec2BertModel(Wav2Vec2BertPreTrainedModel):
    def __init__(self, config: Wav2Vec2BertConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_features: torch.Tensor | None,
        attention_mask: torch.Tensor | None = ...,
        mask_time_indices: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | Wav2Vec2BertBaseModelOutput: ...

_HIDDEN_STATES_START_POSITION = ...

@auto_docstring(custom_intro=...)
class Wav2Vec2BertForCTC(Wav2Vec2BertPreTrainedModel):
    def __init__(self, config, target_lang: str | None = ...) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_features: torch.Tensor | None,
        attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: torch.Tensor | None = ...,
    ) -> tuple | CausalLMOutput: ...

@auto_docstring(custom_intro=...)
class Wav2Vec2BertForSequenceClassification(Wav2Vec2BertPreTrainedModel):
    def __init__(self, config) -> None: ...
    def freeze_base_model(self):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_features: torch.Tensor | None,
        attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: torch.Tensor | None = ...,
    ) -> tuple | SequenceClassifierOutput: ...

@auto_docstring
class Wav2Vec2BertForAudioFrameClassification(Wav2Vec2BertPreTrainedModel):
    def __init__(self, config) -> None: ...
    def freeze_base_model(self):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_features: torch.Tensor | None,
        attention_mask: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | TokenClassifierOutput: ...

class AMSoftmaxLoss(nn.Module):
    def __init__(self, input_dim, num_labels, scale=..., margin=...) -> None: ...
    def forward(self, hidden_states, labels):  # -> Any:
        ...

class TDNNLayer(nn.Module):
    def __init__(self, config, layer_id=...) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

@auto_docstring(custom_intro=...)
class Wav2Vec2BertForXVector(Wav2Vec2BertPreTrainedModel):
    def __init__(self, config) -> None: ...
    def freeze_base_model(self):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_features: torch.Tensor | None,
        attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: torch.Tensor | None = ...,
    ) -> tuple | XVectorOutput: ...

__all__ = [
    "Wav2Vec2BertForAudioFrameClassification",
    "Wav2Vec2BertForCTC",
    "Wav2Vec2BertForSequenceClassification",
    "Wav2Vec2BertForXVector",
    "Wav2Vec2BertModel",
    "Wav2Vec2BertPreTrainedModel",
]
