from dataclasses import dataclass
from typing import Any

from torch import nn

import torch

from .configuration_wav2vec2_conformer import Wav2Vec2ConformerConfig
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    CausalLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    Wav2Vec2BaseModelOutput,
    XVectorOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring

@dataclass
@auto_docstring(custom_intro=...)
class Wav2Vec2ConformerForPreTrainingOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    projected_states: torch.FloatTensor | None = ...
    projected_quantized_states: torch.FloatTensor | None = ...
    codevector_perplexity: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...
    contrastive_loss: torch.FloatTensor | None = ...
    diversity_loss: torch.FloatTensor | None = ...

class Wav2Vec2ConformerSamePadLayer(nn.Module):
    def __init__(self, num_conv_pos_embeddings) -> None: ...
    def forward(self, hidden_states): ...

class Wav2Vec2ConformerPositionalConvEmbedding(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states): ...

class Wav2Vec2ConformerRotaryPositionalEmbedding(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> Tensor:
        ...

class Wav2Vec2ConformerRelPositionalEmbedding(nn.Module):
    def __init__(self, config) -> None: ...
    def extend_pe(self, x):  # -> None:
        ...
    def forward(self, hidden_states: torch.Tensor):  # -> Tensor:
        ...

class Wav2Vec2ConformerNoLayerNormConvLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_id=...) -> None: ...
    def forward(self, hidden_states): ...

class Wav2Vec2ConformerLayerNormConvLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_id=...) -> None: ...
    def forward(self, hidden_states): ...

class Wav2Vec2ConformerGroupNormConvLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_id=...) -> None: ...
    def forward(self, hidden_states): ...

class Wav2Vec2ConformerFeatureEncoder(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, input_values):  # -> Any:
        ...

class Wav2Vec2ConformerFeatureProjection(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> tuple[Any, Any]:
        ...

class Wav2Vec2ConformerFeedForward(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class Wav2Vec2ConformerConvolutionModule(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class Wav2Vec2ConformerSelfAttention(nn.Module):
    def __init__(self, config) -> None: ...
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

class Wav2Vec2ConformerEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states,
        attention_mask: torch.Tensor | None = ...,
        relative_position_embeddings: torch.Tensor | None = ...,
        output_attentions: bool = ...,
    ):  # -> tuple[Any, Any]:
        ...

class Wav2Vec2ConformerEncoder(nn.Module):
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

class Wav2Vec2ConformerGumbelVectorQuantizer(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states, mask_time_indices=...
    ):  # -> tuple[Tensor | Any, Tensor]:
        ...

class Wav2Vec2ConformerAdapter(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class Wav2Vec2ConformerAdapterLayer(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> Tensor:
        ...

@auto_docstring
class Wav2Vec2ConformerPreTrainedModel(PreTrainedModel):
    config: Wav2Vec2ConformerConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...

Wav2Vec2ConformerBaseModelOutput = Wav2Vec2BaseModelOutput

@auto_docstring
class Wav2Vec2ConformerModel(Wav2Vec2ConformerPreTrainedModel):
    def __init__(self, config: Wav2Vec2ConformerConfig) -> None: ...
    def freeze_feature_encoder(self):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_values: torch.Tensor | None,
        attention_mask: torch.Tensor | None = ...,
        mask_time_indices: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | Wav2Vec2ConformerBaseModelOutput: ...

@auto_docstring(custom_intro=...)
class Wav2Vec2ConformerForPreTraining(Wav2Vec2ConformerPreTrainedModel):
    def __init__(self, config: Wav2Vec2ConformerConfig) -> None: ...
    def set_gumbel_temperature(self, temperature: int):  # -> None:
        ...
    def freeze_feature_encoder(self):  # -> None:
        ...
    @staticmethod
    def compute_contrastive_logits(
        target_features: torch.FloatTensor,
        negative_features: torch.FloatTensor,
        predicted_features: torch.FloatTensor,
        temperature: int = ...,
    ):  # -> Tensor:
        ...
    @auto_docstring
    def forward(
        self,
        input_values: torch.Tensor | None,
        attention_mask: torch.Tensor | None = ...,
        mask_time_indices: torch.BoolTensor | None = ...,
        sampled_negative_indices: torch.BoolTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | Wav2Vec2ConformerForPreTrainingOutput: ...

_HIDDEN_STATES_START_POSITION = ...

@auto_docstring(custom_intro=...)
class Wav2Vec2ConformerForCTC(Wav2Vec2ConformerPreTrainedModel):
    def __init__(self, config, target_lang: str | None = ...) -> None: ...
    def freeze_feature_encoder(self):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_values: torch.Tensor | None,
        attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: torch.Tensor | None = ...,
    ) -> tuple | CausalLMOutput: ...

@auto_docstring(custom_intro=...)
class Wav2Vec2ConformerForSequenceClassification(Wav2Vec2ConformerPreTrainedModel):
    def __init__(self, config) -> None: ...
    def freeze_feature_encoder(self):  # -> None:
        ...
    def freeze_base_model(self):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_values: torch.Tensor | None,
        attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: torch.Tensor | None = ...,
    ) -> tuple | SequenceClassifierOutput: ...

@auto_docstring
class Wav2Vec2ConformerForAudioFrameClassification(Wav2Vec2ConformerPreTrainedModel):
    def __init__(self, config) -> None: ...
    def freeze_feature_encoder(self):  # -> None:
        ...
    def freeze_base_model(self):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_values: torch.Tensor | None,
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
class Wav2Vec2ConformerForXVector(Wav2Vec2ConformerPreTrainedModel):
    def __init__(self, config) -> None: ...
    def freeze_feature_encoder(self):  # -> None:
        ...
    def freeze_base_model(self):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_values: torch.Tensor | None,
        attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: torch.Tensor | None = ...,
    ) -> tuple | XVectorOutput: ...

__all__ = [
    "Wav2Vec2ConformerForAudioFrameClassification",
    "Wav2Vec2ConformerForCTC",
    "Wav2Vec2ConformerForPreTraining",
    "Wav2Vec2ConformerForSequenceClassification",
    "Wav2Vec2ConformerForXVector",
    "Wav2Vec2ConformerModel",
    "Wav2Vec2ConformerPreTrainedModel",
]
