from dataclasses import dataclass

from torch import nn

import torch

from .configuration_unispeech import UniSpeechConfig
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    CausalLMOutput,
    ModelOutput,
    SequenceClassifierOutput,
    Wav2Vec2BaseModelOutput,
)
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import auto_docstring

logger = ...

@dataclass
@auto_docstring(custom_intro=...)
class UniSpeechForPreTrainingOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    projected_states: torch.FloatTensor | None = ...
    projected_quantized_states: torch.FloatTensor | None = ...
    codevector_perplexity: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...

class UniSpeechSamePadLayer(nn.Module):
    def __init__(self, num_conv_pos_embeddings) -> None: ...
    def forward(self, hidden_states): ...

class UniSpeechPositionalConvEmbedding(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states): ...

class UniSpeechNoLayerNormConvLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_id=...) -> None: ...
    def forward(self, hidden_states): ...

class UniSpeechLayerNormConvLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_id=...) -> None: ...
    def forward(self, hidden_states): ...

class UniSpeechGroupNormConvLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_id=...) -> None: ...
    def forward(self, hidden_states): ...

class UniSpeechFeatureEncoder(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, input_values):  # -> Any:
        ...

class UniSpeechFeatureProjection(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> tuple[Any, Any]:
        ...

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None = ...,
    dropout: float = ...,
    head_mask: torch.Tensor | None = ...,
    **kwargs,
):  # -> tuple[Tensor, Tensor]:
    ...

class UniSpeechAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = ...,
        is_decoder: bool = ...,
        bias: bool = ...,
        is_causal: bool = ...,
        config: UniSpeechConfig | None = ...,
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        layer_head_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class UniSpeechFeedForward(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class UniSpeechEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states, attention_mask=..., output_attentions=...
    ):  # -> tuple[Any, Any] | tuple[Any]:
        ...

class UniSpeechEncoder(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states: torch.tensor,
        attention_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any | tuple[Any, ...] | tuple[()] | tuple[Any | None, ...], ...] | BaseModelOutput:
        ...

class UniSpeechAttnAdapterLayer(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.FloatTensor):  # -> FloatTensor:
        ...

class UniSpeechEncoderLayerStableLayerNorm(GradientCheckpointingLayer):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
    ):  # -> tuple[Tensor, Any] | tuple[Tensor]:
        ...

class UniSpeechEncoderStableLayerNorm(nn.Module):
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

class UniSpeechGumbelVectorQuantizer(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> tuple[Tensor | Any, Tensor]:
        ...

@auto_docstring
class UniSpeechPreTrainedModel(PreTrainedModel):
    config: UniSpeechConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _supports_flex_attn = ...

UniSpeechBaseModelOutput = Wav2Vec2BaseModelOutput

@auto_docstring
class UniSpeechModel(UniSpeechPreTrainedModel):
    def __init__(self, config: UniSpeechConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_values: torch.Tensor | None,
        attention_mask: torch.Tensor | None = ...,
        mask_time_indices: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | UniSpeechBaseModelOutput: ...

@auto_docstring(custom_intro=...)
class UniSpeechForPreTraining(UniSpeechPreTrainedModel):
    def __init__(self, config: UniSpeechConfig) -> None: ...
    def set_gumbel_temperature(self, temperature: int):  # -> None:
        ...
    def freeze_feature_extractor(self):  # -> None:
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
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | UniSpeechForPreTrainingOutput: ...

_HIDDEN_STATES_START_POSITION = ...

@auto_docstring(custom_intro=...)
class UniSpeechForCTC(UniSpeechPreTrainedModel):
    def __init__(self, config, target_lang: str | None = ...) -> None: ...
    def tie_weights(self):  # -> None:
        ...
    def freeze_feature_extractor(self):  # -> None:
        ...
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
    ) -> tuple | CausalLMOutput: ...

@auto_docstring(custom_intro=...)
class UniSpeechForSequenceClassification(UniSpeechPreTrainedModel):
    def __init__(self, config) -> None: ...
    def freeze_feature_extractor(self):  # -> None:
        ...
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

__all__ = [
    "UniSpeechForCTC",
    "UniSpeechForPreTraining",
    "UniSpeechForSequenceClassification",
    "UniSpeechModel",
    "UniSpeechPreTrainedModel",
]
