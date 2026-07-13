from dataclasses import dataclass

from torch import nn

import torch

from .configuration_wav2vec2 import Wav2Vec2Config
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    CausalLMOutput,
    MaskedLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    Wav2Vec2BaseModelOutput,
    XVectorOutput,
)
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import ModelOutput, auto_docstring

"""PyTorch Wav2Vec2 model."""
WAV2VEC2_ADAPTER_PT_FILE = ...
WAV2VEC2_ADAPTER_SAFE_FILE = ...
logger = ...
_HIDDEN_STATES_START_POSITION = ...

@dataclass
@auto_docstring(custom_intro=...)
class Wav2Vec2ForPreTrainingOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    projected_states: torch.FloatTensor | None = ...
    projected_quantized_states: torch.FloatTensor | None = ...
    codevector_perplexity: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...
    contrastive_loss: torch.FloatTensor | None = ...
    diversity_loss: torch.FloatTensor | None = ...

class Wav2Vec2NoLayerNormConvLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_id=...) -> None: ...
    def forward(self, hidden_states): ...

class Wav2Vec2LayerNormConvLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_id=...) -> None: ...
    def forward(self, hidden_states): ...

class Wav2Vec2GroupNormConvLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_id=...) -> None: ...
    def forward(self, hidden_states): ...

class Wav2Vec2PositionalConvEmbedding(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states): ...

class Wav2Vec2SamePadLayer(nn.Module):
    def __init__(self, num_conv_pos_embeddings) -> None: ...
    def forward(self, hidden_states): ...

class Wav2Vec2FeatureEncoder(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, input_values):  # -> Any:
        ...

class Wav2Vec2FeatureExtractor(Wav2Vec2FeatureEncoder):
    def __init__(self, config) -> None: ...

class Wav2Vec2FeatureProjection(nn.Module):
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

class Wav2Vec2Attention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = ...,
        is_decoder: bool = ...,
        bias: bool = ...,
        is_causal: bool = ...,
        config: Wav2Vec2Config | None = ...,
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

class Wav2Vec2FeedForward(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class Wav2Vec2EncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states, attention_mask=..., output_attentions=...
    ):  # -> tuple[Any, Any] | tuple[Any]:
        ...

class Wav2Vec2EncoderLayerStableLayerNorm(GradientCheckpointingLayer):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
    ):  # -> tuple[Tensor, Any] | tuple[Tensor]:
        ...

class Wav2Vec2Encoder(nn.Module):
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

class Wav2Vec2EncoderStableLayerNorm(nn.Module):
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

class Wav2Vec2GumbelVectorQuantizer(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states, mask_time_indices=...
    ):  # -> tuple[Tensor | Any, Tensor]:
        ...

class Wav2Vec2Adapter(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class Wav2Vec2AdapterLayer(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> Tensor:
        ...

class Wav2Vec2AttnAdapterLayer(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.FloatTensor):  # -> FloatTensor:
        ...

@auto_docstring
class Wav2Vec2PreTrainedModel(PreTrainedModel):
    config: Wav2Vec2Config
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _supports_flex_attn = ...
    def init_adapter_layers(self):  # -> None:
        ...
    def load_adapter(self, target_lang: str, force_load=..., **kwargs):  # -> None:
        ...

@auto_docstring
class Wav2Vec2Model(Wav2Vec2PreTrainedModel):
    def __init__(self, config: Wav2Vec2Config) -> None: ...
    def freeze_feature_extractor(self):  # -> None:
        ...
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
    ) -> tuple | Wav2Vec2BaseModelOutput: ...

@auto_docstring(custom_intro=...)
class Wav2Vec2ForPreTraining(Wav2Vec2PreTrainedModel):
    def __init__(self, config: Wav2Vec2Config) -> None: ...
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
        mask_time_indices: torch.BoolTensor | None = ...,
        sampled_negative_indices: torch.BoolTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | Wav2Vec2ForPreTrainingOutput: ...

@auto_docstring
class Wav2Vec2ForMaskedLM(Wav2Vec2PreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_values: torch.FloatTensor,
        attention_mask: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: torch.Tensor | None = ...,
    ) -> tuple | MaskedLMOutput: ...

@auto_docstring(custom_intro=...)
class Wav2Vec2ForCTC(Wav2Vec2PreTrainedModel):
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
class Wav2Vec2ForSequenceClassification(Wav2Vec2PreTrainedModel):
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

@auto_docstring
class Wav2Vec2ForAudioFrameClassification(Wav2Vec2PreTrainedModel):
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

@auto_docstring(custom_intro=...)
class Wav2Vec2ForXVector(Wav2Vec2PreTrainedModel):
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
    ) -> tuple | XVectorOutput: ...

__all__ = [
    "Wav2Vec2ForAudioFrameClassification",
    "Wav2Vec2ForCTC",
    "Wav2Vec2ForMaskedLM",
    "Wav2Vec2ForPreTraining",
    "Wav2Vec2ForSequenceClassification",
    "Wav2Vec2ForXVector",
    "Wav2Vec2Model",
    "Wav2Vec2PreTrainedModel",
]
