from torch import nn

import torch

from .configuration_wavlm import WavLMConfig
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

class WavLMSamePadLayer(nn.Module):
    def __init__(self, num_conv_pos_embeddings) -> None: ...
    def forward(self, hidden_states): ...

class WavLMPositionalConvEmbedding(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states): ...

class WavLMFeatureProjection(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> tuple[Any, Any]:
        ...

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
    def torch_multi_head_self_attention(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.LongTensor | torch.BoolTensor,
        gated_position_bias: torch.FloatTensor,
        output_attentions: bool,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]: ...
    def compute_bias(self, query_length: int, key_length: int) -> torch.FloatTensor: ...

class WavLMFeedForward(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

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

@auto_docstring
class WavLMPreTrainedModel(PreTrainedModel):
    config: WavLMConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _supports_flex_attn = ...

class WavLMNoLayerNormConvLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_id=...) -> None: ...
    def forward(self, hidden_states): ...

class WavLMLayerNormConvLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_id=...) -> None: ...
    def forward(self, hidden_states): ...

class WavLMGroupNormConvLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_id=...) -> None: ...
    def forward(self, hidden_states): ...

class WavLMFeatureEncoder(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, input_values):  # -> Any:
        ...

class WavLMAdapterLayer(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> Tensor:
        ...

class WavLMAdapter(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

WavLMBaseModelOutput = Wav2Vec2BaseModelOutput

@auto_docstring
class WavLMModel(WavLMPreTrainedModel):
    def __init__(self, config: WavLMConfig) -> None: ...
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
    ) -> tuple | WavLMBaseModelOutput: ...

_HIDDEN_STATES_START_POSITION = ...

@auto_docstring(custom_intro=...)
class WavLMForCTC(WavLMPreTrainedModel):
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
class WavLMForSequenceClassification(WavLMPreTrainedModel):
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
class WavLMForAudioFrameClassification(WavLMPreTrainedModel):
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
class WavLMForXVector(WavLMPreTrainedModel):
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
    "WavLMForAudioFrameClassification",
    "WavLMForCTC",
    "WavLMForSequenceClassification",
    "WavLMForXVector",
    "WavLMModel",
    "WavLMPreTrainedModel",
]
