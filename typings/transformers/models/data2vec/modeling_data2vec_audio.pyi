from torch import nn

import torch

from .configuration_data2vec_audio import Data2VecAudioConfig
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    CausalLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    Wav2Vec2BaseModelOutput,
    XVectorOutput,
)
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import auto_docstring

class Data2VecAudioConvLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_id=...) -> None: ...
    def forward(self, hidden_states): ...

class Data2VecAudioPadLayer(nn.Module):
    def __init__(self, num_conv_pos_embeddings) -> None: ...
    def forward(self, hidden_states): ...

class Data2VecAudioPositionalConvLayer(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states): ...

class Data2VecAudioPositionalConvEmbedding(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class Data2VecAudioFeatureEncoder(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, input_values):  # -> Any:
        ...

class Data2VecAudioFeatureProjection(nn.Module):
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

class Data2VecAudioAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = ...,
        is_decoder: bool = ...,
        bias: bool = ...,
        is_causal: bool = ...,
        config: Data2VecAudioConfig | None = ...,
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

class Data2VecAudioFeedForward(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class Data2VecAudioEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states, attention_mask=..., output_attentions=...
    ):  # -> tuple[Any, Any] | tuple[Any]:
        ...

class Data2VecAudioEncoder(nn.Module):
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

class Data2VecAudioAdapterLayer(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> Tensor:
        ...

class Data2VecAudioAdapter(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

@auto_docstring
class Data2VecAudioPreTrainedModel(PreTrainedModel):
    config: Data2VecAudioConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _supports_flex_attn = ...

Data2VecAudioBaseModelOutput = Wav2Vec2BaseModelOutput

@auto_docstring
class Data2VecAudioModel(Data2VecAudioPreTrainedModel):
    def __init__(self, config: Data2VecAudioConfig) -> None: ...
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
    ) -> tuple | Data2VecAudioBaseModelOutput: ...

_HIDDEN_STATES_START_POSITION = ...

@auto_docstring(custom_intro=...)
class Data2VecAudioForCTC(Data2VecAudioPreTrainedModel):
    def __init__(self, config) -> None: ...
    def freeze_feature_extractor(self):  # -> None:
        ...
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
class Data2VecAudioForSequenceClassification(Data2VecAudioPreTrainedModel):
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
class Data2VecAudioForAudioFrameClassification(Data2VecAudioPreTrainedModel):
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
class Data2VecAudioForXVector(Data2VecAudioPreTrainedModel):
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
    "Data2VecAudioForAudioFrameClassification",
    "Data2VecAudioForCTC",
    "Data2VecAudioForSequenceClassification",
    "Data2VecAudioForXVector",
    "Data2VecAudioModel",
    "Data2VecAudioPreTrainedModel",
]
