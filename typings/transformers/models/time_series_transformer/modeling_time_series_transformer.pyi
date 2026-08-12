from typing import Any
from torch import nn

import torch

from .configuration_time_series_transformer import TimeSeriesTransformerConfig
from ...cache_utils import Cache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    SampleTSPredictionOutput,
    Seq2SeqTSModelOutput,
)
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import auto_docstring
from ...utils.deprecation import deprecate_kwarg

"""PyTorch Time Series Transformer model."""
logger = ...

class TimeSeriesFeatureEmbedder(nn.Module):
    def __init__(self, cardinalities: list[int], embedding_dims: list[int]) -> None: ...
    def forward(self, features: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class TimeSeriesStdScaler(nn.Module):
    def __init__(self, config: TimeSeriesTransformerConfig) -> None: ...
    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

class TimeSeriesMeanScaler(nn.Module):
    def __init__(self, config: TimeSeriesTransformerConfig) -> None: ...
    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

class TimeSeriesNOPScaler(nn.Module):
    def __init__(self, config: TimeSeriesTransformerConfig) -> None: ...
    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor | None = ...
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

def nll(
    input: torch.distributions.Distribution, target: torch.Tensor
) -> torch.Tensor: ...
def weighted_average(
    input_tensor: torch.Tensor, weights: torch.Tensor | None = ..., dim=...
) -> torch.Tensor: ...

class TimeSeriesSinusoidalPositionalEmbedding(nn.Embedding):
    def __init__(
        self, num_positions: int, embedding_dim: int, padding_idx: int | None = ...
    ) -> None: ...
    @torch.no_grad()
    def forward(
        self,
        input_ids_shape: torch.Size,
        past_key_values_length: int = ...,
        position_ids: torch.Tensor | None = ...,
    ) -> torch.Tensor: ...

class TimeSeriesValueEmbedding(nn.Module):
    def __init__(self, feature_size, d_model) -> None: ...
    def forward(self, x):  # -> Any:
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

class TimeSeriesTransformerAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = ...,
        is_decoder: bool = ...,
        bias: bool = ...,
        is_causal: bool = ...,
        config: TimeSeriesTransformerConfig | None = ...,
        layer_idx: int | None = ...,
    ) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        attention_mask: torch.Tensor | None = ...,
        layer_head_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
        cache_position: torch.Tensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class TimeSeriesTransformerEncoderLayer(GradientCheckpointingLayer):
    def __init__(
        self, config: TimeSeriesTransformerConfig, layer_idx: int | None = ...
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        layer_head_mask: torch.FloatTensor,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor | None]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.FloatTensor, torch.FloatTensor | None]: ...

class TimeSeriesTransformerDecoderLayer(GradientCheckpointingLayer):
    def __init__(
        self, config: TimeSeriesTransformerConfig, layer_idx: int | None = ...
    ) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        encoder_hidden_states: torch.Tensor | None = ...,
        encoder_attention_mask: torch.Tensor | None = ...,
        layer_head_mask: torch.Tensor | None = ...,
        cross_attn_layer_head_mask: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        output_attentions: bool | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple[
        torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None
    ]: ...

@auto_docstring
class TimeSeriesTransformerPreTrainedModel(PreTrainedModel):
    config: TimeSeriesTransformerConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _supports_flex_attn = ...

class TimeSeriesTransformerEncoder(TimeSeriesTransformerPreTrainedModel):
    def __init__(self, config: TimeSeriesTransformerConfig) -> None: ...
    def forward(
        self,
        attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutput: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple | BaseModelOutput: ...

class TimeSeriesTransformerDecoder(TimeSeriesTransformerPreTrainedModel):
    def __init__(self, config: TimeSeriesTransformerConfig) -> None: ...
    def forward(
        self,
        attention_mask: torch.Tensor | None = ...,
        encoder_hidden_states: torch.FloatTensor | None = ...,
        encoder_attention_mask: torch.LongTensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        cross_attn_head_mask: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
    ) -> tuple | BaseModelOutputWithPastAndCrossAttentions: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple | BaseModelOutputWithPastAndCrossAttentions: ...

@auto_docstring
class TimeSeriesTransformerModel(TimeSeriesTransformerPreTrainedModel):
    def __init__(self, config: TimeSeriesTransformerConfig) -> None: ...
    def get_lagged_subsequences(
        self, sequence: torch.Tensor, subsequences_length: int, shift: int = ...
    ) -> torch.Tensor: ...
    def create_network_inputs(
        self,
        past_values: torch.Tensor,
        past_time_features: torch.Tensor,
        static_categorical_features: torch.Tensor | None = ...,
        static_real_features: torch.Tensor | None = ...,
        past_observed_mask: torch.Tensor | None = ...,
        future_values: torch.Tensor | None = ...,
        future_time_features: torch.Tensor | None = ...,
    ):  # -> tuple[Tensor, Any, Any, Tensor]:
        ...
    def get_encoder(self):  # -> TimeSeriesTransformerEncoder:
        ...
    @auto_docstring
    def forward(
        self,
        past_values: torch.Tensor,
        past_time_features: torch.Tensor,
        past_observed_mask: torch.Tensor,
        static_categorical_features: torch.Tensor | None = ...,
        static_real_features: torch.Tensor | None = ...,
        future_values: torch.Tensor | None = ...,
        future_time_features: torch.Tensor | None = ...,
        decoder_attention_mask: torch.LongTensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        decoder_head_mask: torch.Tensor | None = ...,
        cross_attn_head_mask: torch.Tensor | None = ...,
        encoder_outputs: list[torch.FloatTensor] | None = ...,
        past_key_values: Cache | None = ...,
        output_hidden_states: bool | None = ...,
        output_attentions: bool | None = ...,
        use_cache: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
    ) -> Seq2SeqTSModelOutput | tuple: ...

@auto_docstring
class TimeSeriesTransformerForPrediction(TimeSeriesTransformerPreTrainedModel):
    def __init__(self, config: TimeSeriesTransformerConfig) -> None: ...
    def output_params(self, dec_output):  # -> Any:
        ...
    def get_encoder(self):  # -> TimeSeriesTransformerEncoder:
        ...
    def get_decoder(self):  # -> PreTrainedModel | TimeSeriesTransformerModel:
        ...
    @torch.jit.ignore
    def output_distribution(
        self, params, loc=..., scale=..., trailing_n=...
    ) -> torch.distributions.Distribution: ...
    @auto_docstring
    def forward(
        self,
        past_values: torch.Tensor,
        past_time_features: torch.Tensor,
        past_observed_mask: torch.Tensor,
        static_categorical_features: torch.Tensor | None = ...,
        static_real_features: torch.Tensor | None = ...,
        future_values: torch.Tensor | None = ...,
        future_time_features: torch.Tensor | None = ...,
        future_observed_mask: torch.Tensor | None = ...,
        decoder_attention_mask: torch.LongTensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        decoder_head_mask: torch.Tensor | None = ...,
        cross_attn_head_mask: torch.Tensor | None = ...,
        encoder_outputs: list[torch.FloatTensor] | None = ...,
        past_key_values: Cache | None = ...,
        output_hidden_states: bool | None = ...,
        output_attentions: bool | None = ...,
        use_cache: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
    ) -> Seq2SeqTSModelOutput | tuple: ...
    @torch.no_grad()
    def generate(
        self,
        past_values: torch.Tensor,
        past_time_features: torch.Tensor,
        future_time_features: torch.Tensor,
        past_observed_mask: torch.Tensor | None = ...,
        static_categorical_features: torch.Tensor | None = ...,
        static_real_features: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
    ) -> SampleTSPredictionOutput: ...

__all__ = [
    "TimeSeriesTransformerForPrediction",
    "TimeSeriesTransformerModel",
    "TimeSeriesTransformerPreTrainedModel",
]
