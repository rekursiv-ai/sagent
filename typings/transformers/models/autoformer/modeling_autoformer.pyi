from typing import Any
from dataclasses import dataclass

from torch import nn

import torch

from .configuration_autoformer import AutoformerConfig
from ...cache_utils import Cache
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    ModelOutput,
    SampleTSPredictionOutput,
    Seq2SeqTSPredictionOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring
from ...utils.deprecation import deprecate_kwarg

"""PyTorch Autoformer model."""
logger = ...

@dataclass
@auto_docstring(custom_intro=...)
class AutoFormerDecoderOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = ...
    trend: torch.FloatTensor | None = ...
    past_key_values: Cache | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...
    cross_attentions: tuple[torch.FloatTensor] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class AutoformerModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = ...
    trend: torch.FloatTensor | None = ...
    past_key_values: Cache | None = ...
    decoder_hidden_states: tuple[torch.FloatTensor] | None = ...
    decoder_attentions: tuple[torch.FloatTensor] | None = ...
    cross_attentions: tuple[torch.FloatTensor] | None = ...
    encoder_last_hidden_state: torch.FloatTensor | None = ...
    encoder_hidden_states: tuple[torch.FloatTensor] | None = ...
    encoder_attentions: tuple[torch.FloatTensor] | None = ...
    loc: torch.FloatTensor | None = ...
    scale: torch.FloatTensor | None = ...
    static_features: torch.FloatTensor | None = ...

class AutoformerFeatureEmbedder(nn.Module):
    def __init__(self, cardinalities: list[int], embedding_dims: list[int]) -> None: ...
    def forward(self, features: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class AutoformerStdScaler(nn.Module):
    def __init__(self, config: AutoformerConfig) -> None: ...
    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

class AutoformerMeanScaler(nn.Module):
    def __init__(self, config: AutoformerConfig) -> None: ...
    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

class AutoformerNOPScaler(nn.Module):
    def __init__(self, config: AutoformerConfig) -> None: ...
    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor | None = ...
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

def weighted_average(
    input_tensor: torch.Tensor, weights: torch.Tensor | None = ..., dim=...
) -> torch.Tensor: ...
def nll(
    input: torch.distributions.Distribution, target: torch.Tensor
) -> torch.Tensor: ...

class AutoformerSinusoidalPositionalEmbedding(nn.Embedding):
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

class AutoformerValueEmbedding(nn.Module):
    def __init__(self, feature_size, d_model) -> None: ...
    def forward(self, x):  # -> Any:
        ...

class AutoformerSeriesDecompositionLayer(nn.Module):
    def __init__(self, config: AutoformerConfig) -> None: ...
    def forward(self, x):  # -> tuple[Any, Any]:
        ...

class AutoformerLayernorm(nn.Module):
    def __init__(self, config: AutoformerConfig) -> None: ...
    def forward(self, x):  # -> Any:
        ...

class AutoformerAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float | None = ...,
        is_decoder: bool | None = ...,
        bias: bool | None = ...,
        autocorrelation_factor: int | None = ...,
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
        output_attentions: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class AutoformerEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: AutoformerConfig) -> None: ...
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

class AutoformerDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: AutoformerConfig, layer_idx=...) -> None: ...
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
class AutoformerPreTrainedModel(PreTrainedModel):
    config: AutoformerConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...

class AutoformerEncoder(AutoformerPreTrainedModel):
    def __init__(self, config: AutoformerConfig) -> None: ...
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

class AutoformerDecoder(AutoformerPreTrainedModel):
    def __init__(self, config: AutoformerConfig) -> None: ...
    def forward(
        self,
        trend: torch.Tensor | None = ...,
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
        cache_position: torch.Tensor | None = ...,
    ) -> tuple | AutoFormerDecoderOutput: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple | AutoFormerDecoderOutput: ...

@auto_docstring
class AutoformerModel(AutoformerPreTrainedModel):
    def __init__(self, config: AutoformerConfig) -> None: ...
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
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]: ...
    def get_encoder(self):  # -> AutoformerEncoder:
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
        cache_position: torch.Tensor | None = ...,
    ) -> AutoformerModelOutput | tuple: ...

@auto_docstring
class AutoformerForPrediction(AutoformerPreTrainedModel):
    def __init__(self, config: AutoformerConfig) -> None: ...
    def output_params(self, decoder_output):  # -> Any:
        ...
    def get_encoder(self):  # -> AutoformerEncoder:
        ...
    def get_decoder(self):  # -> PreTrainedModel | AutoformerModel:
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
    ) -> Seq2SeqTSPredictionOutput | tuple: ...
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

__all__ = ["AutoformerForPrediction", "AutoformerModel", "AutoformerPreTrainedModel"]
