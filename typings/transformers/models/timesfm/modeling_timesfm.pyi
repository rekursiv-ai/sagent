from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from torch import nn

import torch

from .configuration_timesfm import TimesFmConfig
from ...integrations import use_kernel_forward_from_hub
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple

logger = ...

@dataclass
@auto_docstring
class TimesFmOutput(BaseModelOutput):
    loc: torch.Tensor | None = ...
    scale: torch.Tensor | None = ...

@dataclass
@auto_docstring
class TimesFmOutputForPrediction(BaseModelOutput):
    mean_predictions: torch.Tensor | None = ...
    full_predictions: torch.Tensor | None = ...
    loss: torch.Tensor | float | None = ...

class TimesFmMLP(nn.Module):
    def __init__(self, config: TimesFmConfig) -> None: ...
    def forward(self, x, paddings=...): ...

class TimesFmResidualBlock(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims) -> None: ...
    def forward(self, x):  # -> Any:
        ...

@use_kernel_forward_from_hub("RMSNorm")
class TimesFmRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=...) -> None: ...
    def forward(self, hidden_states): ...
    def extra_repr(self):  # -> str:
        ...

class TimesFmPositionalEmbedding(nn.Module):
    def __init__(self, config: TimesFmConfig) -> None: ...
    def forward(self, seq_length=..., position=...):  # -> Tensor:
        ...

def simple_eager_attention_forward(
    module: nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = ...,
    **kwargs: Unpack[TransformersKwargs],
):  # -> tuple[Tensor, Tensor]:
    ...

class TimesFmAttention(nn.Module):
    def __init__(self, config: TimesFmConfig, layer_idx: int) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class TimesFmDecoderLayer(nn.Module):
    def __init__(self, config: TimesFmConfig, layer_idx: int) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        paddings: torch.Tensor,
        output_attentions: bool = ...,
    ) -> tuple[torch.Tensor | None, torch.Tensor]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor | None, torch.Tensor]: ...

@auto_docstring
class TimesFmPreTrainedModel(PreTrainedModel):
    config: TimesFmConfig
    base_model_prefix = ...
    _no_split_modules = ...
    main_input_name = ...
    _supports_sdpa = ...

@auto_docstring
class TimesFmModel(TimesFmPreTrainedModel):
    def __init__(self, config: TimesFmConfig) -> None: ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        past_values: torch.Tensor,
        past_values_padding: torch.LongTensor,
        freq: torch.Tensor,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
    ) -> TimesFmOutput: ...

class TimesFmModelForPrediction(TimesFmPreTrainedModel):
    def __init__(self, config: TimesFmConfig) -> None: ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        past_values: Sequence[torch.Tensor],
        freq: Sequence[torch.Tensor | int] | None = ...,
        window_size: int | None = ...,
        future_values: torch.Tensor | None = ...,
        forecast_context_len: int | None = ...,
        return_forecast_on_context: bool = ...,
        truncate_negative: bool = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
    ) -> TimesFmOutputForPrediction: ...

__all__ = ["TimesFmModel", "TimesFmModelForPrediction", "TimesFmPreTrainedModel"]
