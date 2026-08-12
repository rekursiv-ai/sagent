from typing import Any
from dataclasses import dataclass

from torch import nn
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

import torch

from .configuration_patchtsmixer import PatchTSMixerConfig
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...processing_utils import Unpack
from ...utils import auto_docstring

"""PyTorch PatchTSMixer model."""
logger = ...

class PatchTSMixerGatedAttention(nn.Module):
    def __init__(self, in_size: int, out_size: int) -> None: ...
    def forward(self, inputs): ...

class PatchTSMixerBatchNorm(nn.Module):
    def __init__(self, config: PatchTSMixerConfig) -> None: ...
    def forward(self, inputs: torch.Tensor):  # -> Any:
        ...

class PatchTSMixerPositionalEncoding(nn.Module):
    def __init__(self, config: PatchTSMixerConfig) -> None: ...
    def forward(self, patch_input: torch.Tensor):  # -> Tensor:
        ...

class PatchTSMixerNormLayer(nn.Module):
    def __init__(self, config: PatchTSMixerConfig) -> None: ...
    def forward(self, inputs: torch.Tensor):  # -> Tensor:
        ...

class PatchTSMixerMLP(nn.Module):
    def __init__(self, in_features, out_features, config) -> None: ...
    def forward(self, inputs: torch.Tensor):  # -> Tensor:
        ...

class PatchTSMixerChannelFeatureMixerBlock(nn.Module):
    def __init__(self, config: PatchTSMixerConfig) -> None: ...
    def forward(self, inputs: torch.Tensor):  # -> Tensor:
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

class PatchTSMixerAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = ...,
        is_decoder: bool = ...,
        bias: bool = ...,
        is_causal: bool = ...,
        config: PatchTSMixerConfig | None = ...,
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
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class PatchMixerBlock(nn.Module):
    def __init__(self, config: PatchTSMixerConfig) -> None: ...
    def forward(self, hidden_state): ...

class FeatureMixerBlock(nn.Module):
    def __init__(self, config: PatchTSMixerConfig) -> None: ...
    def forward(self, hidden: torch.Tensor):  # -> Tensor:
        ...

class PatchTSMixerLayer(nn.Module):
    def __init__(self, config: PatchTSMixerConfig) -> None: ...
    def forward(self, hidden: torch.Tensor):  # -> Tensor:
        ...

class PatchTSMixerBlock(nn.Module):
    def __init__(self, config: PatchTSMixerConfig) -> None: ...
    def forward(
        self, hidden_state, output_hidden_states: bool = ...
    ):  # -> tuple[Any, list[Any]] | tuple[Any, None]:
        ...

class PatchTSMixerForPredictionHead(nn.Module):
    def __init__(self, config: PatchTSMixerConfig, distribution_output=...) -> None: ...
    def forward(self, hidden_features):  # -> tuple[Any, ...] | Any:
        ...

class PatchTSMixerLinearHead(nn.Module):
    def __init__(self, config: PatchTSMixerConfig, distribution_output=...) -> None: ...
    def forward(self, hidden_features):  # -> Any:
        ...

@auto_docstring
class PatchTSMixerPreTrainedModel(PreTrainedModel):
    config: PatchTSMixerConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...

class PatchTSMixerPretrainHead(nn.Module):
    def __init__(self, config: PatchTSMixerConfig) -> None: ...
    def forward(self, hidden_features):  # -> Any:
        ...

def random_masking(
    inputs: torch.Tensor,
    mask_ratio: float,
    unmasked_channel_indices: list | None = ...,
    channel_consistent_masking: bool = ...,
    mask_value: int = ...,
):  # -> tuple[Tensor, Tensor]:
    ...
def forecast_masking(
    inputs: torch.Tensor,
    num_forecast_mask_patches: list | int,
    unmasked_channel_indices: list | None = ...,
    mask_value: int = ...,
):  # -> tuple[Tensor, Tensor]:
    ...

class PatchTSMixerPatchify(nn.Module):
    def __init__(self, config: PatchTSMixerConfig) -> None: ...
    def forward(self, past_values: torch.Tensor):  # -> Tensor:
        ...

class PatchTSMixerMasking(nn.Module):
    def __init__(self, config: PatchTSMixerConfig) -> None: ...
    def forward(self, patch_input: torch.Tensor):  # -> tuple[Tensor, Tensor]:
        ...

class PatchTSMixerStdScaler(nn.Module):
    def __init__(self, config: PatchTSMixerConfig) -> None: ...
    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

class PatchTSMixerMeanScaler(nn.Module):
    def __init__(self, config: PatchTSMixerConfig) -> None: ...
    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

class PatchTSMixerNOPScaler(nn.Module):
    def __init__(self, config: PatchTSMixerConfig) -> None: ...
    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor | None = ...
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

@dataclass
@auto_docstring(custom_intro=...)
class PatchTSMixerEncoderOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...

class PatchTSMixerEncoder(PatchTSMixerPreTrainedModel):
    def __init__(self, config: PatchTSMixerConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        past_values: torch.Tensor,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | PatchTSMixerEncoderOutput: ...

@dataclass
@auto_docstring(custom_intro=...)
class PatchTSMixerModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    patch_input: torch.FloatTensor | None = ...
    mask: torch.FloatTensor | None = ...
    loc: torch.FloatTensor | None = ...
    scale: torch.FloatTensor | None = ...

@auto_docstring(custom_intro=...)
class PatchTSMixerModel(PatchTSMixerPreTrainedModel):
    def __init__(self, config: PatchTSMixerConfig, mask_input: bool = ...) -> None: ...
    @auto_docstring
    def forward(
        self,
        past_values: torch.Tensor,
        observed_mask: torch.Tensor | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> PatchTSMixerModelOutput: ...

@dataclass
@auto_docstring(custom_intro=...)
class PatchTSMixerForPreTrainingOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    prediction_outputs: torch.FloatTensor | None = ...
    last_hidden_state: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...

@auto_docstring(
    custom_intro="""
    `PatchTSMixer` for mask pretraining.
    """
)
class PatchTSMixerForPretraining(PatchTSMixerPreTrainedModel):
    def __init__(self, config: PatchTSMixerConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        past_values: torch.Tensor,
        observed_mask: torch.Tensor | None = ...,
        output_hidden_states: bool | None = ...,
        return_loss: bool = ...,
        return_dict: bool | None = ...,
    ) -> PatchTSMixerForPreTrainingOutput: ...

@dataclass
@auto_docstring(custom_intro=...)
class PatchTSMixerForPredictionOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    prediction_outputs: torch.FloatTensor | None = ...
    last_hidden_state: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    loc: torch.FloatTensor | None = ...
    scale: torch.FloatTensor | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class SamplePatchTSMixerPredictionOutput(ModelOutput):
    sequences: torch.FloatTensor | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class SamplePatchTSMixerRegressionOutput(ModelOutput):
    sequences: torch.FloatTensor | None = ...

def nll(
    input: torch.distributions.Distribution, target: torch.Tensor
) -> torch.Tensor: ...
def weighted_average(
    input_tensor: torch.Tensor, weights: torch.Tensor | None = ..., dim=...
) -> torch.Tensor: ...

class PatchTSMixerForPrediction(PatchTSMixerPreTrainedModel):
    def __init__(self, config: PatchTSMixerConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        past_values: torch.Tensor,
        observed_mask: torch.Tensor | None = ...,
        future_values: torch.Tensor | None = ...,
        output_hidden_states: bool | None = ...,
        return_loss: bool = ...,
        return_dict: bool | None = ...,
    ) -> PatchTSMixerForPredictionOutput: ...
    @torch.no_grad()
    def generate(
        self, past_values: torch.Tensor, observed_mask: torch.Tensor | None = ...
    ) -> SamplePatchTSMixerPredictionOutput: ...

@dataclass
@auto_docstring(custom_intro=...)
class PatchTSMixerForTimeSeriesClassificationOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    prediction_outputs: torch.FloatTensor | None = ...
    last_hidden_state: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...

class PatchTSMixerForTimeSeriesClassification(PatchTSMixerPreTrainedModel):
    def __init__(self, config: PatchTSMixerConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        past_values: torch.Tensor,
        target_values: torch.Tensor | None = ...,
        output_hidden_states: bool | None = ...,
        return_loss: bool = ...,
        return_dict: bool | None = ...,
    ) -> PatchTSMixerForTimeSeriesClassificationOutput: ...

@dataclass
@auto_docstring(custom_intro=...)
class PatchTSMixerForRegressionOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    regression_outputs: torch.FloatTensor | None = ...
    last_hidden_state: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...

class InjectScalerStatistics4D(nn.Module):
    def __init__(
        self, d_model: int, num_patches: int, expansion: int = ...
    ) -> None: ...
    def forward(
        self, inputs: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor
    ):  # -> Tensor:
        ...

@auto_docstring(custom_intro=...)
class PatchTSMixerForRegression(PatchTSMixerPreTrainedModel):
    def __init__(self, config: PatchTSMixerConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        past_values: torch.Tensor,
        target_values: torch.Tensor | None = ...,
        output_hidden_states: bool | None = ...,
        return_loss: bool = ...,
        return_dict: bool | None = ...,
    ) -> PatchTSMixerForRegressionOutput: ...
    @torch.no_grad()
    def generate(
        self, past_values: torch.Tensor
    ) -> SamplePatchTSMixerRegressionOutput: ...

__all__ = [
    "PatchTSMixerForPrediction",
    "PatchTSMixerForPretraining",
    "PatchTSMixerForRegression",
    "PatchTSMixerForTimeSeriesClassification",
    "PatchTSMixerModel",
    "PatchTSMixerPreTrainedModel",
]
