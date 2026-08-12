from typing import Any
from dataclasses import dataclass

from torch import nn

import torch

from .configuration_patchtst import PatchTSTConfig
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import ModelOutput, auto_docstring

"""PyTorch PatchTST model."""
logger = ...

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

class PatchTSTAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = ...,
        is_decoder: bool = ...,
        bias: bool = ...,
        is_causal: bool = ...,
        config: PatchTSTConfig | None = ...,
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

class PatchTSTBatchNorm(nn.Module):
    def __init__(self, config: PatchTSTConfig) -> None: ...
    def forward(self, inputs: torch.Tensor):  # -> Any:
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

class PatchTSTPatchify(nn.Module):
    def __init__(self, config: PatchTSTConfig) -> None: ...
    def forward(self, past_values: torch.Tensor):  # -> Tensor:
        ...

class PatchTSTMasking(nn.Module):
    def __init__(self, config: PatchTSTConfig) -> None: ...
    def forward(self, patch_input: torch.Tensor):  # -> tuple[Tensor, Tensor]:
        ...

class PatchTSTEncoderLayer(nn.Module):
    def __init__(self, config: PatchTSTConfig) -> None: ...
    def forward(
        self, hidden_state: torch.Tensor, output_attentions: bool | None = ...
    ):  # -> tuple[Tensor, Any, Any] | tuple[Tensor, Any] | tuple[Tensor]:
        ...

@auto_docstring
class PatchTSTPreTrainedModel(PreTrainedModel):
    config: PatchTSTConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...

class PatchTSTEmbedding(nn.Module):
    def __init__(self, config: PatchTSTConfig) -> None: ...
    def forward(self, patch_input: torch.Tensor):  # -> Tensor:
        ...

class PatchTSTPositionalEncoding(nn.Module):
    def __init__(self, config: PatchTSTConfig, num_patches: int) -> None: ...
    def forward(self, patch_input: torch.Tensor):  # -> Tensor | Any:
        ...

class PatchTSTEncoder(PatchTSTPreTrainedModel):
    def __init__(self, config: PatchTSTConfig, num_patches: int) -> None: ...
    def forward(
        self,
        patch_input: torch.Tensor,
        output_hidden_states: bool | None = ...,
        output_attentions: bool | None = ...,
    ) -> BaseModelOutput: ...
    def __call__(self, *args: Any, **kwargs: Any) -> BaseModelOutput: ...

@dataclass
@auto_docstring(custom_intro=...)
class PatchTSTModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...
    mask: torch.FloatTensor | None = ...
    loc: torch.FloatTensor | None = ...
    scale: torch.FloatTensor | None = ...
    patch_input: torch.FloatTensor | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class PatchTSTForPretrainingOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    prediction_output: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class PatchTSTForRegressionOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    regression_outputs: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class PatchTSTForPredictionOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    prediction_outputs: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...
    loc: torch.FloatTensor | None = ...
    scale: torch.FloatTensor | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class PatchTSTForClassificationOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    prediction_logits: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class SamplePatchTSTOutput(ModelOutput):
    sequences: torch.FloatTensor | None = ...

def nll(
    input: torch.distributions.Distribution, target: torch.Tensor
) -> torch.Tensor: ...
def weighted_average(
    input_tensor: torch.Tensor, weights: torch.Tensor | None = ..., dim=...
) -> torch.Tensor: ...

class PatchTSTStdScaler(nn.Module):
    def __init__(self, config: PatchTSTConfig) -> None: ...
    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

class PatchTSTMeanScaler(nn.Module):
    def __init__(self, config: PatchTSTConfig) -> None: ...
    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

class PatchTSTNOPScaler(nn.Module):
    def __init__(self, config: PatchTSTConfig) -> None: ...
    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor | None = ...
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

class PatchTSTScaler(nn.Module):
    def __init__(self, config: PatchTSTConfig) -> None: ...
    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

@auto_docstring
class PatchTSTModel(PatchTSTPreTrainedModel):
    def __init__(self, config: PatchTSTConfig) -> None: ...
    def forward(
        self,
        past_values: torch.Tensor,
        past_observed_mask: torch.Tensor | None = ...,
        future_values: torch.Tensor | None = ...,
        output_hidden_states: bool | None = ...,
        output_attentions: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | PatchTSTModelOutput: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple | PatchTSTModelOutput: ...

class PatchTSTMaskPretrainHead(nn.Module):
    def __init__(self, config: PatchTSTConfig) -> None: ...
    def forward(self, embedding: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

@auto_docstring(
    custom_intro="""
    The PatchTST for pretrain model.
    """
)
class PatchTSTForPretraining(PatchTSTPreTrainedModel):
    def __init__(self, config: PatchTSTConfig) -> None: ...
    def forward(
        self,
        past_values: torch.Tensor,
        past_observed_mask: torch.Tensor | None = ...,
        output_hidden_states: bool | None = ...,
        output_attentions: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | PatchTSTForPretrainingOutput: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple | PatchTSTForPretrainingOutput: ...

class PatchTSTClassificationHead(nn.Module):
    def __init__(self, config: PatchTSTConfig) -> None: ...
    def forward(self, embedding: torch.Tensor):  # -> Any:
        ...

@auto_docstring(
    custom_intro="""
    The PatchTST for classification model.
    """
)
class PatchTSTForClassification(PatchTSTPreTrainedModel):
    def __init__(self, config: PatchTSTConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        past_values: torch.Tensor,
        target_values: torch.Tensor | None = ...,
        past_observed_mask: bool | None = ...,
        output_hidden_states: bool | None = ...,
        output_attentions: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | PatchTSTForClassificationOutput: ...

@auto_docstring(
    custom_intro="""
    The PatchTST for regression Model.
    """
)
class PatchTSTPredictionHead(nn.Module):
    def __init__(
        self, config: PatchTSTConfig, num_patches: int, distribution_output=...
    ) -> None: ...
    def forward(self, embedding: torch.Tensor):  # -> tuple[Any, ...] | Tensor | Any:
        ...

@auto_docstring(
    custom_intro="""
    The PatchTST for prediction model.
    """
)
class PatchTSTForPrediction(PatchTSTPreTrainedModel):
    def __init__(self, config: PatchTSTConfig) -> None: ...
    def forward(
        self,
        past_values: torch.Tensor,
        past_observed_mask: torch.Tensor | None = ...,
        future_values: torch.Tensor | None = ...,
        output_hidden_states: bool | None = ...,
        output_attentions: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | PatchTSTForPredictionOutput: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple | PatchTSTForPredictionOutput: ...
    @torch.no_grad()
    def generate(
        self,
        past_values: torch.Tensor,
        past_observed_mask: torch.Tensor | None = ...,
    ) -> SamplePatchTSTOutput: ...

class PatchTSTRegressionHead(nn.Module):
    def __init__(self, config: PatchTSTConfig, distribution_output=...) -> None: ...
    def forward(self, embedding: torch.Tensor):  # -> Any:
        ...

@auto_docstring(
    custom_intro="""
    The PatchTST for regression model.
    """
)
class PatchTSTForRegression(PatchTSTPreTrainedModel):
    def __init__(self, config: PatchTSTConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        past_values: torch.Tensor,
        target_values: torch.Tensor | None = ...,
        past_observed_mask: torch.Tensor | None = ...,
        output_hidden_states: bool | None = ...,
        output_attentions: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | PatchTSTForRegressionOutput: ...
    @torch.no_grad()
    def generate(
        self,
        past_values: torch.Tensor,
        past_observed_mask: torch.Tensor | None = ...,
    ) -> SamplePatchTSTOutput: ...

__all__ = [
    "PatchTSTForClassification",
    "PatchTSTForPrediction",
    "PatchTSTForPretraining",
    "PatchTSTForRegression",
    "PatchTSTModel",
    "PatchTSTPreTrainedModel",
]
