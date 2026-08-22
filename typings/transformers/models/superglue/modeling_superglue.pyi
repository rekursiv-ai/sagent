from dataclasses import dataclass
from typing import Any

from torch import nn
from transformers import PreTrainedModel
from transformers.models.superglue.configuration_superglue import SuperGlueConfig

import torch

from ...utils import ModelOutput, auto_docstring

"""PyTorch SuperGlue model."""
logger = ...

def concat_pairs(
    tensor_tuple0: tuple[torch.Tensor], tensor_tuple1: tuple[torch.Tensor]
) -> tuple[torch.Tensor]: ...
def normalize_keypoints(
    keypoints: torch.Tensor, height: int, width: int
) -> torch.Tensor: ...
def log_sinkhorn_iterations(
    log_cost_matrix: torch.Tensor,
    log_source_distribution: torch.Tensor,
    log_target_distribution: torch.Tensor,
    num_iterations: int,
) -> torch.Tensor: ...
def log_optimal_transport(
    scores: torch.Tensor, reg_param: torch.Tensor, iterations: int
) -> torch.Tensor: ...
def arange_like(x, dim: int) -> torch.Tensor: ...

@dataclass
@auto_docstring(custom_intro=...)
class KeypointMatchingOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    matches: torch.FloatTensor | None = ...
    matching_scores: torch.FloatTensor | None = ...
    keypoints: torch.FloatTensor | None = ...
    mask: torch.IntTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...

class SuperGlueMultiLayerPerceptron(nn.Module):
    def __init__(
        self, config: SuperGlueConfig, in_channels: int, out_channels: int
    ) -> None: ...
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class SuperGlueKeypointEncoder(nn.Module):
    def __init__(self, config: SuperGlueConfig) -> None: ...
    def forward(
        self,
        keypoints: torch.Tensor,
        scores: torch.Tensor,
        output_hidden_states: bool | None = ...,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor] | None]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, tuple[torch.Tensor] | None]: ...

class SuperGlueSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=...) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        encoder_hidden_states: torch.FloatTensor | None = ...,
        encoder_attention_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.Tensor]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor]: ...

class SuperGlueSelfOutput(nn.Module):
    def __init__(self, config: SuperGlueConfig) -> None: ...
    def forward(self, hidden_states: torch.Tensor, *args) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

SUPERGLUE_SELF_ATTENTION_CLASSES = ...

class SuperGlueAttention(nn.Module):
    def __init__(self, config, position_embedding_type=...) -> None: ...
    def prune_heads(self, heads):  # -> None:
        ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        encoder_hidden_states: torch.FloatTensor | None = ...,
        encoder_attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.Tensor]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor]: ...

class SuperGlueAttentionalPropagation(nn.Module):
    def __init__(self, config: SuperGlueConfig) -> None: ...
    def forward(
        self,
        descriptors: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        encoder_hidden_states: torch.Tensor | None = ...,
        encoder_attention_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
    ) -> tuple[
        torch.Tensor, tuple[torch.Tensor] | None, tuple[torch.Tensor] | None
    ]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[
        torch.Tensor, tuple[torch.Tensor] | None, tuple[torch.Tensor] | None
    ]: ...

class SuperGlueAttentionalGNN(nn.Module):
    def __init__(self, config: SuperGlueConfig) -> None: ...
    def forward(
        self,
        descriptors: torch.Tensor,
        mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool | None = ...,
    ) -> tuple[torch.Tensor, tuple | None, tuple | None]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, tuple | None, tuple | None]: ...

class SuperGlueFinalProjection(nn.Module):
    def __init__(self, config: SuperGlueConfig) -> None: ...
    def forward(self, descriptors: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

@auto_docstring
class SuperGluePreTrainedModel(PreTrainedModel):
    config: SuperGlueConfig
    base_model_prefix = ...
    main_input_name = ...

@auto_docstring(custom_intro=...)
class SuperGlueForKeypointMatching(SuperGluePreTrainedModel):
    def __init__(self, config: SuperGlueConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | KeypointMatchingOutput: ...

__all__ = ["SuperGlueForKeypointMatching", "SuperGluePreTrainedModel"]
