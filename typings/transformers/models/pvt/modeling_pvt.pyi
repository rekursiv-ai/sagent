from collections.abc import Iterable
from typing import Any

from torch import nn

import torch

from .configuration_pvt import PvtConfig
from ...modeling_outputs import BaseModelOutput, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring

"""PyTorch PVT model."""
logger = ...

def drop_path(
    input: torch.Tensor, drop_prob: float = ..., training: bool = ...
) -> torch.Tensor: ...

class PvtDropPath(nn.Module):
    def __init__(self, drop_prob: float | None = ...) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...
    def extra_repr(self) -> str: ...

class PvtPatchEmbeddings(nn.Module):
    def __init__(
        self,
        config: PvtConfig,
        image_size: int | Iterable[int],
        patch_size: int | Iterable[int],
        stride: int,
        num_channels: int,
        hidden_size: int,
        cls_token: bool = ...,
    ) -> None: ...
    def interpolate_pos_encoding(
        self, embeddings: torch.Tensor, height: int, width: int
    ) -> torch.Tensor: ...
    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, int, int]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, int, int]: ...

class PvtSelfOutput(nn.Module):
    def __init__(self, config: PvtConfig, hidden_size: int) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class PvtEfficientSelfAttention(nn.Module):
    def __init__(
        self,
        config: PvtConfig,
        hidden_size: int,
        num_attention_heads: int,
        sequences_reduction_ratio: float,
    ) -> None: ...
    def transpose_for_scores(self, hidden_states: int) -> torch.Tensor: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        height: int,
        width: int,
        output_attentions: bool = ...,
    ) -> tuple[torch.Tensor]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor]: ...

class PvtAttention(nn.Module):
    def __init__(
        self,
        config: PvtConfig,
        hidden_size: int,
        num_attention_heads: int,
        sequences_reduction_ratio: float,
    ) -> None: ...
    def prune_heads(self, heads):  # -> None:
        ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        height: int,
        width: int,
        output_attentions: bool = ...,
    ) -> tuple[torch.Tensor]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor]: ...

class PvtFFN(nn.Module):
    def __init__(
        self,
        config: PvtConfig,
        in_features: int,
        hidden_features: int | None = ...,
        out_features: int | None = ...,
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class PvtLayer(nn.Module):
    def __init__(
        self,
        config: PvtConfig,
        hidden_size: int,
        num_attention_heads: int,
        drop_path: float,
        sequences_reduction_ratio: float,
        mlp_ratio: float,
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        height: int,
        width: int,
        output_attentions: bool = ...,
    ):  # -> Any:
        ...

class PvtEncoder(nn.Module):
    def __init__(self, config: PvtConfig) -> None: ...
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutput: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple | BaseModelOutput: ...

@auto_docstring
class PvtPreTrainedModel(PreTrainedModel):
    config: PvtConfig
    base_model_prefix = ...
    main_input_name = ...
    _no_split_modules = ...

@auto_docstring
class PvtModel(PvtPreTrainedModel):
    def __init__(self, config: PvtConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutput: ...

@auto_docstring(custom_intro=...)
class PvtForImageClassification(PvtPreTrainedModel):
    def __init__(self, config: PvtConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor | None,
        labels: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | ImageClassifierOutput: ...

__all__ = ["PvtForImageClassification", "PvtModel", "PvtPreTrainedModel"]
