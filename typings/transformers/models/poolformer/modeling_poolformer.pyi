from typing import Any
from torch import nn

import torch

from .configuration_poolformer import PoolFormerConfig
from ...modeling_outputs import (
    BaseModelOutputWithNoAttention,
    ImageClassifierOutputWithNoAttention,
)
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring

"""PyTorch PoolFormer model."""
logger = ...

def drop_path(
    input: torch.Tensor, drop_prob: float = ..., training: bool = ...
) -> torch.Tensor: ...

class PoolFormerDropPath(nn.Module):
    def __init__(self, drop_prob: float | None = ...) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...
    def extra_repr(self) -> str: ...

class PoolFormerEmbeddings(nn.Module):
    def __init__(
        self, hidden_size, num_channels, patch_size, stride, padding, norm_layer=...
    ) -> None: ...
    def forward(self, pixel_values):  # -> Any:
        ...

class PoolFormerGroupNorm(nn.GroupNorm):
    def __init__(self, num_channels, **kwargs) -> None: ...

class PoolFormerPooling(nn.Module):
    def __init__(self, pool_size) -> None: ...
    def forward(self, hidden_states): ...

class PoolFormerOutput(nn.Module):
    def __init__(
        self, config, dropout_prob, hidden_size, intermediate_size
    ) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class PoolFormerLayer(nn.Module):
    def __init__(
        self, config, num_channels, pool_size, hidden_size, intermediate_size, drop_path
    ) -> None: ...
    def forward(self, hidden_states):  # -> tuple[Any]:
        ...

class PoolFormerEncoder(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, pixel_values, output_hidden_states=..., return_dict=...
    ):  # -> tuple[Any | tuple[()] | tuple[Any, ...], ...] | BaseModelOutputWithNoAttention:
        ...

@auto_docstring
class PoolFormerPreTrainedModel(PreTrainedModel):
    config: PoolFormerConfig
    base_model_prefix = ...
    main_input_name = ...
    _no_split_modules = ...

@auto_docstring
class PoolFormerModel(PoolFormerPreTrainedModel):
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> Tensor | Module:
        ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutputWithNoAttention: ...

class PoolFormerFinalPooler(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

@auto_docstring(custom_intro=...)
class PoolFormerForImageClassification(PoolFormerPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | ImageClassifierOutputWithNoAttention: ...

__all__ = [
    "PoolFormerForImageClassification",
    "PoolFormerModel",
    "PoolFormerPreTrainedModel",
]
