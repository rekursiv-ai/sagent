from dataclasses import dataclass

from torch import nn

import torch

from .configuration_levit import LevitConfig
from ...modeling_outputs import (
    BaseModelOutputWithPoolingAndNoAttention,
    ImageClassifierOutputWithNoAttention,
    ModelOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring

"""PyTorch LeViT model."""
logger = ...

@dataclass
@auto_docstring(custom_intro=...)
class LevitForImageClassificationWithTeacherOutput(ModelOutput):
    logits: torch.FloatTensor | None = ...
    cls_logits: torch.FloatTensor | None = ...
    distillation_logits: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...

class LevitConvEmbeddings(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation=...,
        groups=...,
        bn_weight_init=...,
    ) -> None: ...
    def forward(self, embeddings):  # -> Any:
        ...

class LevitPatchEmbeddings(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, pixel_values):  # -> Any:
        ...

class MLPLayerWithBN(nn.Module):
    def __init__(self, input_dim, output_dim, bn_weight_init=...) -> None: ...
    def forward(self, hidden_state):  # -> Any:
        ...

class LevitSubsample(nn.Module):
    def __init__(self, stride, resolution) -> None: ...
    def forward(self, hidden_state): ...

class LevitAttention(nn.Module):
    def __init__(
        self, hidden_sizes, key_dim, num_attention_heads, attention_ratio, resolution
    ) -> None: ...
    @torch.no_grad()
    def train(self, mode=...):  # -> None:
        ...
    def get_attention_biases(self, device):  # -> Tensor:
        ...
    def forward(self, hidden_state):  # -> Any:
        ...

class LevitAttentionSubsample(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        key_dim,
        num_attention_heads,
        attention_ratio,
        stride,
        resolution_in,
        resolution_out,
    ) -> None: ...
    @torch.no_grad()
    def train(self, mode=...):  # -> None:
        ...
    def get_attention_biases(self, device):  # -> Tensor:
        ...
    def forward(self, hidden_state):  # -> Any:
        ...

class LevitMLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim) -> None: ...
    def forward(self, hidden_state):  # -> Any:
        ...

class LevitResidualLayer(nn.Module):
    def __init__(self, module, drop_rate) -> None: ...
    def forward(self, hidden_state): ...

class LevitStage(nn.Module):
    def __init__(
        self,
        config,
        idx,
        hidden_sizes,
        key_dim,
        depths,
        num_attention_heads,
        attention_ratio,
        mlp_ratio,
        down_ops,
        resolution_in,
    ) -> None: ...
    def get_resolution(self):  # -> Any:
        ...
    def forward(self, hidden_state):  # -> Any:
        ...

class LevitEncoder(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_state, output_hidden_states=..., return_dict=...
    ):  # -> tuple[Any | tuple[Any, ...] | tuple[()], ...] | BaseModelOutputWithNoAttention:
        ...

class LevitClassificationLayer(nn.Module):
    def __init__(self, input_dim, output_dim) -> None: ...
    def forward(self, hidden_state):  # -> Any:
        ...

@auto_docstring
class LevitPreTrainedModel(PreTrainedModel):
    config: LevitConfig
    base_model_prefix = ...
    main_input_name = ...
    _no_split_modules = ...

@auto_docstring
class LevitModel(LevitPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutputWithPoolingAndNoAttention: ...

@auto_docstring(custom_intro=...)
class LevitForImageClassification(LevitPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | ImageClassifierOutputWithNoAttention: ...

@auto_docstring(custom_intro=...)
class LevitForImageClassificationWithTeacher(LevitPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | LevitForImageClassificationWithTeacherOutput: ...

__all__ = [
    "LevitForImageClassification",
    "LevitForImageClassificationWithTeacher",
    "LevitModel",
    "LevitPreTrainedModel",
]
