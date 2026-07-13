from dataclasses import dataclass

from torch import nn

import torch

from .configuration_cvt import CvtConfig
from ...modeling_outputs import ImageClassifierOutputWithNoAttention, ModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring

"""PyTorch CvT model."""
logger = ...

@dataclass
@auto_docstring(custom_intro=...)
class BaseModelOutputWithCLSToken(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = ...
    cls_token_value: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...

def drop_path(
    input: torch.Tensor, drop_prob: float = ..., training: bool = ...
) -> torch.Tensor: ...

class CvtDropPath(nn.Module):
    def __init__(self, drop_prob: float | None = ...) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def extra_repr(self) -> str: ...

class CvtEmbeddings(nn.Module):
    def __init__(
        self, patch_size, num_channels, embed_dim, stride, padding, dropout_rate
    ) -> None: ...
    def forward(self, pixel_values):  # -> Any:
        ...

class CvtConvEmbeddings(nn.Module):
    def __init__(
        self, patch_size, num_channels, embed_dim, stride, padding
    ) -> None: ...
    def forward(self, pixel_values):  # -> Any:
        ...

class CvtSelfAttentionConvProjection(nn.Module):
    def __init__(self, embed_dim, kernel_size, padding, stride) -> None: ...
    def forward(self, hidden_state):  # -> Any:
        ...

class CvtSelfAttentionLinearProjection(nn.Module):
    def forward(self, hidden_state): ...

class CvtSelfAttentionProjection(nn.Module):
    def __init__(
        self, embed_dim, kernel_size, padding, stride, projection_method=...
    ) -> None: ...
    def forward(self, hidden_state):  # -> Any:
        ...

class CvtSelfAttention(nn.Module):
    def __init__(
        self,
        num_heads,
        embed_dim,
        kernel_size,
        padding_q,
        padding_kv,
        stride_q,
        stride_kv,
        qkv_projection_method,
        qkv_bias,
        attention_drop_rate,
        with_cls_token=...,
        **kwargs,
    ) -> None: ...
    def rearrange_for_multi_head_attention(self, hidden_state): ...
    def forward(self, hidden_state, height, width):  # -> Tensor:
        ...

class CvtSelfOutput(nn.Module):
    def __init__(self, embed_dim, drop_rate) -> None: ...
    def forward(self, hidden_state, input_tensor):  # -> Any:
        ...

class CvtAttention(nn.Module):
    def __init__(
        self,
        num_heads,
        embed_dim,
        kernel_size,
        padding_q,
        padding_kv,
        stride_q,
        stride_kv,
        qkv_projection_method,
        qkv_bias,
        attention_drop_rate,
        drop_rate,
        with_cls_token=...,
    ) -> None: ...
    def prune_heads(self, heads):  # -> None:
        ...
    def forward(self, hidden_state, height, width):  # -> Any:
        ...

class CvtIntermediate(nn.Module):
    def __init__(self, embed_dim, mlp_ratio) -> None: ...
    def forward(self, hidden_state):  # -> Any:
        ...

class CvtOutput(nn.Module):
    def __init__(self, embed_dim, mlp_ratio, drop_rate) -> None: ...
    def forward(self, hidden_state, input_tensor): ...

class CvtLayer(nn.Module):
    def __init__(
        self,
        num_heads,
        embed_dim,
        kernel_size,
        padding_q,
        padding_kv,
        stride_q,
        stride_kv,
        qkv_projection_method,
        qkv_bias,
        attention_drop_rate,
        drop_rate,
        mlp_ratio,
        drop_path_rate,
        with_cls_token=...,
    ) -> None: ...
    def forward(self, hidden_state, height, width):  # -> Any:
        ...

class CvtStage(nn.Module):
    def __init__(self, config, stage) -> None: ...
    def forward(self, hidden_state):  # -> tuple[Tensor | Any, Tensor | None]:
        ...

class CvtEncoder(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, pixel_values, output_hidden_states=..., return_dict=...
    ):  # -> tuple[Any | tuple[()] | tuple[Any, ...], ...] | BaseModelOutputWithCLSToken:
        ...

@auto_docstring
class CvtPreTrainedModel(PreTrainedModel):
    config: CvtConfig
    base_model_prefix = ...
    main_input_name = ...
    _no_split_modules = ...

@auto_docstring
class CvtModel(CvtPreTrainedModel):
    def __init__(self, config, add_pooling_layer=...) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutputWithCLSToken: ...

@auto_docstring(custom_intro=...)
class CvtForImageClassification(CvtPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | ImageClassifierOutputWithNoAttention: ...

__all__ = ["CvtForImageClassification", "CvtModel", "CvtPreTrainedModel"]
