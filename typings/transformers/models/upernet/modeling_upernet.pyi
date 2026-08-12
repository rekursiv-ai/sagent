from typing import Any
from torch import nn

import torch

from .configuration_upernet import UperNetConfig
from ...modeling_outputs import SemanticSegmenterOutput
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring

"""PyTorch UperNet model. Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation."""

class UperNetConvModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        padding: int | tuple[int, int] | str = ...,
        bias: bool = ...,
        dilation: int | tuple[int, int] = ...,
    ) -> None: ...
    def forward(self, input: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class UperNetPyramidPoolingBlock(nn.Module):
    def __init__(self, pool_scale: int, in_channels: int, channels: int) -> None: ...
    def forward(self, input: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class UperNetPyramidPoolingModule(nn.Module):
    def __init__(
        self,
        pool_scales: tuple[int, ...],
        in_channels: int,
        channels: int,
        align_corners: bool,
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> list[torch.Tensor]: ...

class UperNetHead(nn.Module):
    def __init__(self, config, in_channels) -> None: ...
    def psp_forward(self, inputs):  # -> Any:
        ...
    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class UperNetFCNHead(nn.Module):
    def __init__(
        self,
        config,
        in_channels,
        in_index: int = ...,
        kernel_size: int = ...,
        dilation: int | tuple[int, int] = ...,
    ) -> None: ...
    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

@auto_docstring
class UperNetPreTrainedModel(PreTrainedModel):
    config: UperNetConfig
    main_input_name = ...
    _no_split_modules = ...

@auto_docstring(custom_intro=...)
class UperNetForSemanticSegmentation(UperNetPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        labels: torch.Tensor | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | SemanticSegmenterOutput: ...

__all__ = ["UperNetForSemanticSegmentation", "UperNetPreTrainedModel"]
