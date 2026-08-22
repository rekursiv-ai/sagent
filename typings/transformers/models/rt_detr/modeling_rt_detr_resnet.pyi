from typing import Any

from torch import Tensor, nn

from .configuration_rt_detr_resnet import RTDetrResNetConfig
from ...modeling_outputs import BackboneOutput, BaseModelOutputWithNoAttention
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring
from ...utils.backbone_utils import BackboneMixin

"""
PyTorch RTDetr specific ResNet model. The main difference between hugginface ResNet model is that this RTDetrResNet model forces to use shortcut at the first layer in the resnet-18/34 models.
See https://github.com/lyuwenyu/RT-DETR/blob/5b628eaa0a2fc25bdafec7e6148d5296b144af85/rtdetr_pytorch/src/nn/backbone/presnet.py#L126 for details.
"""
logger = ...

class RTDetrResNetConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = ...,
        stride: int = ...,
        activation: str = ...,
    ) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...

class RTDetrResNetEmbeddings(nn.Module):
    def __init__(self, config: RTDetrResNetConfig) -> None: ...
    def forward(self, pixel_values: Tensor) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...

class RTDetrResNetShortCut(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, stride: int = ...
    ) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...

class RTDetrResNetBasicLayer(nn.Module):
    def __init__(
        self,
        config: RTDetrResNetConfig,
        in_channels: int,
        out_channels: int,
        stride: int = ...,
        should_apply_shortcut: bool = ...,
    ) -> None: ...
    def forward(self, hidden_state): ...

class RTDetrResNetBottleNeckLayer(nn.Module):
    def __init__(
        self,
        config: RTDetrResNetConfig,
        in_channels: int,
        out_channels: int,
        stride: int = ...,
    ) -> None: ...
    def forward(self, hidden_state): ...

class RTDetrResNetStage(nn.Module):
    def __init__(
        self,
        config: RTDetrResNetConfig,
        in_channels: int,
        out_channels: int,
        stride: int = ...,
        depth: int = ...,
    ) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...

class RTDetrResNetEncoder(nn.Module):
    def __init__(self, config: RTDetrResNetConfig) -> None: ...
    def forward(
        self,
        hidden_state: Tensor,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ) -> BaseModelOutputWithNoAttention: ...
    def __call__(self, *args: Any, **kwargs: Any) -> BaseModelOutputWithNoAttention: ...

@auto_docstring
class RTDetrResNetPreTrainedModel(PreTrainedModel):
    config: RTDetrResNetConfig
    base_model_prefix = ...
    main_input_name = ...
    _no_split_modules = ...

@auto_docstring(custom_intro=...)
class RTDetrResNetBackbone(RTDetrResNetPreTrainedModel, BackboneMixin):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: Tensor,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> BackboneOutput: ...

__all__ = ["RTDetrResNetBackbone", "RTDetrResNetPreTrainedModel"]
