from typing import Any

from torch import Tensor, nn

import torch

from ..rt_detr.modeling_rt_detr_resnet import RTDetrResNetConvLayer
from ...configuration_utils import PretrainedConfig
from ...modeling_outputs import (
    BackboneOutput,
    BaseModelOutputWithNoAttention,
    ImageClassifierOutputWithNoAttention,
)
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring
from ...utils.backbone_utils import BackboneConfigMixin, BackboneMixin

class HGNetV2Config(BackboneConfigMixin, PretrainedConfig):
    def __init__(
        self,
        num_channels=...,
        embedding_size=...,
        depths=...,
        hidden_sizes=...,
        hidden_act=...,
        out_features=...,
        out_indices=...,
        stem_channels=...,
        stage_in_channels=...,
        stage_mid_channels=...,
        stage_out_channels=...,
        stage_num_blocks=...,
        stage_downsample=...,
        stage_light_block=...,
        stage_kernel_size=...,
        stage_numb_of_layers=...,
        use_learnable_affine_block=...,
        initializer_range=...,
        **kwargs,
    ) -> None: ...

@auto_docstring
class HGNetV2PreTrainedModel(PreTrainedModel):
    config: HGNetV2Config
    base_model_prefix = ...
    main_input_name = ...
    _no_split_modules = ...

class HGNetV2LearnableAffineBlock(nn.Module):
    def __init__(self, scale_value: float = ..., bias_value: float = ...) -> None: ...
    def forward(self, hidden_state: Tensor) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...

class HGNetV2ConvLayer(RTDetrResNetConvLayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = ...,
        groups: int = ...,
        activation: str = ...,
        use_learnable_affine_block: bool = ...,
    ) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...

class HGNetV2ConvLayerLight(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        use_learnable_affine_block: bool = ...,
    ) -> None: ...
    def forward(self, hidden_state: Tensor) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...

class HGNetV2Embeddings(nn.Module):
    def __init__(self, config: HGNetV2Config) -> None: ...
    def forward(self, pixel_values: Tensor) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...

class HGNetV2BasicLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        middle_channels: int,
        out_channels: int,
        layer_num: int,
        kernel_size: int = ...,
        residual: bool = ...,
        light_block: bool = ...,
        drop_path: float = ...,
        use_learnable_affine_block: bool = ...,
    ) -> None: ...
    def forward(self, hidden_state: Tensor) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...

class HGNetV2Stage(nn.Module):
    def __init__(
        self, config: HGNetV2Config, stage_index: int, drop_path: float = ...
    ) -> None: ...
    def forward(self, hidden_state: Tensor) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...

class HGNetV2Encoder(nn.Module):
    def __init__(self, config: HGNetV2Config) -> None: ...
    def forward(
        self,
        hidden_state: Tensor,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ) -> BaseModelOutputWithNoAttention: ...
    def __call__(self, *args: Any, **kwargs: Any) -> BaseModelOutputWithNoAttention: ...

class HGNetV2Backbone(HGNetV2PreTrainedModel, BackboneMixin):
    def __init__(self, config: HGNetV2Config) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: Tensor,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> BackboneOutput: ...

@auto_docstring(custom_intro=...)
class HGNetV2ForImageClassification(HGNetV2PreTrainedModel):
    def __init__(self, config: HGNetV2Config) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> ImageClassifierOutputWithNoAttention: ...

__all__ = [
    "HGNetV2Backbone",
    "HGNetV2Config",
    "HGNetV2ForImageClassification",
    "HGNetV2PreTrainedModel",
]
