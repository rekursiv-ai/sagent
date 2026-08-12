from typing import Any
from torch import nn

import torch

from .configuration_mobilenet_v1 import MobileNetV1Config
from ...modeling_outputs import (
    BaseModelOutputWithPoolingAndNoAttention,
    ImageClassifierOutputWithNoAttention,
)
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring

"""PyTorch MobileNetV1 model."""
logger = ...

def load_tf_weights_in_mobilenet_v1(model, config, tf_checkpoint_path): ...
def apply_tf_padding(features: torch.Tensor, conv_layer: nn.Conv2d) -> torch.Tensor: ...

class MobileNetV1ConvLayer(nn.Module):
    def __init__(
        self,
        config: MobileNetV1Config,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int | None = ...,
        groups: int | None = ...,
        bias: bool = ...,
        use_normalization: bool | None = ...,
        use_activation: bool | str | None = ...,
    ) -> None: ...
    def forward(self, features: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

@auto_docstring
class MobileNetV1PreTrainedModel(PreTrainedModel):
    config: MobileNetV1Config
    load_tf_weights = ...
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...

@auto_docstring
class MobileNetV1Model(MobileNetV1PreTrainedModel):
    def __init__(
        self, config: MobileNetV1Config, add_pooling_layer: bool = ...
    ) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutputWithPoolingAndNoAttention: ...

@auto_docstring(custom_intro=...)
class MobileNetV1ForImageClassification(MobileNetV1PreTrainedModel):
    def __init__(self, config: MobileNetV1Config) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor | None = ...,
        output_hidden_states: bool | None = ...,
        labels: torch.Tensor | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | ImageClassifierOutputWithNoAttention: ...

__all__ = [
    "MobileNetV1ForImageClassification",
    "MobileNetV1Model",
    "MobileNetV1PreTrainedModel",
    "load_tf_weights_in_mobilenet_v1",
]
