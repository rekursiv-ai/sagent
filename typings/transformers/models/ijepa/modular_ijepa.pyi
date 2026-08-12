from typing import Any
from transformers.models.ijepa.configuration_ijepa import IJepaConfig

import torch

from ..vit.modeling_vit import (
    ViTEmbeddings,
    ViTForImageClassification,
    ViTModel,
    ViTPreTrainedModel,
)
from ...modeling_outputs import ImageClassifierOutput
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring

class IJepaEmbeddings(ViTEmbeddings):
    def __init__(self, config: IJepaConfig, use_mask_token: bool = ...) -> None: ...
    def interpolate_pos_encoding(
        self, embeddings: torch.Tensor, height: int, width: int
    ) -> torch.Tensor: ...
    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: torch.BoolTensor | None = ...,
        interpolate_pos_encoding: bool = ...,
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

@auto_docstring
class IJepaPreTrainedModel(ViTPreTrainedModel): ...

class IJepaModel(IJepaPreTrainedModel, ViTModel):
    def __init__(
        self,
        config: IJepaConfig,
        add_pooling_layer: bool = ...,
        use_mask_token: bool = ...,
    ) -> None: ...

@auto_docstring(custom_intro=...)
class IJepaForImageClassification(IJepaPreTrainedModel, ViTForImageClassification):
    def __init__(self, config: IJepaConfig) -> None: ...
    def forward(
        self,
        pixel_values: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        interpolate_pos_encoding: bool | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> ImageClassifierOutput: ...
    def __call__(self, *args: Any, **kwargs: Any) -> ImageClassifierOutput: ...

__all__ = ["IJepaForImageClassification", "IJepaModel", "IJepaPreTrainedModel"]
