from typing import Any

from torch import nn
from transformers.models.siglip.configuration_siglip import (
    SiglipConfig,
    SiglipTextConfig,
    SiglipVisionConfig,
)
from transformers.models.siglip.modeling_siglip import (
    BaseModelOutputWithPooling,
    ImageClassifierOutput,
    SiglipForImageClassification,
    SiglipModel,
    SiglipMultiheadAttentionPoolingHead,
    SiglipOutput,
    SiglipPreTrainedModel,
    SiglipTextModel,
    SiglipTextModelOutput,
    SiglipVisionModel,
    SiglipVisionModelOutput,
    SiglipVisionTransformer,
)

import torch

from ...utils import auto_docstring, filter_out_non_signature_kwargs

class Siglip2TextConfig(SiglipTextConfig): ...

class Siglip2VisionConfig(SiglipVisionConfig):
    def __init__(
        self,
        hidden_size=...,
        intermediate_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        num_channels=...,
        num_patches=...,
        patch_size=...,
        hidden_act=...,
        layer_norm_eps=...,
        attention_dropout=...,
        **kwargs,
    ) -> None: ...

class Siglip2Config(SiglipConfig): ...
class Siglip2VisionOutput(SiglipVisionModelOutput): ...
class Siglip2TextOutput(SiglipTextModelOutput): ...
class Siglip2Output(SiglipOutput): ...

class Siglip2VisionEmbeddings(nn.Module):
    def __init__(self, config: Siglip2VisionConfig) -> None: ...
    @staticmethod
    def resize_positional_embeddings(
        positional_embeddings: torch.Tensor,
        spatial_shapes: torch.LongTensor,
        max_length: int,
    ) -> torch.Tensor: ...
    def forward(
        self, pixel_values: torch.FloatTensor, spatial_shapes: torch.LongTensor
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class Siglip2VisionTransformer(SiglipVisionTransformer):
    def __init__(self, config: Siglip2VisionConfig) -> None: ...
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        attention_mask: torch.Tensor,
        spatial_shapes: torch.LongTensor,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
    ) -> BaseModelOutputWithPooling: ...
    def __call__(self, *args: Any, **kwargs: Any) -> BaseModelOutputWithPooling: ...

class Siglip2PreTrainedModel(SiglipPreTrainedModel): ...
class Siglip2TextModel(SiglipTextModel): ...

class Siglip2MultiheadAttentionPoolingHead(SiglipMultiheadAttentionPoolingHead):
    def __init__(self, config: Siglip2VisionConfig) -> None: ...
    def forward(
        self, hidden_state: torch.Tensor, attention_mask: torch.Tensor | None = ...
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class Siglip2VisionModel(SiglipVisionModel):
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_attention_mask: torch.Tensor,
        spatial_shapes: torch.LongTensor,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
    ) -> BaseModelOutputWithPooling: ...
    def __call__(self, *args: Any, **kwargs: Any) -> BaseModelOutputWithPooling: ...

class Siglip2Model(SiglipModel):
    @filter_out_non_signature_kwargs()
    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        pixel_attention_mask: torch.Tensor | None = ...,
        spatial_shapes: torch.LongTensor | None = ...,
    ) -> torch.FloatTensor: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        pixel_attention_mask: torch.Tensor | None = ...,
        spatial_shapes: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        return_loss: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
    ) -> Siglip2Output: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Siglip2Output: ...

class Siglip2ForImageClassification(SiglipForImageClassification):
    def forward(
        self,
        pixel_values: torch.Tensor | None = ...,
        pixel_attention_mask: torch.Tensor | None = ...,
        spatial_shapes: torch.LongTensor | None = ...,
        labels: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
    ) -> ImageClassifierOutput: ...
    def __call__(self, *args: Any, **kwargs: Any) -> ImageClassifierOutput: ...

__all__ = [
    "Siglip2Config",
    "Siglip2ForImageClassification",
    "Siglip2Model",
    "Siglip2PreTrainedModel",
    "Siglip2TextConfig",
    "Siglip2TextModel",
    "Siglip2VisionConfig",
    "Siglip2VisionModel",
]
