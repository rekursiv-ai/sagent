from typing import Any
from torch import nn

import torch

from ...configuration_utils import PretrainedConfig
from ...modeling_outputs import BackboneOutput, ImageClassifierOutput
from ...processing_utils import Unpack
from ...utils import TransformersKwargs
from ...utils.backbone_utils import BackboneConfigMixin
from ....transformers.models.dinov2.modeling_dinov2 import (
    Dinov2Backbone,
    Dinov2Encoder,
    Dinov2ForImageClassification,
    Dinov2Model,
    Dinov2PatchEmbeddings,
    Dinov2PreTrainedModel,
)

logger = ...

class Dinov2WithRegistersConfig(BackboneConfigMixin, PretrainedConfig):
    def __init__(
        self,
        hidden_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        mlp_ratio=...,
        hidden_act=...,
        hidden_dropout_prob=...,
        attention_probs_dropout_prob=...,
        initializer_range=...,
        layer_norm_eps=...,
        image_size=...,
        patch_size=...,
        num_channels=...,
        qkv_bias=...,
        layerscale_value=...,
        drop_path_rate=...,
        use_swiglu_ffn=...,
        num_register_tokens=...,
        out_features=...,
        out_indices=...,
        apply_layernorm=...,
        reshape_hidden_states=...,
        **kwargs,
    ) -> None: ...

class Dinov2WithRegistersPatchEmbeddings(Dinov2PatchEmbeddings): ...

class Dinov2WithRegistersEmbeddings(nn.Module):
    def __init__(self, config: Dinov2WithRegistersConfig) -> None: ...
    def interpolate_pos_encoding(
        self, embeddings: torch.Tensor, height: int, width: int
    ) -> torch.Tensor: ...
    def forward(
        self, pixel_values: torch.Tensor, bool_masked_pos: torch.Tensor | None = ...
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class Dinov2WithRegistersEncoder(Dinov2Encoder): ...
class Dinov2WithRegistersPreTrainedModel(Dinov2PreTrainedModel): ...
class Dinov2WithRegistersModel(Dinov2Model): ...

class Dinov2WithRegistersForImageClassification(Dinov2ForImageClassification):
    def forward(
        self,
        pixel_values: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> ImageClassifierOutput: ...
    def __call__(self, *args: Any, **kwargs: Any) -> ImageClassifierOutput: ...

class Dinov2WithRegistersBackbone(Dinov2Backbone):
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self) -> Dinov2WithRegistersPatchEmbeddings: ...
    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: bool | None = ...,
        **kwargs,
    ) -> BackboneOutput: ...
    def __call__(self, *args: Any, **kwargs: Any) -> BackboneOutput: ...

__all__ = [
    "Dinov2WithRegistersBackbone",
    "Dinov2WithRegistersConfig",
    "Dinov2WithRegistersForImageClassification",
    "Dinov2WithRegistersModel",
    "Dinov2WithRegistersPreTrainedModel",
]
