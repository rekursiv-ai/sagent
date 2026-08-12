from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from torch import Tensor, nn

import torch

from .configuration_owlv2 import Owlv2Config, Owlv2TextConfig, Owlv2VisionConfig
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    auto_docstring,
    filter_out_non_signature_kwargs,
)

"""PyTorch OWLv2 model."""
logger = ...

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor: ...
def owlv2_loss(similarity: torch.Tensor) -> torch.Tensor: ...

@dataclass
@auto_docstring
class Owlv2Output(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits_per_image: torch.FloatTensor | None = ...
    logits_per_text: torch.FloatTensor | None = ...
    text_embeds: torch.FloatTensor | None = ...
    image_embeds: torch.FloatTensor | None = ...
    text_model_output: BaseModelOutputWithPooling = ...
    vision_model_output: BaseModelOutputWithPooling = ...
    def to_tuple(self) -> tuple[Any]: ...

def box_area(boxes: Tensor) -> Tensor: ...
def box_iou(boxes1, boxes2):  # -> tuple[Any, Any]:
    ...
def generalized_box_iou(boxes1, boxes2): ...

@dataclass
@auto_docstring(custom_intro=...)
class Owlv2ObjectDetectionOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    loss_dict: dict | None = ...
    logits: torch.FloatTensor | None = ...
    objectness_logits: torch.FloatTensor | None = ...
    pred_boxes: torch.FloatTensor | None = ...
    text_embeds: torch.FloatTensor | None = ...
    image_embeds: torch.FloatTensor | None = ...
    class_embeds: torch.FloatTensor | None = ...
    text_model_output: BaseModelOutputWithPooling = ...
    vision_model_output: BaseModelOutputWithPooling = ...
    def to_tuple(self) -> tuple[Any]: ...

@dataclass
@auto_docstring(custom_intro=...)
class Owlv2ImageGuidedObjectDetectionOutput(ModelOutput):
    logits: torch.FloatTensor | None = ...
    image_embeds: torch.FloatTensor | None = ...
    query_image_embeds: torch.FloatTensor | None = ...
    target_pred_boxes: torch.FloatTensor | None = ...
    query_pred_boxes: torch.FloatTensor | None = ...
    class_embeds: torch.FloatTensor | None = ...
    text_model_output: BaseModelOutputWithPooling = ...
    vision_model_output: BaseModelOutputWithPooling = ...
    def to_tuple(self) -> tuple[Any]: ...

class Owlv2VisionEmbeddings(nn.Module):
    def __init__(self, config: Owlv2VisionConfig) -> None: ...
    def interpolate_pos_encoding(
        self, embeddings: torch.Tensor, height: int, width: int
    ) -> torch.Tensor: ...
    def forward(
        self, pixel_values: torch.FloatTensor, interpolate_pos_encoding: bool = ...
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class Owlv2TextEmbeddings(nn.Module):
    def __init__(self, config: Owlv2TextConfig) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class Owlv2Attention(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        causal_attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class Owlv2MLP(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class Owlv2EncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Owlv2Config) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.FloatTensor]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple[torch.FloatTensor]: ...

@auto_docstring
class Owlv2PreTrainedModel(PreTrainedModel):
    config: Owlv2Config
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...

class Owlv2Encoder(nn.Module):
    def __init__(self, config: Owlv2Config) -> None: ...
    def forward(
        self,
        inputs_embeds,
        attention_mask: torch.Tensor | None = ...,
        causal_attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutput: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple | BaseModelOutput: ...

class Owlv2TextTransformer(nn.Module):
    def __init__(self, config: Owlv2TextConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutputWithPooling: ...

class Owlv2TextModel(Owlv2PreTrainedModel):
    config: Owlv2TextConfig
    def __init__(self, config: Owlv2TextConfig) -> None: ...
    def get_input_embeddings(self) -> nn.Module: ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutputWithPooling: ...

class Owlv2VisionTransformer(nn.Module):
    def __init__(self, config: Owlv2VisionConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        interpolate_pos_encoding: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutputWithPooling: ...

class Owlv2VisionModel(Owlv2PreTrainedModel):
    config: Owlv2VisionConfig
    main_input_name = ...
    def __init__(self, config: Owlv2VisionConfig) -> None: ...
    def get_input_embeddings(self) -> nn.Module: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        interpolate_pos_encoding: bool = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutputWithPooling: ...

@auto_docstring
class Owlv2Model(Owlv2PreTrainedModel):
    config: Owlv2Config
    def __init__(self, config: Owlv2Config) -> None: ...
    @filter_out_non_signature_kwargs()
    @auto_docstring
    def get_text_features(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = ...
    ) -> torch.FloatTensor: ...
    @filter_out_non_signature_kwargs()
    @auto_docstring
    def get_image_features(
        self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = ...
    ) -> torch.FloatTensor: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        return_loss: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        interpolate_pos_encoding: bool = ...,
        return_base_image_embeds: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | Owlv2Output: ...

class Owlv2BoxPredictionHead(nn.Module):
    def __init__(self, config: Owlv2Config, out_dim: int = ...) -> None: ...
    def forward(self, image_features: torch.Tensor) -> torch.FloatTensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.FloatTensor: ...

class Owlv2ClassPredictionHead(nn.Module):
    def __init__(self, config: Owlv2Config) -> None: ...
    def forward(
        self,
        image_embeds: torch.FloatTensor,
        query_embeds: torch.FloatTensor | None,
        query_mask: torch.Tensor | None,
    ) -> tuple[torch.FloatTensor]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple[torch.FloatTensor]: ...

class Owlv2ForObjectDetection(Owlv2PreTrainedModel):
    config: Owlv2Config
    def __init__(self, config: Owlv2Config) -> None: ...
    @staticmethod
    def normalize_grid_corner_coordinates(
        num_patches_height: int, num_patches_width: int
    ) -> torch.Tensor: ...
    def objectness_predictor(
        self, image_features: torch.FloatTensor
    ) -> torch.FloatTensor: ...
    @lru_cache(maxsize=2)  # noqa: B019 -- mirrors upstream @lru_cache on a method; not executed in a stub
    def compute_box_bias(
        self,
        num_patches_height: int,
        num_patches_width: int,
        feature_map: torch.FloatTensor | None = ...,
    ) -> torch.Tensor: ...
    def box_predictor(
        self,
        image_feats: torch.FloatTensor,
        feature_map: torch.FloatTensor,
        interpolate_pos_encoding: bool = ...,
    ) -> torch.FloatTensor: ...
    def class_predictor(
        self,
        image_feats: torch.FloatTensor,
        query_embeds: torch.FloatTensor | None = ...,
        query_mask: torch.Tensor | None = ...,
    ) -> tuple[torch.FloatTensor]: ...
    def image_text_embedder(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.FloatTensor,
        attention_mask: torch.Tensor,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        interpolate_pos_encoding: bool = ...,
    ) -> tuple[torch.FloatTensor]: ...
    def image_embedder(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        interpolate_pos_encoding: bool = ...,
    ) -> tuple[torch.FloatTensor]: ...
    def embed_image_query(
        self,
        query_image_features: torch.FloatTensor,
        query_feature_map: torch.FloatTensor,
        interpolate_pos_encoding: bool = ...,
    ) -> torch.FloatTensor: ...
    @auto_docstring
    def image_guided_detection(
        self,
        pixel_values: torch.FloatTensor,
        query_pixel_values: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        interpolate_pos_encoding: bool = ...,
        return_dict: bool | None = ...,
    ) -> Owlv2ImageGuidedObjectDetectionOutput: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.FloatTensor,
        attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        interpolate_pos_encoding: bool = ...,
        return_dict: bool | None = ...,
    ) -> Owlv2ObjectDetectionOutput: ...

__all__ = [
    "Owlv2ForObjectDetection",
    "Owlv2Model",
    "Owlv2PreTrainedModel",
    "Owlv2TextModel",
    "Owlv2VisionModel",
]
