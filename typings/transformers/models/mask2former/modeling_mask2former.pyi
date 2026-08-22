from dataclasses import dataclass
from typing import Any

from torch import Tensor, nn

import numpy as np
import torch

from .configuration_mask2former import Mask2FormerConfig
from ...file_utils import ModelOutput
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import compile_compatible_method_lru_cache
from ...utils import auto_docstring

"""PyTorch Mask2Former model."""
logger = ...

@dataclass
@auto_docstring(custom_intro=...)
class Mask2FormerPixelDecoderOutput(ModelOutput):
    multi_scale_features: tuple[torch.FloatTensor] | None = ...
    mask_features: torch.FloatTensor | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class Mask2FormerMaskedAttentionDecoderOutput(BaseModelOutputWithCrossAttentions):
    last_hidden_state: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: torch.FloatTensor | None = ...
    masks_queries_logits: tuple[torch.FloatTensor] | None = ...
    intermediate_hidden_states: tuple[torch.FloatTensor] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class Mask2FormerPixelLevelModuleOutput(ModelOutput):
    encoder_last_hidden_state: torch.FloatTensor | None = ...
    encoder_hidden_states: tuple[torch.FloatTensor] | None = ...
    decoder_last_hidden_state: torch.FloatTensor | None = ...
    decoder_hidden_states: tuple[torch.FloatTensor] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class Mask2FormerModelOutput(ModelOutput):
    encoder_last_hidden_state: torch.FloatTensor | None = ...
    pixel_decoder_last_hidden_state: torch.FloatTensor | None = ...
    transformer_decoder_last_hidden_state: torch.FloatTensor | None = ...
    encoder_hidden_states: tuple[torch.FloatTensor] | None = ...
    pixel_decoder_hidden_states: tuple[torch.FloatTensor] | None = ...
    transformer_decoder_hidden_states: tuple[torch.FloatTensor] | None = ...
    transformer_decoder_intermediate_states: tuple[torch.FloatTensor] | None = ...
    masks_queries_logits: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class Mask2FormerForUniversalSegmentationOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    class_queries_logits: torch.FloatTensor | None = ...
    masks_queries_logits: torch.FloatTensor | None = ...
    auxiliary_logits: list[dict[str, torch.FloatTensor]] | None = ...
    encoder_last_hidden_state: torch.FloatTensor | None = ...
    pixel_decoder_last_hidden_state: torch.FloatTensor | None = ...
    transformer_decoder_last_hidden_state: torch.FloatTensor | None = ...
    encoder_hidden_states: tuple[torch.FloatTensor] | None = ...
    pixel_decoder_hidden_states: tuple[torch.FloatTensor] | None = ...
    transformer_decoder_hidden_states: torch.FloatTensor | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...

def sample_point(
    input_features: torch.Tensor, point_coordinates: torch.Tensor, add_dim=..., **kwargs
) -> torch.Tensor: ...
def dice_loss(inputs: Tensor, labels: Tensor, num_masks: int) -> Tensor: ...
def sigmoid_cross_entropy_loss(
    inputs: torch.Tensor, labels: torch.Tensor, num_masks: int
) -> torch.Tensor: ...
def pair_wise_dice_loss(inputs: Tensor, labels: Tensor) -> Tensor: ...
def pair_wise_sigmoid_cross_entropy_loss(
    inputs: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor: ...

class Mask2FormerHungarianMatcher(nn.Module):
    def __init__(
        self,
        cost_class: float = ...,
        cost_mask: float = ...,
        cost_dice: float = ...,
        num_points: int = ...,
    ) -> None: ...
    @torch.no_grad()
    def forward(
        self,
        masks_queries_logits: torch.Tensor,
        class_queries_logits: torch.Tensor,
        mask_labels: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> list[tuple[Tensor]]: ...

class Mask2FormerLoss(nn.Module):
    def __init__(
        self, config: Mask2FormerConfig, weight_dict: dict[str, float]
    ) -> None: ...
    def loss_labels(
        self,
        class_queries_logits: Tensor,
        class_labels: list[Tensor],
        indices: tuple[np.array],
    ) -> dict[str, Tensor]: ...
    def loss_masks(
        self,
        masks_queries_logits: torch.Tensor,
        mask_labels: list[torch.Tensor],
        indices: tuple[np.array],
        num_masks: int,
    ) -> dict[str, torch.Tensor]: ...
    def calculate_uncertainty(self, logits: torch.Tensor) -> torch.Tensor: ...
    def sample_points_using_uncertainty(
        self,
        logits: torch.Tensor,
        uncertainty_function,
        num_points: int,
        oversample_ratio: int,
        importance_sample_ratio: float,
    ) -> torch.Tensor: ...
    def forward(
        self,
        masks_queries_logits: torch.Tensor,
        class_queries_logits: torch.Tensor,
        mask_labels: list[torch.Tensor],
        class_labels: list[torch.Tensor],
        auxiliary_predictions: dict[str, torch.Tensor] | None = ...,
    ) -> dict[str, torch.Tensor]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]: ...
    def get_num_masks(
        self, class_labels: torch.Tensor, device: torch.device
    ) -> torch.Tensor: ...

def multi_scale_deformable_attention(
    value: Tensor,
    value_spatial_shapes: Tensor | list[tuple],
    sampling_locations: Tensor,
    attention_weights: Tensor,
) -> Tensor: ...

class Mask2FormerSinePositionEmbedding(nn.Module):
    def __init__(
        self,
        num_pos_feats: int = ...,
        temperature: int = ...,
        normalize: bool = ...,
        scale: float | None = ...,
    ) -> None: ...
    @compile_compatible_method_lru_cache(maxsize=1)
    def forward(
        self,
        shape: torch.Size,
        device: torch.device | str,
        dtype: torch.dtype,
        mask: Tensor | None = ...,
    ) -> Tensor: ...

class Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, n_levels: int, n_points: int
    ) -> None: ...
    def with_pos_embed(
        self, tensor: torch.Tensor, position_embeddings: Tensor | None
    ):  # -> Tensor:
        ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        encoder_hidden_states=...,
        encoder_attention_mask=...,
        position_embeddings: torch.Tensor | None = ...,
        reference_points=...,
        spatial_shapes_list=...,
        level_start_index=...,
        output_attentions: bool = ...,
    ):  # -> tuple[Any, Tensor]:
        ...

class Mask2FormerPixelDecoderEncoderLayer(nn.Module):
    def __init__(self, config: Mask2FormerConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: torch.Tensor | None = ...,
        reference_points=...,
        spatial_shapes_list=...,
        level_start_index=...,
        output_attentions: bool = ...,
    ):  # -> tuple[Tensor, Any] | tuple[Tensor]:
        ...

class Mask2FormerPixelDecoderEncoderOnly(nn.Module):
    def __init__(self, config: Mask2FormerConfig) -> None: ...
    @staticmethod
    def get_reference_points(spatial_shapes_list, valid_ratios, device): ...
    def forward(
        self,
        inputs_embeds=...,
        attention_mask=...,
        position_embeddings=...,
        spatial_shapes_list=...,
        level_start_index=...,
        valid_ratios=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
    ):  # -> BaseModelOutput:
        ...

class Mask2FormerPixelDecoder(nn.Module):
    def __init__(self, config: Mask2FormerConfig, feature_channels) -> None: ...
    def get_valid_ratio(self, mask, dtype=...):  # -> Tensor:
        ...
    def forward(
        self,
        features,
        encoder_outputs=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
    ):  # -> Mask2FormerPixelDecoderOutput:
        ...

class Mask2FormerPixelLevelModule(nn.Module):
    def __init__(self, config: Mask2FormerConfig) -> None: ...
    def forward(
        self, pixel_values: Tensor, output_hidden_states: bool = ...
    ) -> Mask2FormerPixelLevelModuleOutput: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> Mask2FormerPixelLevelModuleOutput: ...

class Mask2FormerAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = ...,
        is_decoder: bool = ...,
        bias: bool = ...,
    ) -> None: ...
    def with_pos_embed(
        self, tensor: torch.Tensor, position_embeddings: Tensor | None
    ):  # -> Tensor:
        ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        position_embeddings: torch.Tensor | None = ...,
        key_value_states: torch.Tensor | None = ...,
        key_value_position_embeddings: torch.Tensor | None = ...,
        output_attentions: bool = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class Mask2FormerMaskedAttentionDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Mask2FormerConfig) -> None: ...
    def with_pos_embed(self, tensor, pos: Tensor | None): ...
    def forward_post(
        self,
        hidden_states: torch.Tensor,
        level_index: int | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_embeddings: torch.Tensor | None = ...,
        query_position_embeddings: torch.Tensor | None = ...,
        encoder_hidden_states: torch.Tensor | None = ...,
        encoder_attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
    ):  # -> tuple[Tensor, Any, Any] | tuple[Tensor]:
        ...
    def forward_pre(
        self,
        hidden_states: torch.Tensor,
        level_index: int | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_embeddings: torch.Tensor | None = ...,
        query_position_embeddings: torch.Tensor | None = ...,
        encoder_hidden_states: torch.Tensor | None = ...,
        encoder_attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
    ):  # -> tuple[Tensor, Any, Any] | tuple[Tensor]:
        ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        level_index: int | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_embeddings: torch.Tensor | None = ...,
        query_position_embeddings: torch.Tensor | None = ...,
        encoder_hidden_states: torch.Tensor | None = ...,
        encoder_attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
    ):  # -> tuple[Tensor, Any, Any] | tuple[Tensor]:
        ...

class Mask2FormerMaskedAttentionDecoder(nn.Module):
    def __init__(self, config: Mask2FormerConfig) -> None: ...
    def forward(
        self,
        inputs_embeds: torch.Tensor | None = ...,
        multi_stage_positional_embeddings: torch.Tensor | None = ...,
        pixel_embeddings: torch.Tensor | None = ...,
        encoder_hidden_states: torch.Tensor | None = ...,
        query_position_embeddings: torch.Tensor | None = ...,
        feature_size_list: list | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ):  # -> tuple[Tensor | Any | tuple[Tensor | Any, ...] | tuple[()] | tuple[Any, ...] | tuple[Any] | tuple[Any, Any], ...] | Mask2FormerMaskedAttentionDecoderOutput:
        ...

class Mask2FormerPredictionBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, activation: nn.Module) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...

class Mask2FormerMLPPredictionHead(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = ...
    ) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...

class Mask2FormerMaskPredictor(nn.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, mask_feature_size: torch.Tensor
    ) -> None: ...
    def forward(
        self,
        outputs: torch.Tensor,
        pixel_embeddings: torch.Tensor,
        attention_mask_target_size: int | None = ...,
    ):  # -> tuple[Tensor, Tensor]:
        ...

class Mask2FormerTransformerModule(nn.Module):
    def __init__(self, in_features: int, config: Mask2FormerConfig) -> None: ...
    def forward(
        self,
        multi_scale_features: list[Tensor],
        mask_features: Tensor,
        output_hidden_states: bool = ...,
        output_attentions: bool = ...,
    ) -> Mask2FormerMaskedAttentionDecoderOutput: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> Mask2FormerMaskedAttentionDecoderOutput: ...

@auto_docstring
class Mask2FormerPreTrainedModel(PreTrainedModel):
    config: Mask2FormerConfig
    base_model_prefix = ...
    main_input_name = ...

@auto_docstring
class Mask2FormerModel(Mask2FormerPreTrainedModel):
    main_input_name = ...
    def __init__(self, config: Mask2FormerConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: Tensor,
        pixel_mask: Tensor | None = ...,
        output_hidden_states: bool | None = ...,
        output_attentions: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> Mask2FormerModelOutput: ...

@auto_docstring(custom_intro=...)
class Mask2FormerForUniversalSegmentation(Mask2FormerPreTrainedModel):
    main_input_name = ...
    def __init__(self, config: Mask2FormerConfig) -> None: ...
    def get_loss_dict(
        self,
        masks_queries_logits: Tensor,
        class_queries_logits: Tensor,
        mask_labels: Tensor,
        class_labels: Tensor,
        auxiliary_predictions: dict[str, Tensor],
    ) -> dict[str, Tensor]: ...
    def get_loss(self, loss_dict: dict[str, Tensor]) -> Tensor: ...
    def get_auxiliary_logits(
        self, classes: torch.Tensor, output_masks: torch.Tensor
    ):  # -> list[dict[str, Tensor]]:
        ...
    @auto_docstring
    def forward(
        self,
        pixel_values: Tensor,
        mask_labels: list[Tensor] | None = ...,
        class_labels: list[Tensor] | None = ...,
        pixel_mask: Tensor | None = ...,
        output_hidden_states: bool | None = ...,
        output_auxiliary_logits: bool | None = ...,
        output_attentions: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> Mask2FormerForUniversalSegmentationOutput: ...

__all__ = [
    "Mask2FormerForUniversalSegmentation",
    "Mask2FormerModel",
    "Mask2FormerPreTrainedModel",
]
