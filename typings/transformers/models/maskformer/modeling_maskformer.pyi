from typing import Any
from dataclasses import dataclass

from torch import Tensor, nn

import numpy as np
import torch

from .configuration_maskformer import MaskFormerConfig
from ..detr import DetrConfig
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import compile_compatible_method_lru_cache
from ...utils import (
    ModelOutput,
    auto_docstring,
)

"""PyTorch MaskFormer model."""
logger = ...

@dataclass
@auto_docstring(custom_intro=...)
class DetrDecoderOutput(BaseModelOutputWithCrossAttentions):
    intermediate_hidden_states: torch.FloatTensor | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class MaskFormerPixelLevelModuleOutput(ModelOutput):
    encoder_last_hidden_state: torch.FloatTensor | None = ...
    decoder_last_hidden_state: torch.FloatTensor | None = ...
    encoder_hidden_states: tuple[torch.FloatTensor] | None = ...
    decoder_hidden_states: tuple[torch.FloatTensor] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class MaskFormerPixelDecoderOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class MaskFormerModelOutput(ModelOutput):
    encoder_last_hidden_state: torch.FloatTensor | None = ...
    pixel_decoder_last_hidden_state: torch.FloatTensor | None = ...
    transformer_decoder_last_hidden_state: torch.FloatTensor | None = ...
    encoder_hidden_states: tuple[torch.FloatTensor] | None = ...
    pixel_decoder_hidden_states: tuple[torch.FloatTensor] | None = ...
    transformer_decoder_hidden_states: tuple[torch.FloatTensor] | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class MaskFormerForInstanceSegmentationOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    class_queries_logits: torch.FloatTensor | None = ...
    masks_queries_logits: torch.FloatTensor | None = ...
    auxiliary_logits: torch.FloatTensor | None = ...
    encoder_last_hidden_state: torch.FloatTensor | None = ...
    pixel_decoder_last_hidden_state: torch.FloatTensor | None = ...
    transformer_decoder_last_hidden_state: torch.FloatTensor | None = ...
    encoder_hidden_states: tuple[torch.FloatTensor] | None = ...
    pixel_decoder_hidden_states: tuple[torch.FloatTensor] | None = ...
    transformer_decoder_hidden_states: tuple[torch.FloatTensor] | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...

def upsample_like(pixel_values: Tensor, like: Tensor, mode: str = ...) -> Tensor: ...
def dice_loss(inputs: Tensor, labels: Tensor, num_masks: int) -> Tensor: ...
def sigmoid_focal_loss(
    inputs: Tensor,
    labels: Tensor,
    num_masks: int,
    alpha: float = ...,
    gamma: float = ...,
) -> Tensor: ...
def pair_wise_dice_loss(inputs: Tensor, labels: Tensor) -> Tensor: ...
def pair_wise_sigmoid_focal_loss(
    inputs: Tensor, labels: Tensor, alpha: float = ..., gamma: float = ...
) -> Tensor: ...

class DetrAttention(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, dropout: float = ..., bias: bool = ...
    ) -> None: ...
    def with_pos_embed(
        self, tensor: torch.Tensor, object_queries: Tensor | None
    ):  # -> Tensor:
        ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        object_queries: torch.Tensor | None = ...,
        key_value_states: torch.Tensor | None = ...,
        spatial_position_embeddings: torch.Tensor | None = ...,
        output_attentions: bool = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class DetrDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: DetrConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        object_queries: torch.Tensor | None = ...,
        query_position_embeddings: torch.Tensor | None = ...,
        encoder_hidden_states: torch.Tensor | None = ...,
        encoder_attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
    ):  # -> tuple[Tensor, Any, Any | None] | tuple[Tensor]:
        ...

class DetrDecoder(nn.Module):
    def __init__(self, config: DetrConfig) -> None: ...
    def forward(
        self,
        inputs_embeds=...,
        attention_mask=...,
        encoder_hidden_states=...,
        encoder_attention_mask=...,
        object_queries=...,
        query_position_embeddings=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
    ):  # -> tuple[Any | tuple[Any, ...] | tuple[()] | Tensor, ...] | DetrDecoderOutput:
        ...

class MaskFormerHungarianMatcher(nn.Module):
    def __init__(
        self, cost_class: float = ..., cost_mask: float = ..., cost_dice: float = ...
    ) -> None: ...
    @torch.no_grad()
    def forward(
        self, masks_queries_logits, class_queries_logits, mask_labels, class_labels
    ) -> list[tuple[Tensor]]: ...
    def __repr__(self):  # -> str:
        ...

class MaskFormerLoss(nn.Module):
    def __init__(
        self,
        num_labels: int,
        matcher: MaskFormerHungarianMatcher,
        weight_dict: dict[str, float],
        eos_coef: float,
    ) -> None: ...
    def loss_labels(
        self,
        class_queries_logits: Tensor,
        class_labels: list[Tensor],
        indices: tuple[np.array],
    ) -> dict[str, Tensor]: ...
    def loss_masks(
        self,
        masks_queries_logits: Tensor,
        mask_labels: list[Tensor],
        indices: tuple[np.array],
        num_masks: int,
    ) -> dict[str, Tensor]: ...
    def forward(
        self,
        masks_queries_logits: Tensor,
        class_queries_logits: Tensor,
        mask_labels: list[Tensor],
        class_labels: list[Tensor],
        auxiliary_predictions: dict[str, Tensor] | None = ...,
    ) -> dict[str, Tensor]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, Tensor]: ...
    def get_num_masks(
        self, class_labels: torch.Tensor, device: torch.device
    ) -> torch.Tensor: ...

class MaskFormerFPNConvLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int = ...,
        padding: int = ...,
    ) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...

class MaskFormerFPNLayer(nn.Module):
    def __init__(self, in_features: int, lateral_features: int) -> None: ...
    def forward(self, down: Tensor, left: Tensor) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...

class MaskFormerFPNModel(nn.Module):
    def __init__(
        self, in_features: int, lateral_widths: list[int], feature_size: int = ...
    ) -> None: ...
    def forward(self, features: list[Tensor]) -> list[Tensor]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> list[Tensor]: ...

class MaskFormerPixelDecoder(nn.Module):
    def __init__(
        self, *args, feature_size: int = ..., mask_feature_size: int = ..., **kwargs
    ) -> None: ...
    def forward(
        self,
        features: list[Tensor],
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ) -> MaskFormerPixelDecoderOutput: ...
    def __call__(self, *args: Any, **kwargs: Any) -> MaskFormerPixelDecoderOutput: ...

class MaskFormerSinePositionEmbedding(nn.Module):
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

class PredictionBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, activation: nn.Module) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...

class MaskformerMLPPredictionHead(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = ...
    ) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...

class MaskFormerPixelLevelModule(nn.Module):
    def __init__(self, config: MaskFormerConfig) -> None: ...
    def forward(
        self,
        pixel_values: Tensor,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ) -> MaskFormerPixelLevelModuleOutput: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> MaskFormerPixelLevelModuleOutput: ...

class MaskFormerTransformerModule(nn.Module):
    def __init__(self, in_features: int, config: MaskFormerConfig) -> None: ...
    def forward(
        self,
        image_features: Tensor,
        output_hidden_states: bool = ...,
        output_attentions: bool = ...,
        return_dict: bool | None = ...,
    ) -> DetrDecoderOutput: ...
    def __call__(self, *args: Any, **kwargs: Any) -> DetrDecoderOutput: ...

@auto_docstring
class MaskFormerPreTrainedModel(PreTrainedModel):
    config: MaskFormerConfig
    base_model_prefix = ...
    main_input_name = ...

@auto_docstring
class MaskFormerModel(MaskFormerPreTrainedModel):
    def __init__(self, config: MaskFormerConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: Tensor,
        pixel_mask: Tensor | None = ...,
        output_hidden_states: bool | None = ...,
        output_attentions: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> MaskFormerModelOutput: ...

class MaskFormerForInstanceSegmentation(MaskFormerPreTrainedModel):
    def __init__(self, config: MaskFormerConfig) -> None: ...
    def get_loss_dict(
        self,
        masks_queries_logits: Tensor,
        class_queries_logits: Tensor,
        mask_labels: Tensor,
        class_labels: Tensor,
        auxiliary_logits: dict[str, Tensor],
    ) -> dict[str, Tensor]: ...
    def get_loss(self, loss_dict: dict[str, Tensor]) -> Tensor: ...
    def get_logits(
        self, outputs: MaskFormerModelOutput
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: Tensor,
        mask_labels: list[Tensor] | None = ...,
        class_labels: list[Tensor] | None = ...,
        pixel_mask: Tensor | None = ...,
        output_auxiliary_logits: bool | None = ...,
        output_hidden_states: bool | None = ...,
        output_attentions: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> MaskFormerForInstanceSegmentationOutput: ...

__all__ = [
    "MaskFormerForInstanceSegmentation",
    "MaskFormerModel",
    "MaskFormerPreTrainedModel",
]
