from typing import Any
from dataclasses import dataclass

from torch import Tensor, nn

import numpy as np
import torch

from .configuration_oneformer import OneFormerConfig
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import compile_compatible_method_lru_cache
from ...utils import (
    ModelOutput,
    auto_docstring,
)

"""PyTorch OneFormer model."""
logger = ...

def multi_scale_deformable_attention(
    value: Tensor,
    value_spatial_shapes: Tensor | list[tuple],
    sampling_locations: Tensor,
    attention_weights: Tensor,
) -> Tensor: ...
def dice_loss(inputs: Tensor, labels: Tensor, num_masks: int) -> Tensor: ...
def sigmoid_cross_entropy_loss(
    inputs: torch.Tensor, labels: torch.Tensor, num_masks: int
) -> torch.Tensor: ...
def pair_wise_dice_loss(inputs: Tensor, labels: Tensor) -> Tensor: ...
def pair_wise_sigmoid_cross_entropy_loss(
    inputs: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor: ...
def sample_point(
    input_features: torch.Tensor, point_coordinates: torch.Tensor, add_dim=..., **kwargs
) -> torch.Tensor: ...

class OneFormerHungarianMatcher(nn.Module):
    def __init__(
        self,
        cost_class: float = ...,
        cost_mask: float = ...,
        cost_dice: float = ...,
        num_points: int = ...,
    ) -> None: ...
    @torch.no_grad()
    def forward(
        self, masks_queries_logits, class_queries_logits, mask_labels, class_labels
    ) -> list[tuple[Tensor]]: ...

class OneFormerLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        matcher: OneFormerHungarianMatcher,
        weight_dict: dict[str, float],
        eos_coef: float,
        num_points: int,
        oversample_ratio: float,
        importance_sample_ratio: float,
        contrastive_temperature: float | None = ...,
    ) -> None: ...
    def loss_contrastive(
        self, contrastive_queries_logits: Tensor, text_queries: Tensor
    ):  # -> dict[str, Tensor]:
        ...
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
        masks_queries_logits: Tensor,
        class_queries_logits: Tensor,
        contrastive_queries_logits: Tensor,
        mask_labels: list[Tensor],
        class_labels: list[Tensor],
        text_queries: Tensor,
        auxiliary_predictions: dict[str, Tensor] | None = ...,
        calculate_contrastive_loss: bool = ...,
    ) -> dict[str, Tensor]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, Tensor]: ...
    def get_num_masks(
        self, class_labels: torch.Tensor, device: torch.device
    ) -> torch.Tensor: ...

@dataclass
@auto_docstring(custom_intro=...)
class OneFormerTransformerDecoderOutput(BaseModelOutput):
    object_queries: torch.FloatTensor | None = ...
    contrastive_logits: torch.FloatTensor | None = ...
    prediction_masks: torch.FloatTensor | None = ...
    prediction_class: torch.FloatTensor | None = ...
    auxiliary_predictions: tuple[dict[str, torch.FloatTensor]] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class OneFormerPixelDecoderOutput(ModelOutput):
    multi_scale_features: tuple[torch.FloatTensor] | None = ...
    mask_features: torch.FloatTensor | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class OneFormerPixelLevelModuleOutput(ModelOutput):
    encoder_features: list[torch.FloatTensor] | None = ...
    decoder_features: list[torch.FloatTensor] | None = ...
    decoder_last_feature: torch.FloatTensor | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class OneFormerModelOutput(ModelOutput):
    encoder_hidden_states: tuple[torch.FloatTensor] | None = ...
    pixel_decoder_hidden_states: tuple[torch.FloatTensor] | None = ...
    transformer_decoder_hidden_states: torch.FloatTensor | None = ...
    transformer_decoder_object_queries: torch.FloatTensor | None = ...
    transformer_decoder_contrastive_queries: torch.FloatTensor | None = ...
    transformer_decoder_mask_predictions: torch.FloatTensor | None = ...
    transformer_decoder_class_predictions: torch.FloatTensor | None = ...
    transformer_decoder_auxiliary_predictions: (
        tuple[dict[str, torch.FloatTensor]] | None
    ) = ...
    text_queries: torch.FloatTensor | None = ...
    task_token: torch.FloatTensor | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class OneFormerForUniversalSegmentationOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    class_queries_logits: torch.FloatTensor | None = ...
    masks_queries_logits: torch.FloatTensor | None = ...
    auxiliary_predictions: list[dict[str, torch.FloatTensor]] = ...
    encoder_hidden_states: tuple[torch.FloatTensor] | None = ...
    pixel_decoder_hidden_states: list[torch.FloatTensor] | None = ...
    transformer_decoder_hidden_states: torch.FloatTensor | None = ...
    transformer_decoder_object_queries: torch.FloatTensor | None = ...
    transformer_decoder_contrastive_queries: torch.FloatTensor | None = ...
    transformer_decoder_mask_predictions: torch.FloatTensor | None = ...
    transformer_decoder_class_predictions: torch.FloatTensor | None = ...
    transformer_decoder_auxiliary_predictions: (
        list[dict[str, torch.FloatTensor]] | None
    ) = ...
    text_queries: torch.FloatTensor | None = ...
    task_token: torch.FloatTensor | None = ...
    attentions: tuple[tuple[torch.FloatTensor]] | None = ...

class OneFormerPixelDecoderFrozenBatchNorm2d(nn.Module):
    def __init__(self, n) -> None: ...
    def forward(self, x): ...

class OneFormerPixelDecoderEncoderMultiscaleDeformableAttention(nn.Module):
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
        spatial_shapes=...,
        level_start_index=...,
        output_attentions: bool = ...,
    ):  # -> tuple[Any, Tensor]:
        ...

class OneFormerPixelDecoderEncoderLayer(nn.Module):
    def __init__(self, config: OneFormerConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: torch.Tensor | None = ...,
        reference_points=...,
        spatial_shapes=...,
        level_start_index=...,
        output_attentions: bool = ...,
    ):  # -> tuple[Tensor, Any] | tuple[Tensor]:
        ...

class OneFormerPixelDecoderEncoderOnly(nn.Module):
    def __init__(self, config: OneFormerConfig) -> None: ...
    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device): ...
    def forward(
        self,
        inputs_embeds=...,
        attention_mask=...,
        position_embeddings=...,
        spatial_shapes=...,
        level_start_index=...,
        valid_ratios=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
    ):  # -> BaseModelOutput:
        ...

class OneFormerPixelDecoder(nn.Module):
    def __init__(self, config: OneFormerConfig, feature_channels) -> None: ...
    def get_valid_ratio(self, mask, dtype=...):  # -> Tensor:
        ...
    def forward(
        self,
        features,
        encoder_outputs=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
    ):  # -> OneFormerPixelDecoderOutput:
        ...

class OneFormerPixelLevelModule(nn.Module):
    def __init__(self, config: OneFormerConfig) -> None: ...
    def forward(
        self, pixel_values: Tensor, output_hidden_states: bool = ...
    ) -> OneFormerPixelLevelModuleOutput: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> OneFormerPixelLevelModuleOutput: ...

class OneFormerAttention(nn.Module):
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

class OneFormerTransformerDecoderSelfAttentionLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=...,
        activation=...,
        normalize_before=...,
        layer_norm_eps=...,
    ) -> None: ...
    def with_pos_embed(self, tensor, pos: Tensor | None): ...
    def forward_post(
        self,
        output,
        output_mask: Tensor | None = ...,
        output_key_padding_mask: Tensor | None = ...,
        query_pos: Tensor | None = ...,
    ):  # -> tuple[Any, Any]:
        ...
    def forward_pre(
        self,
        output,
        output_mask: Tensor | None = ...,
        output_key_padding_mask: Tensor | None = ...,
        query_pos: Tensor | None = ...,
    ):  # -> tuple[Any, Any]:
        ...
    def forward(
        self,
        output,
        output_mask: Tensor | None = ...,
        output_key_padding_mask: Tensor | None = ...,
        query_pos: Tensor | None = ...,
    ):  # -> tuple[Any, Any]:
        ...

class OneFormerTransformerDecoderCrossAttentionLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=...,
        activation=...,
        normalize_before=...,
        layer_norm_eps=...,
    ) -> None: ...
    def with_pos_embed(self, tensor, pos: Tensor | None): ...
    def forward_post(
        self,
        output,
        memory,
        memory_mask: Tensor | None = ...,
        memory_key_padding_mask: Tensor | None = ...,
        pos: Tensor | None = ...,
        query_pos: Tensor | None = ...,
    ):  # -> tuple[Any, Any]:
        ...
    def forward_pre(
        self,
        output,
        memory,
        memory_mask: Tensor | None = ...,
        memory_key_padding_mask: Tensor | None = ...,
        pos: Tensor | None = ...,
        query_pos: Tensor | None = ...,
    ):  # -> tuple[Any, Any]:
        ...
    def forward(
        self,
        output,
        memory,
        memory_mask: Tensor | None = ...,
        memory_key_padding_mask: Tensor | None = ...,
        pos: Tensor | None = ...,
        query_pos: Tensor | None = ...,
    ):  # -> tuple[Any, Any]:
        ...

class OneFormerTransformerDecoderFFNLayer(nn.Module):
    def __init__(
        self,
        d_model,
        dim_feedforward=...,
        dropout=...,
        activation=...,
        normalize_before=...,
        layer_norm_eps=...,
    ) -> None: ...
    def with_pos_embed(self, tensor, pos: Tensor | None): ...
    def forward_post(self, output):  # -> Any:
        ...
    def forward_pre(self, output): ...
    def forward(self, output):  # -> Any:
        ...

class OneFormerMLPPredictionHead(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = ...
    ) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...

class OneFormerTransformerDecoderLayer(nn.Module):
    def __init__(self, config: OneFormerConfig) -> None: ...
    def forward(
        self,
        index: int,
        output: torch.Tensor,
        multi_stage_features: list[torch.Tensor],
        multi_stage_positional_embeddings: list[torch.Tensor],
        attention_mask: torch.Tensor | None = ...,
        query_embeddings: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
    ):  # -> tuple[Tensor, Any, Any] | tuple[Tensor]:
        ...

class OneFormerTransformerDecoderQueryTransformerDecoder(nn.Module):
    def __init__(
        self, decoder_layer, num_layers, norm=..., return_intermediate=...
    ) -> None: ...
    def forward(
        self,
        output,
        memory,
        output_mask: Tensor | None = ...,
        memory_mask: Tensor | None = ...,
        output_key_padding_mask: Tensor | None = ...,
        memory_key_padding_mask: Tensor | None = ...,
        pos: Tensor | None = ...,
        query_pos: Tensor | None = ...,
    ):  # -> Tensor | Any:
        ...

class OneFormerTransformerDecoderQueryTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=...,
        dropout=...,
        activation=...,
        normalize_before=...,
        layer_norm_eps=...,
    ) -> None: ...
    def with_pos_embed(self, tensor, pos: Tensor | None): ...
    def forward_post(
        self,
        output,
        memory,
        output_mask: Tensor | None = ...,
        memory_mask: Tensor | None = ...,
        output_key_padding_mask: Tensor | None = ...,
        memory_key_padding_mask: Tensor | None = ...,
        pos: Tensor | None = ...,
        query_pos: Tensor | None = ...,
    ):  # -> Any:
        ...
    def forward_pre(
        self,
        output,
        memory,
        output_mask: Tensor | None = ...,
        memory_mask: Tensor | None = ...,
        output_key_padding_mask: Tensor | None = ...,
        memory_key_padding_mask: Tensor | None = ...,
        pos: Tensor | None = ...,
        query_pos: Tensor | None = ...,
    ): ...
    def forward(
        self,
        output,
        memory,
        output_mask: Tensor | None = ...,
        memory_mask: Tensor | None = ...,
        output_key_padding_mask: Tensor | None = ...,
        memory_key_padding_mask: Tensor | None = ...,
        pos: Tensor | None = ...,
        query_pos: Tensor | None = ...,
    ):  # -> Any:
        ...

class OneFormerTransformerDecoderQueryTransformer(nn.Module):
    def __init__(
        self,
        d_model=...,
        nhead=...,
        num_decoder_layers=...,
        dim_feedforward=...,
        dropout=...,
        activation=...,
        normalize_before=...,
        return_intermediate_dec=...,
        layer_norm_eps=...,
    ) -> None: ...
    def forward(self, src, mask, query_embed, pos_embed, task_token=...):  # -> Any:
        ...

class OneFormerTransformerDecoder(nn.Module):
    def __init__(self, in_channels: int, config: OneFormerConfig) -> None: ...
    def forward(
        self,
        task_token=...,
        multi_stage_features=...,
        multi_stage_positional_embeddings=...,
        mask_features=...,
        query_features=...,
        query_embeddings=...,
        query_embedder=...,
        size_list=...,
        output_attentions=...,
    ):  # -> OneFormerTransformerDecoderOutput:
        ...
    def forward_prediction_heads(
        self, output, mask_features, attention_mask_target_size
    ):  # -> tuple[Any, Tensor, Tensor]:
        ...

class OneFormerTransformerModule(nn.Module):
    def __init__(self, in_features: int, config: OneFormerConfig) -> None: ...
    def forward(
        self,
        multi_scale_features: list[Tensor],
        mask_features: Tensor,
        task_token: Tensor,
        output_attentions: bool = ...,
    ) -> OneFormerTransformerDecoderOutput: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> OneFormerTransformerDecoderOutput: ...

class OneFormerSinePositionEmbedding(nn.Module):
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

class OneFormerTextMapperAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=...,
        qkv_bias=...,
        qk_scale=...,
        attn_drop=...,
        proj_drop=...,
    ) -> None: ...
    def forward(self, q, k, v):  # -> Any:
        ...

class OneFormerTextTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=..., layer_norm_eps=...) -> None: ...
    def forward(self, hidden_state, mem): ...

class OneFormerTextContextDecoder(nn.Module):
    def __init__(
        self,
        transformer_width=...,
        transformer_heads=...,
        transformer_layers=...,
        visual_dim=...,
        dropout=...,
        layer_norm_eps=...,
        **kwargs,
    ) -> None: ...
    def forward(self, text, visual):  # -> Any:
        ...

class OneFormerTextMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int | None = ...,
        intermediate_size: int | None = ...,
        output_size: int | None = ...,
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class OneFormerTextTransformerLayer(GradientCheckpointingLayer):
    def __init__(
        self, width: int, heads: int, attn_mask: torch.Tensor, layer_norm_eps=...
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_padding_mask: torch.Tensor | None = ...,
    ) -> torch.FloatTensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.FloatTensor: ...

class OneFormerTextTransformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        attn_mask: torch.Tensor | None = ...,
        use_checkpoint=...,
        layer_norm_eps=...,
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor):  # -> Tensor:
        ...

class OneFormerTextEncoder(nn.Module):
    def __init__(
        self,
        context_length: int,
        width: int,
        layers: int,
        vocab_size,
        use_checkpoint=...,
        layer_norm_eps=...,
    ) -> None: ...
    def build_attention_mask(self):  # -> Tensor:
        ...
    def forward(self, text):  # -> Any:
        ...

class OneFormerTextMapper(nn.Module):
    def __init__(self, config: OneFormerConfig) -> None: ...
    def forward(self, inputs: Tensor) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...
    def encode_text(self, text):  # -> Tensor | Any:
        ...

class OneFormerTaskModel(nn.Module):
    def __init__(self, config: OneFormerConfig) -> None: ...
    def forward(self, inputs: Tensor) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...

@auto_docstring
class OneFormerPreTrainedModel(PreTrainedModel):
    config: OneFormerConfig
    base_model_prefix = ...
    main_input_name = ...

@auto_docstring
class OneFormerModel(OneFormerPreTrainedModel):
    main_input_name = ...
    def __init__(self, config: OneFormerConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: Tensor,
        task_inputs: Tensor,
        text_inputs: Tensor | None = ...,
        pixel_mask: Tensor | None = ...,
        output_hidden_states: bool | None = ...,
        output_attentions: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> OneFormerModelOutput: ...

@auto_docstring(custom_intro=...)
class OneFormerForUniversalSegmentation(OneFormerPreTrainedModel):
    main_input_name = ...
    def __init__(self, config: OneFormerConfig) -> None: ...
    def get_loss_dict(
        self,
        masks_queries_logits: Tensor,
        class_queries_logits: Tensor,
        contrastive_queries_logits: Tensor,
        mask_labels: Tensor,
        class_labels: Tensor,
        text_queries: Tensor,
        auxiliary_predictions: dict[str, Tensor],
        calculate_contrastive_loss: bool,
    ) -> dict[str, Tensor]: ...
    def get_loss(self, loss_dict: dict[str, Tensor]) -> Tensor: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: Tensor,
        task_inputs: Tensor,
        text_inputs: Tensor | None = ...,
        mask_labels: list[Tensor] | None = ...,
        class_labels: list[Tensor] | None = ...,
        pixel_mask: Tensor | None = ...,
        output_auxiliary_logits: bool | None = ...,
        output_hidden_states: bool | None = ...,
        output_attentions: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> OneFormerForUniversalSegmentationOutput: ...

__all__ = [
    "OneFormerForUniversalSegmentation",
    "OneFormerModel",
    "OneFormerPreTrainedModel",
]
