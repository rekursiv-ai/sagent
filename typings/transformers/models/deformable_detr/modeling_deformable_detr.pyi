from dataclasses import dataclass
from typing import Any

from torch import Tensor, nn

import torch

from .configuration_deformable_detr import DeformableDetrConfig
from ...integrations import use_kernel_forward_from_hub
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring

"""PyTorch Deformable DETR model."""
logger = ...
logger = ...

@use_kernel_forward_from_hub("MultiScaleDeformableAttention")
class MultiScaleDeformableAttention(nn.Module):
    def forward(
        self,
        value: Tensor,
        value_spatial_shapes: Tensor,
        value_spatial_shapes_list: list[tuple],
        level_start_index: Tensor,
        sampling_locations: Tensor,
        attention_weights: Tensor,
        im2col_step: int,
    ):  # -> Tensor:
        ...

@dataclass
@auto_docstring(custom_intro=...)
class DeformableDetrDecoderOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = ...
    intermediate_hidden_states: torch.FloatTensor | None = ...
    intermediate_reference_points: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...
    cross_attentions: tuple[torch.FloatTensor] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class DeformableDetrModelOutput(ModelOutput):
    init_reference_points: torch.FloatTensor | None = ...
    last_hidden_state: torch.FloatTensor | None = ...
    intermediate_hidden_states: torch.FloatTensor | None = ...
    intermediate_reference_points: torch.FloatTensor | None = ...
    decoder_hidden_states: tuple[torch.FloatTensor] | None = ...
    decoder_attentions: tuple[torch.FloatTensor] | None = ...
    cross_attentions: tuple[torch.FloatTensor] | None = ...
    encoder_last_hidden_state: torch.FloatTensor | None = ...
    encoder_hidden_states: tuple[torch.FloatTensor] | None = ...
    encoder_attentions: tuple[torch.FloatTensor] | None = ...
    enc_outputs_class: torch.FloatTensor | None = ...
    enc_outputs_coord_logits: torch.FloatTensor | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class DeformableDetrObjectDetectionOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    loss_dict: dict | None = ...
    logits: torch.FloatTensor | None = ...
    pred_boxes: torch.FloatTensor | None = ...
    auxiliary_outputs: list[dict] | None = ...
    init_reference_points: torch.FloatTensor | None = ...
    last_hidden_state: torch.FloatTensor | None = ...
    intermediate_hidden_states: torch.FloatTensor | None = ...
    intermediate_reference_points: torch.FloatTensor | None = ...
    decoder_hidden_states: tuple[torch.FloatTensor] | None = ...
    decoder_attentions: tuple[torch.FloatTensor] | None = ...
    cross_attentions: tuple[torch.FloatTensor] | None = ...
    encoder_last_hidden_state: torch.FloatTensor | None = ...
    encoder_hidden_states: tuple[torch.FloatTensor] | None = ...
    encoder_attentions: tuple[torch.FloatTensor] | None = ...
    enc_outputs_class: Any = ...
    enc_outputs_coord_logits: torch.FloatTensor | None = ...

def inverse_sigmoid(x, eps=...):  # -> Tensor:
    ...

class DeformableDetrFrozenBatchNorm2d(nn.Module):
    def __init__(self, n) -> None: ...
    def forward(self, x): ...

def replace_batch_norm(model):  # -> None:
    ...

class DeformableDetrConvEncoder(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor
    ):  # -> list[Any]:
        ...

class DeformableDetrConvModel(nn.Module):
    def __init__(self, conv_encoder, position_embedding) -> None: ...
    def forward(self, pixel_values, pixel_mask):  # -> tuple[Any, list[Any]]:
        ...

class DeformableDetrSinePositionEmbedding(nn.Module):
    def __init__(
        self, embedding_dim=..., temperature=..., normalize=..., scale=...
    ) -> None: ...
    def forward(self, pixel_values, pixel_mask):  # -> Tensor:
        ...

class DeformableDetrLearnedPositionEmbedding(nn.Module):
    def __init__(self, embedding_dim=...) -> None: ...
    def forward(self, pixel_values, pixel_mask=...):  # -> Tensor:
        ...

def build_position_encoding(
    config,
):  # -> DeformableDetrSinePositionEmbedding | DeformableDetrLearnedPositionEmbedding:
    ...

class DeformableDetrMultiscaleDeformableAttention(nn.Module):
    def __init__(
        self, config: DeformableDetrConfig, num_heads: int, n_points: int
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
        spatial_shapes_list=...,
        level_start_index=...,
        output_attentions: bool = ...,
    ):  # -> tuple[Any, Tensor]:
        ...

class DeformableDetrMultiheadAttention(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, dropout: float = ..., bias: bool = ...
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
        output_attentions: bool = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class DeformableDetrEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: DeformableDetrConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: torch.Tensor | None = ...,
        reference_points=...,
        spatial_shapes=...,
        spatial_shapes_list=...,
        level_start_index=...,
        output_attentions: bool = ...,
    ):  # -> tuple[Tensor, Any] | tuple[Tensor]:
        ...

class DeformableDetrDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: DeformableDetrConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor | None = ...,
        reference_points=...,
        spatial_shapes=...,
        spatial_shapes_list=...,
        level_start_index=...,
        encoder_hidden_states: torch.Tensor | None = ...,
        encoder_attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
    ):  # -> tuple[Tensor, Any, Any] | tuple[Tensor]:
        ...

@auto_docstring
class DeformableDetrPreTrainedModel(PreTrainedModel):
    config: DeformableDetrConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...

class DeformableDetrEncoder(DeformableDetrPreTrainedModel):
    def __init__(self, config: DeformableDetrConfig) -> None: ...
    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device): ...
    def forward(
        self,
        inputs_embeds=...,
        attention_mask=...,
        position_embeddings=...,
        spatial_shapes=...,
        spatial_shapes_list=...,
        level_start_index=...,
        valid_ratios=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
    ):  # -> tuple[Tensor | Any | tuple[Tensor | Any, ...] | tuple[()] | tuple[Any, ...], ...] | BaseModelOutput:
        ...

class DeformableDetrDecoder(DeformableDetrPreTrainedModel):
    def __init__(self, config: DeformableDetrConfig) -> None: ...
    def forward(
        self,
        inputs_embeds=...,
        encoder_hidden_states=...,
        encoder_attention_mask=...,
        position_embeddings=...,
        reference_points=...,
        spatial_shapes=...,
        spatial_shapes_list=...,
        level_start_index=...,
        valid_ratios=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
    ):  # -> tuple[Any | Tensor | tuple[Any, ...] | tuple[()], ...] | DeformableDetrDecoderOutput:
        ...

@auto_docstring(custom_intro=...)
class DeformableDetrModel(DeformableDetrPreTrainedModel):
    def __init__(self, config: DeformableDetrConfig) -> None: ...
    def get_encoder(self):  # -> DeformableDetrEncoder:
        ...
    def freeze_backbone(self):  # -> None:
        ...
    def unfreeze_backbone(self):  # -> None:
        ...
    def get_valid_ratio(self, mask, dtype=...):  # -> Tensor:
        ...
    def get_proposal_pos_embed(self, proposals):  # -> Tensor:
        ...
    def gen_encoder_output_proposals(
        self, enc_output, padding_mask, spatial_shapes
    ):  # -> tuple[Any, Tensor]:
        ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: torch.LongTensor | None = ...,
        decoder_attention_mask: torch.FloatTensor | None = ...,
        encoder_outputs: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        decoder_inputs_embeds: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.FloatTensor] | DeformableDetrModelOutput: ...

class DeformableDetrMLPPredictionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers) -> None: ...
    def forward(self, x):  # -> Tensor | Any:
        ...

@auto_docstring(custom_intro=...)
class DeformableDetrForObjectDetection(DeformableDetrPreTrainedModel):
    _tied_weights_keys = ...
    _no_split_modules = ...
    def __init__(self, config: DeformableDetrConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: torch.LongTensor | None = ...,
        decoder_attention_mask: torch.FloatTensor | None = ...,
        encoder_outputs: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        decoder_inputs_embeds: torch.FloatTensor | None = ...,
        labels: list[dict] | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.FloatTensor] | DeformableDetrObjectDetectionOutput: ...

__all__ = [
    "DeformableDetrForObjectDetection",
    "DeformableDetrModel",
    "DeformableDetrPreTrainedModel",
]
