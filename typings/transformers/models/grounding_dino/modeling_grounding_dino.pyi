from typing import Any
from dataclasses import dataclass

from torch import Tensor, nn

import torch

from .configuration_grounding_dino import GroundingDinoConfig
from ...file_utils import ModelOutput
from ...integrations import use_kernel_forward_from_hub
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring

"""PyTorch Grounding DINO model."""
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
class GroundingDinoDecoderOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = ...
    intermediate_hidden_states: torch.FloatTensor | None = ...
    intermediate_reference_points: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[tuple[torch.FloatTensor]] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class GroundingDinoEncoderOutput(ModelOutput):
    last_hidden_state_vision: torch.FloatTensor | None = ...
    last_hidden_state_text: torch.FloatTensor | None = ...
    vision_hidden_states: tuple[torch.FloatTensor] | None = ...
    text_hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[tuple[torch.FloatTensor]] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class GroundingDinoModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = ...
    init_reference_points: torch.FloatTensor | None = ...
    intermediate_hidden_states: torch.FloatTensor | None = ...
    intermediate_reference_points: torch.FloatTensor | None = ...
    decoder_hidden_states: tuple[torch.FloatTensor] | None = ...
    decoder_attentions: tuple[tuple[torch.FloatTensor]] | None = ...
    encoder_last_hidden_state_vision: torch.FloatTensor | None = ...
    encoder_last_hidden_state_text: torch.FloatTensor | None = ...
    encoder_vision_hidden_states: tuple[torch.FloatTensor] | None = ...
    encoder_text_hidden_states: tuple[torch.FloatTensor] | None = ...
    encoder_attentions: tuple[tuple[torch.FloatTensor]] | None = ...
    enc_outputs_class: torch.FloatTensor | None = ...
    enc_outputs_coord_logits: torch.FloatTensor | None = ...
    encoder_logits: torch.FloatTensor | None = ...
    encoder_pred_boxes: torch.FloatTensor | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class GroundingDinoObjectDetectionOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    loss_dict: dict | None = ...
    logits: torch.FloatTensor | None = ...
    pred_boxes: torch.FloatTensor | None = ...
    auxiliary_outputs: list[dict] | None = ...
    last_hidden_state: torch.FloatTensor | None = ...
    init_reference_points: torch.FloatTensor | None = ...
    intermediate_hidden_states: torch.FloatTensor | None = ...
    intermediate_reference_points: torch.FloatTensor | None = ...
    decoder_hidden_states: tuple[torch.FloatTensor] | None = ...
    decoder_attentions: tuple[tuple[torch.FloatTensor]] | None = ...
    encoder_last_hidden_state_vision: torch.FloatTensor | None = ...
    encoder_last_hidden_state_text: torch.FloatTensor | None = ...
    encoder_vision_hidden_states: tuple[torch.FloatTensor] | None = ...
    encoder_text_hidden_states: tuple[torch.FloatTensor] | None = ...
    encoder_attentions: tuple[tuple[torch.FloatTensor]] | None = ...
    enc_outputs_class: torch.FloatTensor | None = ...
    enc_outputs_coord_logits: torch.FloatTensor | None = ...
    encoder_logits: torch.FloatTensor | None = ...
    encoder_pred_boxes: torch.FloatTensor | None = ...
    input_ids: torch.LongTensor | None = ...

class GroundingDinoFrozenBatchNorm2d(nn.Module):
    def __init__(self, n) -> None: ...
    def forward(self, x): ...

def replace_batch_norm(model):  # -> None:
    ...

class GroundingDinoConvEncoder(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor
    ):  # -> list[Any]:
        ...

class GroundingDinoConvModel(nn.Module):
    def __init__(self, conv_encoder, position_embedding) -> None: ...
    def forward(self, pixel_values, pixel_mask):  # -> tuple[Any, list[Any]]:
        ...

class GroundingDinoSinePositionEmbedding(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, pixel_values, pixel_mask):  # -> Tensor:
        ...

class GroundingDinoLearnedPositionEmbedding(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, pixel_values, pixel_mask=...):  # -> Tensor:
        ...

def build_position_encoding(
    config,
):  # -> GroundingDinoSinePositionEmbedding | GroundingDinoLearnedPositionEmbedding:
    ...

class GroundingDinoMultiscaleDeformableAttention(nn.Module):
    def __init__(
        self, config: GroundingDinoConfig, num_heads: int, n_points: int
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

class GroundingDinoTextEnhancerLayer(nn.Module):
    def __init__(self, config) -> None: ...
    def with_pos_embed(
        self, hidden_state: Tensor, position_embeddings: Tensor | None
    ):  # -> Tensor:
        ...
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_masks: torch.BoolTensor | None = ...,
        position_embeddings: torch.FloatTensor | None = ...,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]: ...

class GroundingDinoBiMultiHeadAttention(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        vision_features: torch.FloatTensor,
        text_features: torch.FloatTensor,
        vision_attention_mask: torch.BoolTensor | None = ...,
        text_attention_mask: torch.BoolTensor | None = ...,
    ) -> tuple[
        tuple[torch.FloatTensor, torch.FloatTensor],
        tuple[torch.FloatTensor, torch.FloatTensor],
    ]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[
        tuple[torch.FloatTensor, torch.FloatTensor],
        tuple[torch.FloatTensor, torch.FloatTensor],
    ]: ...

def drop_path(
    input: torch.Tensor, drop_prob: float = ..., training: bool = ...
) -> torch.Tensor: ...

class GroundingDinoDropPath(nn.Module):
    def __init__(self, drop_prob: float | None = ...) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...
    def extra_repr(self) -> str: ...

class GroundingDinoFusionLayer(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        vision_features: torch.FloatTensor,
        text_features: torch.FloatTensor,
        attention_mask_vision: torch.BoolTensor | None = ...,
        attention_mask_text: torch.BoolTensor | None = ...,
    ) -> tuple[
        tuple[torch.FloatTensor, torch.FloatTensor],
        tuple[torch.FloatTensor, torch.FloatTensor],
    ]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[
        tuple[torch.FloatTensor, torch.FloatTensor],
        tuple[torch.FloatTensor, torch.FloatTensor],
    ]: ...

class GroundingDinoDeformableLayer(nn.Module):
    def __init__(self, config: GroundingDinoConfig) -> None: ...
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
    ):  # -> tuple[Tensor, Any]:
        ...

def get_sine_pos_embed(
    pos_tensor: torch.Tensor,
    num_pos_feats: int = ...,
    temperature: int = ...,
    exchange_xy: bool = ...,
) -> Tensor: ...

class GroundingDinoEncoderLayer(nn.Module):
    def __init__(self, config) -> None: ...
    def get_text_position_embeddings(
        self,
        text_features: Tensor,
        text_position_embedding: torch.Tensor | None,
        text_position_ids: torch.Tensor | None,
    ) -> Tensor: ...
    def forward(
        self,
        vision_features: Tensor,
        vision_position_embedding: Tensor,
        spatial_shapes: Tensor,
        spatial_shapes_list: list[tuple[int, int]],
        level_start_index: Tensor,
        key_padding_mask: Tensor,
        reference_points: Tensor,
        text_features: Tensor | None = ...,
        text_attention_mask: Tensor | None = ...,
        text_position_embedding: Tensor | None = ...,
        text_self_attention_masks: Tensor | None = ...,
        text_position_ids: Tensor | None = ...,
    ):  # -> tuple[tuple[Tensor, Tensor | None], tuple[Any, Any, Any, Any]]:
        ...

class GroundingDinoMultiheadAttention(nn.Module):
    def __init__(self, config, num_attention_heads=...) -> None: ...
    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.Tensor]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor]: ...

class GroundingDinoDecoderLayer(nn.Module):
    def __init__(self, config: GroundingDinoConfig) -> None: ...
    def with_pos_embed(
        self, tensor: torch.Tensor, position_embeddings: Tensor | None
    ):  # -> Tensor:
        ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor | None = ...,
        reference_points=...,
        spatial_shapes=...,
        spatial_shapes_list=...,
        level_start_index=...,
        vision_encoder_hidden_states: torch.Tensor | None = ...,
        vision_encoder_attention_mask: torch.Tensor | None = ...,
        text_encoder_hidden_states: torch.Tensor | None = ...,
        text_encoder_attention_mask: torch.Tensor | None = ...,
        self_attn_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
    ):  # -> tuple[Tensor, Any, Any, Any] | tuple[Tensor]:
        ...

class GroundingDinoContrastiveEmbedding(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        vision_hidden_state: torch.FloatTensor,
        text_hidden_state: torch.FloatTensor,
        text_token_mask: torch.BoolTensor,
    ) -> torch.FloatTensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.FloatTensor: ...

@auto_docstring
class GroundingDinoPreTrainedModel(PreTrainedModel):
    config: GroundingDinoConfig
    base_model_prefix = ...
    main_input_name = ...

class GroundingDinoEncoder(GroundingDinoPreTrainedModel):
    def __init__(self, config: GroundingDinoConfig) -> None: ...
    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device): ...
    def forward(
        self,
        vision_features: Tensor,
        vision_attention_mask: Tensor,
        vision_position_embedding: Tensor,
        spatial_shapes: Tensor,
        spatial_shapes_list: list[tuple[int, int]],
        level_start_index: Tensor,
        valid_ratios=...,
        text_features: Tensor | None = ...,
        text_attention_mask: Tensor | None = ...,
        text_position_embedding: Tensor | None = ...,
        text_self_attention_masks: Tensor | None = ...,
        text_position_ids: Tensor | None = ...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
    ):  # -> tuple[Tensor | tuple[Tensor, ...] | Any | tuple[()] | tuple[Tensor | None, ...] | tuple[tuple[()] | tuple[Any, ...] | None, tuple[()] | tuple[Any, ...] | None, tuple[()] | tuple[Any, ...] | None, tuple[()] | tuple[Any, ...] | None], ...] | GroundingDinoEncoderOutput:
        ...

class GroundingDinoDecoder(GroundingDinoPreTrainedModel):
    def __init__(self, config: GroundingDinoConfig) -> None: ...
    def forward(
        self,
        inputs_embeds,
        vision_encoder_hidden_states,
        vision_encoder_attention_mask=...,
        text_encoder_hidden_states=...,
        text_encoder_attention_mask=...,
        reference_points=...,
        spatial_shapes=...,
        spatial_shapes_list=...,
        level_start_index=...,
        valid_ratios=...,
        self_attn_mask=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
    ):  # -> tuple[Any | Tensor | tuple[Any, ...] | tuple[()] | tuple[tuple[()] | tuple[Any, ...] | None, ...], ...] | GroundingDinoDecoderOutput:
        ...

SPECIAL_TOKENS = ...

def generate_masks_with_special_tokens_and_transfer_map(
    input_ids: torch.LongTensor,
) -> tuple[Tensor, Tensor]: ...

@auto_docstring(custom_intro=...)
class GroundingDinoModel(GroundingDinoPreTrainedModel):
    def __init__(self, config: GroundingDinoConfig) -> None: ...
    def freeze_backbone(self):  # -> None:
        ...
    def unfreeze_backbone(self):  # -> None:
        ...
    def get_valid_ratio(self, mask):  # -> Tensor:
        ...
    def generate_encoder_output_proposals(
        self, enc_output, padding_mask, spatial_shapes
    ):  # -> tuple[Any, Tensor]:
        ...
    @auto_docstring
    def forward(
        self,
        pixel_values: Tensor,
        input_ids: Tensor,
        token_type_ids: Tensor | None = ...,
        attention_mask: Tensor | None = ...,
        pixel_mask: Tensor | None = ...,
        encoder_outputs=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
    ):  # -> GroundingDinoModelOutput:
        ...

class GroundingDinoMLPPredictionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers) -> None: ...
    def forward(self, x):  # -> Tensor | Any:
        ...

def build_label_maps(
    logits: torch.FloatTensor, input_ids: torch.LongTensor
) -> tuple[torch.FloatTensor]: ...
def build_text_mask(logits, attention_mask):  # -> Tensor:
    ...

@auto_docstring(custom_intro=...)
class GroundingDinoForObjectDetection(GroundingDinoPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config: GroundingDinoConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        token_type_ids: torch.LongTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        pixel_mask: torch.BoolTensor | None = ...,
        encoder_outputs: GroundingDinoEncoderOutput | tuple | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: list[dict[str, torch.LongTensor | torch.FloatTensor]] | None = ...,
    ):  # -> tuple[Any | int | Tensor | dict[str, Any] | LongTensor, ...] | GroundingDinoObjectDetectionOutput:
        ...

__all__ = [
    "GroundingDinoForObjectDetection",
    "GroundingDinoModel",
    "GroundingDinoPreTrainedModel",
]
