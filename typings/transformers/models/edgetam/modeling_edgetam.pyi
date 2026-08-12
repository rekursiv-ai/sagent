from typing import Any
from dataclasses import dataclass

from torch import Tensor, nn
from transformers.utils.generic import TransformersKwargs, check_model_inputs

import torch

from .configuration_edgetam import (
    EdgeTamConfig,
    EdgeTamMaskDecoderConfig,
    EdgeTamPromptEncoderConfig,
    EdgeTamVisionConfig,
)
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...pytorch_utils import compile_compatible_method_lru_cache
from ...utils import ModelOutput, auto_docstring

class EdgeTamLayerNorm(nn.LayerNorm):
    def __init__(
        self, normalized_shape, *, eps=..., data_format=..., **kwargs
    ) -> None: ...
    def forward(self, features: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

@dataclass
@auto_docstring(custom_intro="Base class for the vision encoder's outputs.")
class EdgeTamVisionEncoderOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = ...
    fpn_hidden_states: torch.FloatTensor | None = ...
    fpn_position_encoding: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = ...,
    **kwargs,
):  # -> tuple[Tensor, Tensor]:
    ...

class EdgeTamAttention(nn.Module):
    def __init__(self, config, downsample_rate=...) -> None: ...
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_similarity: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

class EdgeTamTwoWayAttentionBlock(nn.Module):
    def __init__(
        self, config: EdgeTamMaskDecoderConfig, skip_first_layer_pe: bool = ...
    ) -> None: ...
    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        query_point_embedding: Tensor,
        key_point_embedding: Tensor,
        attention_similarity: Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ):  # -> tuple[Tensor, Tensor, Any]:
        ...

class EdgeTamFeedForward(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: str = ...,
        sigmoid_output: bool = ...,
    ) -> None: ...
    def forward(self, hidden_states):  # -> Tensor | Any:
        ...

@auto_docstring
class EdgeTamPreTrainedModel(PreTrainedModel):
    config_class = EdgeTamConfig
    base_model_prefix = ...
    main_input_name = ...
    _supports_sdpa = ...
    _supports_flash_attn_2 = ...
    _supports_attention_backend = ...

class EdgeTamSinePositionEmbedding(nn.Module):
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

class EdgeTamVisionNeck(nn.Module):
    def __init__(self, config: EdgeTamVisionConfig) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]: ...

@auto_docstring(custom_intro=...)
class EdgeTamVisionModel(EdgeTamPreTrainedModel):
    config_class = EdgeTamVisionConfig
    main_input_name = ...
    _can_record_outputs = ...
    def __init__(self, config: EdgeTamVisionConfig) -> None: ...
    @check_model_inputs
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | EdgeTamVisionEncoderOutput: ...

@dataclass
@auto_docstring(custom_intro="Base class for the EdgeTam model's output.")
class EdgeTamImageSegmentationOutput(ModelOutput):
    iou_scores: torch.FloatTensor | None = ...
    pred_masks: torch.FloatTensor | None = ...
    object_score_logits: torch.FloatTensor | None = ...
    image_embeddings: tuple[torch.FloatTensor, ...] = ...
    vision_hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    vision_attentions: tuple[torch.FloatTensor, ...] | None = ...
    mask_decoder_attentions: tuple[torch.FloatTensor, ...] | None = ...

class EdgeTamPositionalEmbedding(nn.Module):
    def __init__(self, config: EdgeTamPromptEncoderConfig) -> None: ...
    def forward(self, input_coords, input_shape=...):  # -> Tensor:
        ...

class EdgeTamMaskEmbedding(nn.Module):
    def __init__(self, config: EdgeTamPromptEncoderConfig) -> None: ...
    def forward(self, masks):  # -> Any:
        ...

class EdgeTamPromptEncoder(nn.Module):
    def __init__(self, config: EdgeTamPromptEncoderConfig) -> None: ...
    def forward(
        self,
        input_points: tuple[torch.Tensor, torch.Tensor] | None,
        input_labels: torch.Tensor | None,
        input_boxes: torch.Tensor | None,
        input_masks: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

class EdgeTamTwoWayTransformer(nn.Module):
    def __init__(self, config: EdgeTamMaskDecoderConfig) -> None: ...
    def forward(
        self,
        point_embeddings: Tensor,
        image_embeddings: Tensor,
        image_positional_embeddings: Tensor,
        attention_similarity: Tensor,
        target_embedding=...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutput: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple | BaseModelOutput: ...

class EdgeTamMaskDecoder(nn.Module):
    def __init__(self, config: EdgeTamMaskDecoderConfig) -> None: ...
    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_positional_embeddings: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        high_resolution_features: list[torch.Tensor],
        attention_similarity: torch.Tensor | None = ...,
        target_embedding: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...

@auto_docstring(custom_intro=...)
class EdgeTamModel(EdgeTamPreTrainedModel):
    _tied_weights_keys = ...
    _keys_to_ignore_on_load_missing = ...
    _can_record_outputs = ...
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config: EdgeTamConfig) -> None: ...
    def get_image_wide_positional_embeddings(self) -> torch.Tensor: ...
    @torch.no_grad()
    def get_image_embeddings(
        self, pixel_values: torch.FloatTensor, **kwargs: Unpack[TransformersKwargs]
    ) -> list[torch.Tensor]: ...
    @torch.no_grad()
    def get_prompt_embeddings(
        self,
        input_points: torch.FloatTensor | None = ...,
        input_labels: torch.LongTensor | None = ...,
        input_boxes: torch.FloatTensor | None = ...,
        input_masks: torch.LongTensor | None = ...,
    ):  # -> Any:
        ...
    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        input_points: torch.FloatTensor | None = ...,
        input_labels: torch.LongTensor | None = ...,
        input_boxes: torch.FloatTensor | None = ...,
        input_masks: torch.LongTensor | None = ...,
        image_embeddings: torch.FloatTensor | None = ...,
        multimask_output: bool = ...,
        attention_similarity: torch.FloatTensor | None = ...,
        target_embedding: torch.FloatTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> EdgeTamImageSegmentationOutput: ...
    def get_image_features(
        self, pixel_values: torch.FloatTensor, **kwargs: Unpack[TransformersKwargs]
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        tuple[torch.FloatTensor, ...] | None,
        tuple[torch.FloatTensor, ...] | None,
    ]: ...

__all__ = ["EdgeTamModel", "EdgeTamPreTrainedModel", "EdgeTamVisionModel"]
