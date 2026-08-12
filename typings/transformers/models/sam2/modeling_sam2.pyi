from typing import Any
from dataclasses import dataclass

from torch import Tensor, nn

import torch

from .configuration_sam2 import (
    Sam2Config,
    Sam2HieraDetConfig,
    Sam2MaskDecoderConfig,
    Sam2PromptEncoderConfig,
    Sam2VisionConfig,
)
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...pytorch_utils import compile_compatible_method_lru_cache
from ...utils import ModelOutput, auto_docstring
from ...utils.generic import TransformersKwargs, check_model_inputs

@dataclass
@auto_docstring(custom_intro="Base class for the vision encoder's outputs.")
class Sam2VisionEncoderOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = ...
    fpn_hidden_states: torch.FloatTensor | None = ...
    fpn_position_encoding: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...

@dataclass
@auto_docstring(custom_intro="Base class for the Sam2 model's output.")
class Sam2ImageSegmentationOutput(ModelOutput):
    iou_scores: torch.FloatTensor | None = ...
    pred_masks: torch.FloatTensor | None = ...
    object_score_logits: torch.FloatTensor | None = ...
    image_embeddings: tuple[torch.FloatTensor, ...] = ...
    vision_hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    vision_attentions: tuple[torch.FloatTensor, ...] | None = ...
    mask_decoder_attentions: tuple[torch.FloatTensor, ...] | None = ...

class Sam2PatchEmbeddings(nn.Module):
    def __init__(self, config: Sam2HieraDetConfig) -> None: ...
    def forward(self, pixel_values):  # -> Any:
        ...

class Sam2SinePositionEmbedding(nn.Module):
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

class Sam2VisionNeck(nn.Module):
    def __init__(self, config: Sam2VisionConfig) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]: ...

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
def do_pool(x: torch.Tensor, query_stride: int | None = ...) -> torch.Tensor: ...

class Sam2MultiScaleAttention(nn.Module):
    def __init__(
        self,
        config: Sam2HieraDetConfig,
        dim: int,
        dim_out: int,
        num_attention_heads: int,
        query_stride: tuple[int, int] | None = ...,
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class Sam2FeedForward(nn.Module):
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

def window_partition(hidden_state, window_size):  # -> tuple[Tensor, tuple[Any, Any]]:
    ...
def window_unpartition(windows, window_size, pad_height_width, height_width): ...

class Sam2MultiScaleBlock(GradientCheckpointingLayer):
    def __init__(
        self,
        config: Sam2HieraDetConfig,
        stage_idx: int,
        block_idx: int,
        total_block_idx: int,
    ) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, **kwargs: Unpack[TransformersKwargs]
    ) -> torch.FloatTensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.FloatTensor: ...

@dataclass
@auto_docstring(custom_intro=...)
class Sam2HieraDetModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = ...
    intermediate_hidden_states: tuple[torch.FloatTensor, ...] | None = ...

@auto_docstring
class Sam2PreTrainedModel(PreTrainedModel):
    config_class = Sam2Config
    base_model_prefix = ...
    main_input_name = ...
    _supports_sdpa = ...
    _supports_flash_attn_2 = ...
    _supports_attention_backend = ...

class Sam2HieraDetModel(Sam2PreTrainedModel):
    config_class = Sam2HieraDetConfig
    main_input_name = ...
    _can_record_outputs = ...
    def __init__(self, config: Sam2HieraDetConfig) -> None: ...
    def get_input_embeddings(self):  # -> Sam2PatchEmbeddings:
        ...
    @check_model_inputs
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Sam2HieraDetModelOutput: ...

@auto_docstring(custom_intro=...)
class Sam2VisionModel(Sam2PreTrainedModel):
    config_class = Sam2VisionConfig
    main_input_name = ...
    _can_record_outputs = ...
    def __init__(self, config: Sam2VisionConfig) -> None: ...
    def get_input_embeddings(self):  # -> Any:
        ...
    @check_model_inputs
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Sam2VisionEncoderOutput: ...

class Sam2PositionalEmbedding(nn.Module):
    def __init__(self, config: Sam2PromptEncoderConfig) -> None: ...
    def forward(self, input_coords, input_shape=...):  # -> Tensor:
        ...

class Sam2MaskEmbedding(nn.Module):
    def __init__(self, config: Sam2PromptEncoderConfig) -> None: ...
    def forward(self, masks):  # -> Any:
        ...

class Sam2PromptEncoder(nn.Module):
    def __init__(self, config: Sam2PromptEncoderConfig) -> None: ...
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

class Sam2Attention(nn.Module):
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

class Sam2TwoWayAttentionBlock(nn.Module):
    def __init__(
        self, config: Sam2MaskDecoderConfig, skip_first_layer_pe: bool = ...
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

class Sam2TwoWayTransformer(nn.Module):
    def __init__(self, config: Sam2MaskDecoderConfig) -> None: ...
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

class Sam2LayerNorm(nn.LayerNorm):
    def __init__(
        self, normalized_shape, *, eps=..., data_format=..., **kwargs
    ) -> None: ...
    def forward(self, features: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class Sam2MaskDecoder(nn.Module):
    def __init__(self, config: Sam2MaskDecoderConfig) -> None: ...
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
class Sam2Model(Sam2PreTrainedModel):
    _tied_weights_keys = ...
    _keys_to_ignore_on_load_missing = ...
    _can_record_outputs = ...
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config: Sam2Config) -> None: ...
    def get_input_embeddings(self):  # -> Any:
        ...
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
    ) -> Sam2ImageSegmentationOutput: ...
    def get_image_features(
        self, pixel_values: torch.FloatTensor, **kwargs: Unpack[TransformersKwargs]
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        tuple[torch.FloatTensor, ...] | None,
        tuple[torch.FloatTensor, ...] | None,
    ]: ...

__all__ = ["Sam2HieraDetModel", "Sam2Model", "Sam2PreTrainedModel", "Sam2VisionModel"]
