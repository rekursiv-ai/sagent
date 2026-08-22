from dataclasses import dataclass
from typing import Any

from torch import Tensor, nn
from transformers.modeling_outputs import ModelOutput
from transformers.utils.generic import TransformersKwargs, check_model_inputs

import torch

from .configuration_sam_hq import (
    SamHQConfig,
    SamHQMaskDecoderConfig,
    SamHQPromptEncoderConfig,
    SamHQVisionConfig,
)
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import auto_docstring

logger = ...

@dataclass
@auto_docstring(custom_intro=...)
class SamHQVisionEncoderOutput(ModelOutput):
    image_embeds: torch.FloatTensor | None = ...
    last_hidden_state: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...
    intermediate_embeddings: list[torch.FloatTensor] | None = ...

@dataclass
class SamHQMMaskDecoderOutputs(ModelOutput):
    masks: torch.FloatTensor
    iou_scores: torch.FloatTensor | None = ...
    mask_decoder_attentions: torch.FloatTensor | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class SamHQImageSegmentationOutput(ModelOutput):
    iou_scores: torch.FloatTensor | None = ...
    pred_masks: torch.FloatTensor | None = ...
    vision_hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    vision_attentions: tuple[torch.FloatTensor, ...] | None = ...
    mask_decoder_attentions: tuple[torch.FloatTensor, ...] | None = ...

class SamHQVisionAttention(nn.Module):
    def __init__(self, config, window_size) -> None: ...
    def get_rel_pos(
        self, q_size: int, k_size: int, rel_pos: torch.Tensor
    ) -> torch.Tensor: ...
    def get_decomposed_rel_pos(
        self,
        query: torch.Tensor,
        rel_pos_h: torch.Tensor,
        rel_pos_w: torch.Tensor,
        q_size: tuple[int, int],
        k_size: tuple[int, int],
    ) -> torch.Tensor: ...
    def forward(
        self, hidden_states: torch.Tensor, output_attentions=...
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

class SamHQMLPBlock(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class SamHQVisionSdpaAttention(SamHQVisionAttention):
    def __init__(self, config, window_size) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, output_attentions=...
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

SAM_HQ_VISION_ATTENTION_CLASSES = ...

class SamHQVisionLayer(GradientCheckpointingLayer):
    def __init__(self, config, window_size) -> None: ...
    def window_partition(
        self, hidden_states: torch.Tensor, window_size: int
    ) -> tuple[torch.Tensor, tuple[int, int]]: ...
    def window_unpartition(
        self,
        windows: torch.Tensor,
        window_size: int,
        padding_shape: tuple[int, int],
        original_shape: tuple[int, int],
    ) -> torch.Tensor: ...
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.FloatTensor]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple[torch.FloatTensor]: ...

@auto_docstring
class SamHQPreTrainedModel(PreTrainedModel):
    config: SamHQConfig
    base_model_prefix = ...
    main_input_name = ...
    _no_split_modules = ...
    supports_gradient_checkpointing = ...
    _supports_sdpa = ...

class SamHQPatchEmbeddings(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, pixel_values):  # -> Any:
        ...

class SamHQVisionNeck(nn.Module):
    def __init__(self, config: SamHQVisionConfig) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class SamHQVisionEncoder(SamHQPreTrainedModel):
    _can_record_outputs = ...
    def __init__(self, config: SamHQVisionConfig) -> None: ...
    def get_input_embeddings(self):  # -> SamHQPatchEmbeddings:
        ...
    @check_model_inputs
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | SamHQVisionEncoderOutput: ...

class SamHQLayerNorm(nn.LayerNorm):
    def __init__(
        self, normalized_shape, *, eps=..., data_format=..., **kwargs
    ) -> None: ...
    def forward(self, features: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

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

class SamHQAttention(nn.Module):
    def __init__(self, config, downsample_rate=...) -> None: ...
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_similarity: Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...

class SamHQTwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        config,
        attention_downsample_rate: int = ...,
        skip_first_layer_pe: bool = ...,
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

class SamHQTwoWayTransformer(nn.Module):
    def __init__(self, config: SamHQMaskDecoderConfig) -> None: ...
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

class SamHQFeedForward(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = ...,
    ) -> None: ...
    def forward(self, hidden_states):  # -> Tensor | Any:
        ...

class SamHQMaskDecoder(nn.Module):
    def __init__(self, config: SamHQMaskDecoderConfig) -> None: ...
    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_positional_embeddings: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        hq_token_only: bool,
        intermediate_embeddings: list[torch.Tensor] | None = ...,
        attention_similarity: torch.Tensor | None = ...,
        target_embedding: torch.Tensor | None = ...,
    ) -> SamHQMMaskDecoderOutputs: ...
    def __call__(self, *args: Any, **kwargs: Any) -> SamHQMMaskDecoderOutputs: ...

@auto_docstring(custom_intro=...)
class SamHQVisionModel(SamHQPreTrainedModel):
    config: SamHQVisionConfig
    main_input_name = ...
    def __init__(self, config: SamHQVisionConfig) -> None: ...
    def get_input_embeddings(self) -> nn.Module: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | SamHQVisionEncoderOutput: ...

class SamHQPositionalEmbedding(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, input_coords, input_shape=...):  # -> Tensor:
        ...

class SamHQMaskEmbedding(nn.Module):
    def __init__(self, config: SamHQPromptEncoderConfig) -> None: ...
    def forward(self, masks):  # -> Any:
        ...

class SamHQPromptEncoder(nn.Module):
    def __init__(self, config: SamHQConfig) -> None: ...
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

@auto_docstring(custom_intro=...)
class SamHQModel(SamHQPreTrainedModel):
    _tied_weights_keys = ...
    _keys_to_ignore_on_load_missing = ...
    _can_record_outputs = ...
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> SamHQPatchEmbeddings:
        ...
    def get_image_wide_positional_embeddings(self):  # -> Any:
        ...
    @torch.no_grad()
    def get_image_embeddings(self, pixel_values):  # -> tuple[Any, Any]:
        ...
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
        hq_token_only: bool = ...,
        attention_similarity: torch.FloatTensor | None = ...,
        target_embedding: torch.FloatTensor | None = ...,
        intermediate_embeddings: list[torch.FloatTensor] | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> list[dict[str, torch.Tensor]]: ...

__all__ = ["SamHQModel", "SamHQPreTrainedModel", "SamHQVisionModel"]
