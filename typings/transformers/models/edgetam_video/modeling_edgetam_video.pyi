from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from torch import Tensor, nn

import torch

from .configuration_edgetam_video import (
    EdgeTamVideoConfig,
    EdgeTamVideoMaskDecoderConfig,
    EdgeTamVideoPromptEncoderConfig,
)
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...pytorch_utils import compile_compatible_method_lru_cache
from ...utils import ModelOutput, auto_docstring
from ...utils.generic import TransformersKwargs

class EdgeTamVideoLayerNorm(nn.LayerNorm):
    def __init__(
        self, normalized_shape, *, eps=..., data_format=..., **kwargs
    ) -> None: ...
    def forward(self, features: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class EdgeTamVideoMemoryFuserCXBlock(GradientCheckpointingLayer):
    def __init__(self, config: EdgeTamVideoConfig) -> None: ...
    def forward(self, hidden_states): ...

@dataclass
@auto_docstring(custom_intro="Base class for the vision encoder's outputs.")
class EdgeTamVideoVisionEncoderOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = ...
    fpn_hidden_states: torch.FloatTensor | None = ...
    fpn_position_encoding: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...

class EdgeTamVideoVisionRotaryEmbedding(nn.Module):
    def __init__(
        self,
        config: EdgeTamVideoConfig,
        end_x: int | None = ...,
        end_y: int | None = ...,
    ) -> None: ...
    @torch.no_grad()
    def forward(self) -> tuple[torch.Tensor, torch.Tensor]: ...

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

class EdgeTamVideoAttention(nn.Module):
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

def rotate_pairwise(x):  # -> Tensor:
    ...
def apply_rotary_pos_emb_2d_self_attn(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]: ...

class EdgeTamVideoRoPESelfAttention(nn.Module):
    def __init__(self, config: EdgeTamVideoConfig) -> None: ...
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...

def apply_rotary_pos_emb_2d_cross_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cos_k: torch.Tensor,
    sin_k: torch.Tensor,
    num_k_exclude_rope: int = ...,
    repeat_freqs_k: int = ...,
) -> tuple[torch.Tensor, torch.Tensor]: ...

class EdgeTamVideoRoPECrossAttention(nn.Module):
    def __init__(self, config: EdgeTamVideoConfig, kv_in_dim: int) -> None: ...
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_embeddings_k: tuple[torch.Tensor, torch.Tensor],
        num_k_exclude_rope: int = ...,
        rope_k_repeat: int = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...

class EdgeTamVideoTwoWayAttentionBlock(nn.Module):
    def __init__(
        self, config: EdgeTamVideoMaskDecoderConfig, skip_first_layer_pe: bool = ...
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

class EdgeTamVideoPositionEmbeddingSine(nn.Module):
    def __init__(
        self,
        num_pos_feats: int = ...,
        temperature: int = ...,
        normalize: bool = ...,
        scale: float | None = ...,
    ) -> None: ...
    @compile_compatible_method_lru_cache(maxsize=2)
    def forward(
        self,
        shape: torch.Size,
        device: torch.device | str,
        dtype: torch.dtype,
        mask: Tensor | None = ...,
    ) -> Tensor: ...

class EdgeTamVideoMemoryFuser(nn.Module):
    def __init__(self, config: EdgeTamVideoConfig) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class EdgeTamVideoMaskDownSamplerLayer(nn.Module):
    def __init__(
        self, config: EdgeTamVideoConfig, in_channels: int, out_channels: int
    ) -> None: ...
    def forward(self, x): ...

class EdgeTamVideoMaskDownSampler(nn.Module):
    def __init__(self, config: EdgeTamVideoConfig) -> None: ...
    def forward(self, x):  # -> Any:
        ...

class EdgeTamVideoMemoryEncoder(nn.Module):
    def __init__(self, config: EdgeTamVideoConfig) -> None: ...
    def forward(
        self, vision_features: torch.Tensor, masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

class EdgeTamVideoFeedForward(nn.Module):
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
class EdgeTamVideoPreTrainedModel(PreTrainedModel):
    config_class = EdgeTamVideoConfig
    base_model_prefix = ...
    main_input_name = ...
    _supports_sdpa = ...
    _supports_flash_attn_2 = ...
    _supports_attention_backend = ...

class EdgeTamVideoInferenceCache:
    def __init__(
        self,
        inference_device: torch.device | str = ...,
        inference_state_device: torch.device | str = ...,
        max_vision_features_cache_size: int = ...,
    ) -> None: ...
    def cache_vision_features(self, frame_idx: int, features: dict):  # -> None:
        ...
    def get_vision_features(self, frame_idx: int) -> dict | None: ...
    def clear_all(self):  # -> None:
        ...

class EdgeTamVideoInferenceSession:
    def __init__(
        self,
        video: torch.FloatTensor | None = ...,
        video_height: int | None = ...,
        video_width: int | None = ...,
        inference_device: torch.device | str = ...,
        inference_state_device: torch.device | str = ...,
        video_storage_device: torch.device | str = ...,
        dtype: torch.dtype | str = ...,
        max_vision_features_cache_size: int = ...,
    ) -> None: ...
    @property
    def num_frames(self) -> int | None: ...
    def obj_id_to_idx(self, obj_id: int) -> int: ...
    def obj_idx_to_id(self, obj_idx: int) -> int: ...
    def get_obj_num(self) -> int: ...
    def add_point_inputs(self, obj_idx: int, frame_idx: int, inputs: dict):  # -> None:
        ...
    def remove_point_inputs(self, obj_idx: int, frame_idx: int):  # -> None:
        ...
    def add_mask_inputs(
        self, obj_idx: int, frame_idx: int, inputs: torch.Tensor
    ):  # -> None:
        ...
    def remove_mask_inputs(self, obj_idx: int, frame_idx: int):  # -> None:
        ...
    def store_output(
        self,
        obj_idx: int,
        frame_idx: int,
        output_key: str | None = ...,
        output_value: torch.Tensor | dict | None = ...,
        is_conditioning_frame: bool = ...,
    ):  # -> None:
        ...
    def get_output(
        self,
        obj_idx: int,
        frame_idx: int,
        output_key: str,
        is_conditioning_frame: bool = ...,
    ):  # -> Tensor | None:
        ...
    def add_new_frame(
        self, pixel_values: torch.Tensor, frame_idx: int | None = ...
    ) -> int: ...
    def get_frame(self, frame_idx: int) -> torch.Tensor: ...
    def reset_tracking_data(self):  # -> None:
        ...
    def reset_inference_session(self):  # -> None:
        ...

class EdgeTamVideoMemoryAttentionMLP(nn.Module):
    def __init__(self, config: EdgeTamVideoConfig) -> None: ...
    def forward(self, x):  # -> Any:
        ...

class EdgeTamVideoMemoryAttentionLayer(nn.Module):
    def __init__(self, config: EdgeTamVideoConfig) -> None: ...
    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        key_point_embedding: Tensor,
        rope_position_embeddings: tuple[Tensor, Tensor],
        rope_position_embeddings_k: tuple[Tensor, Tensor] | None = ...,
        num_k_exclude_rope: int = ...,
        rope_k_repeat: int = ...,
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class EdgeTamVideoMemoryAttention(nn.Module):
    def __init__(self, config: EdgeTamVideoConfig) -> None: ...
    def forward(
        self,
        current_vision_features: torch.Tensor,
        memory: torch.Tensor,
        current_vision_position_embeddings: Tensor | None = ...,
        memory_posision_embeddings: Tensor | None = ...,
        num_object_pointer_tokens: int = ...,
        num_spatial_memory_tokens: int = ...,
    ):  # -> Any:
        ...

class EdgeTamVideoPerceiverMLP(nn.Module):
    def __init__(self, config: EdgeTamVideoConfig) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class EdgeTamVideoPerceiverAttention(nn.Module):
    def __init__(self, config: EdgeTamVideoConfig) -> None: ...
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positional_encoding: torch.Tensor | None = ...,
        **kwargs,
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class EdgeTamVideoPerceiverEncoderLayer(nn.Module):
    def __init__(self, config: EdgeTamVideoConfig) -> None: ...
    def forward(
        self,
        latents: torch.Tensor,
        input_features: torch.Tensor,
        positional_encoding: torch.Tensor | None = ...,
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

def window_partition(hidden_state, window_size):  # -> tuple[Tensor, tuple[Any, Any]]:
    ...

class EdgeTamVideoPerceiverResampler(nn.Module):
    def __init__(self, config: EdgeTamVideoConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        positional_encoding: torch.Tensor | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

@dataclass
@auto_docstring(custom_intro="Base class for the EdgeTamVideo model's output.")
class EdgeTamVideoImageSegmentationOutput(ModelOutput):
    iou_scores: torch.FloatTensor | None = ...
    pred_masks: torch.FloatTensor | None = ...
    object_score_logits: torch.FloatTensor | None = ...
    image_embeddings: tuple[torch.FloatTensor, ...] = ...
    vision_hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    vision_attentions: tuple[torch.FloatTensor, ...] | None = ...
    mask_decoder_attentions: tuple[torch.FloatTensor, ...] | None = ...
    high_res_masks: torch.FloatTensor | None = ...
    object_pointer: torch.FloatTensor | None = ...

@dataclass
@auto_docstring(custom_intro="Base class for the Sam2 model's output.")
class EdgeTamVideoSegmentationOutput(ModelOutput):
    pred_masks: torch.FloatTensor | None = ...
    frame_idx: int | None = ...

class EdgeTamVideoPositionalEmbedding(nn.Module):
    def __init__(self, config: EdgeTamVideoPromptEncoderConfig) -> None: ...
    def forward(self, input_coords, input_shape=...):  # -> Tensor:
        ...

class EdgeTamVideoMaskEmbedding(nn.Module):
    def __init__(self, config: EdgeTamVideoPromptEncoderConfig) -> None: ...
    def forward(self, masks):  # -> Any:
        ...

class EdgeTamVideoPromptEncoder(nn.Module):
    def __init__(self, config: EdgeTamVideoPromptEncoderConfig) -> None: ...
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

class EdgeTamVideoTwoWayTransformer(nn.Module):
    def __init__(self, config: EdgeTamVideoMaskDecoderConfig) -> None: ...
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

class EdgeTamVideoMaskDecoder(nn.Module):
    def __init__(self, config: EdgeTamVideoMaskDecoderConfig) -> None: ...
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

NO_OBJ_SCORE = ...

def get_1d_sine_pe(pos_inds, dim, temperature=...):  # -> Tensor:
    ...

@auto_docstring
class EdgeTamVideoModel(EdgeTamVideoPreTrainedModel):
    _tied_weights_keys = ...
    _keys_to_ignore_on_load_missing = ...
    _can_record_outputs = ...
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config: EdgeTamVideoConfig) -> None: ...
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
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    @torch.inference_mode()
    @auto_docstring(custom_intro=...)
    def forward(
        self,
        inference_session: EdgeTamVideoInferenceSession,
        frame_idx: int | None = ...,
        frame: torch.Tensor | None = ...,
        reverse: bool = ...,
    ) -> EdgeTamVideoSegmentationOutput: ...
    def get_image_features(
        self, pixel_values: torch.FloatTensor, **kwargs: Unpack[TransformersKwargs]
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        tuple[torch.FloatTensor, ...] | None,
        tuple[torch.FloatTensor, ...] | None,
    ]: ...
    @torch.inference_mode()
    @auto_docstring(custom_intro=...)
    def propagate_in_video_iterator(
        self,
        inference_session: EdgeTamVideoInferenceSession,
        start_frame_idx: int | None = ...,
        max_frame_num_to_track: int | None = ...,
        reverse: bool = ...,
    ) -> Iterator[EdgeTamVideoSegmentationOutput]: ...

__all__ = [
    "EdgeTamVideoInferenceSession",
    "EdgeTamVideoModel",
    "EdgeTamVideoPreTrainedModel",
]
