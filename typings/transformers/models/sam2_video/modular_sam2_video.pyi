from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from torch import Tensor, nn

import numpy as np
import torch

from ..sam2.configuration_sam2 import Sam2MaskDecoderConfig, Sam2PromptEncoderConfig
from ..sam2.modeling_sam2 import (
    Sam2FeedForward,
    Sam2ImageSegmentationOutput,
    Sam2LayerNorm,
    Sam2Model,
    Sam2SinePositionEmbedding,
    Sam2TwoWayAttentionBlock,
)
from ..sam2.processing_sam2 import Sam2Processor
from ...configuration_utils import PretrainedConfig
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import ModelOutput, auto_docstring
from ...video_utils import VideoInput

"""PyTorch SAM 2 model."""
logger = ...

class Sam2VideoPromptEncoderConfig(Sam2PromptEncoderConfig): ...
class Sam2VideoMaskDecoderConfig(Sam2MaskDecoderConfig): ...

class Sam2VideoConfig(PretrainedConfig):
    def __init__(
        self,
        vision_config=...,
        prompt_encoder_config=...,
        mask_decoder_config=...,
        initializer_range=...,
        num_maskmem=...,
        image_size=...,
        sigmoid_scale_for_mem_enc=...,
        sigmoid_bias_for_mem_enc=...,
        enable_occlusion_spatial_embedding=...,
        multimask_output_in_sam=...,
        multimask_min_pt_num=...,
        multimask_max_pt_num=...,
        multimask_output_for_tracking=...,
        max_object_pointers_in_encoder=...,
        enable_temporal_pos_encoding_for_object_pointers=...,
        memory_attention_hidden_size=...,
        memory_attention_num_layers=...,
        memory_attention_num_attention_heads=...,
        memory_attention_downsample_rate=...,
        memory_attention_feed_forward_hidden_size=...,
        memory_attention_feed_forward_hidden_act=...,
        memory_attention_dropout=...,
        memory_attention_rope_theta=...,
        memory_attention_rope_feat_sizes=...,
        memory_attention_rope_dropout=...,
        memory_encoder_hidden_size=...,
        memory_encoder_output_channels=...,
        mask_downsampler_embed_dim=...,
        mask_downsampler_kernel_size=...,
        mask_downsampler_stride=...,
        mask_downsampler_padding=...,
        mask_downsampler_total_stride=...,
        mask_downsampler_hidden_act=...,
        memory_fuser_num_layers=...,
        memory_fuser_embed_dim=...,
        memory_fuser_intermediate_dim=...,
        memory_fuser_kernel_size=...,
        memory_fuser_padding=...,
        memory_fuser_layer_scale_init_value=...,
        memory_fuser_hidden_act=...,
        **kwargs,
    ) -> None: ...

class Sam2VideoInferenceCache:
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

class Sam2VideoInferenceSession:
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

class Sam2VideoProcessor(Sam2Processor):
    attributes = ...
    image_processor_class = ...
    video_processor_class = ...
    def __init__(
        self,
        image_processor,
        video_processor,
        target_size: int | None = ...,
        point_pad_value: int = ...,
        **kwargs,
    ) -> None: ...
    def init_video_session(
        self,
        video: VideoInput | None = ...,
        inference_device: str | torch.device = ...,
        inference_state_device: str | torch.device = ...,
        processing_device: str | torch.device = ...,
        video_storage_device: str | torch.device = ...,
        max_vision_features_cache_size: int = ...,
        dtype: torch.dtype = ...,
    ):  # -> Sam2VideoInferenceSession:
        ...
    def add_inputs_to_inference_session(
        self,
        inference_session: Sam2VideoInferenceSession,
        frame_idx: int,
        obj_ids: list[int] | int,
        input_points: list[list[list[list[float]]]] | torch.Tensor | None = ...,
        input_labels: list[list[list[int]]] | torch.Tensor | None = ...,
        input_boxes: list[list[list[float]]] | torch.Tensor | None = ...,
        input_masks: np.ndarray
        | torch.Tensor
        | list[np.ndarray]
        | list[torch.Tensor]
        | None = ...,
        original_size: tuple[int, int] | None = ...,
        clear_old_inputs: bool = ...,
    ) -> Sam2VideoInferenceSession: ...
    def process_new_points_or_boxes_for_video_frame(
        self,
        inference_session: Sam2VideoInferenceSession,
        frame_idx: int,
        obj_ids: list[int],
        input_points: list[list[list[list[float]]]] | torch.Tensor | None = ...,
        input_labels: list[list[list[int]]] | torch.Tensor | None = ...,
        input_boxes: list[list[list[float]]] | torch.Tensor | None = ...,
        original_size: tuple[int, int] | None = ...,
        clear_old_inputs: bool = ...,
    ) -> Sam2VideoInferenceSession: ...
    def process_new_mask_for_video_frame(
        self,
        inference_session: Sam2VideoInferenceSession,
        frame_idx: int,
        obj_ids: list[int],
        input_masks: np.ndarray | torch.Tensor | list[np.ndarray] | list[torch.Tensor],
    ):  # -> None:
        ...

class Sam2VideoLayerNorm(Sam2LayerNorm): ...
class Sam2VideoPositionEmbeddingSine(Sam2SinePositionEmbedding): ...
class Sam2VideoTwoWayAttentionBlock(Sam2TwoWayAttentionBlock): ...
class Sam2VideoFeedForward(Sam2FeedForward): ...

class Sam2VideoImageSegmentationOutput(Sam2ImageSegmentationOutput):
    high_res_masks: torch.FloatTensor | None = ...
    object_pointer: torch.FloatTensor | None = ...

@dataclass
@auto_docstring(custom_intro="Base class for the Sam2 model's output.")
class Sam2VideoSegmentationOutput(ModelOutput):
    pred_masks: torch.FloatTensor | None = ...
    frame_idx: int | None = ...

@auto_docstring
class Sam2VideoPreTrainedModel(PreTrainedModel):
    config_class = Sam2VideoConfig
    base_model_prefix = ...
    main_input_name = ...
    _supports_sdpa = ...
    _supports_flash_attn_2 = ...
    _supports_attention_backend = ...

class Sam2VideoVisionRotaryEmbedding(nn.Module):
    def __init__(self, config: Sam2VideoConfig) -> None: ...
    @torch.no_grad()
    def forward(self) -> tuple[torch.Tensor, torch.Tensor]: ...

def rotate_pairwise(x):  # -> Tensor:
    ...
def apply_rotary_pos_emb_2d(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_k_exclude_rope: int = ...,
    repeat_freqs_k: bool = ...,
) -> tuple[torch.Tensor, torch.Tensor]: ...

class Sam2VideoRoPEAttention(nn.Module):
    def __init__(
        self, config: Sam2VideoConfig, kv_in_dim: int | None = ..., rope_k_repeat=...
    ) -> None: ...
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        num_k_exclude_rope: int = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...

class Sam2VideoMemoryAttentionLayer(nn.Module):
    def __init__(self, config: Sam2VideoConfig) -> None: ...
    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        key_point_embedding: Tensor,
        rope_position_embeddings: tuple[Tensor, Tensor],
        num_k_exclude_rope: int = ...,
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class Sam2VideoMemoryAttention(nn.Module):
    def __init__(self, config: Sam2VideoConfig) -> None: ...
    def forward(
        self,
        current_vision_features: torch.Tensor,
        memory: torch.Tensor,
        current_vision_position_embeddings: Tensor | None = ...,
        memory_posision_embeddings: Tensor | None = ...,
        num_object_pointer_tokens: int = ...,
    ):  # -> Any:
        ...

class Sam2VideoMemoryFuserCXBlock(GradientCheckpointingLayer):
    def __init__(self, config: Sam2VideoConfig) -> None: ...
    def forward(self, hidden_states): ...

class Sam2VideoMemoryFuser(nn.Module):
    def __init__(self, config: Sam2VideoConfig) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class Sam2VideoMaskDownSamplerLayer(nn.Module):
    def __init__(
        self, config: Sam2VideoConfig, in_channels: int, out_channels: int
    ) -> None: ...
    def forward(self, x): ...

class Sam2VideoMaskDownSampler(nn.Module):
    def __init__(self, config: Sam2VideoConfig) -> None: ...
    def forward(self, x):  # -> Any:
        ...

class Sam2VideoMemoryEncoder(nn.Module):
    def __init__(self, config: Sam2VideoConfig) -> None: ...
    def forward(
        self, vision_features: torch.Tensor, masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

NO_OBJ_SCORE = ...

def get_1d_sine_pe(pos_inds, dim, temperature=...):  # -> Tensor:
    ...

@auto_docstring
class Sam2VideoModel(Sam2Model):
    _tied_weights_keys = ...
    _keys_to_ignore_on_load_missing = ...
    _keys_to_ignore_on_load_unexpected = ...
    _can_record_outputs = ...
    def __init__(self, config: Sam2VideoConfig) -> None: ...
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
        inference_session: Sam2VideoInferenceSession,
        frame_idx: int | None = ...,
        frame: torch.Tensor | None = ...,
        reverse: bool = ...,
    ) -> Sam2VideoSegmentationOutput: ...
    @torch.inference_mode()
    @auto_docstring(custom_intro=...)
    def propagate_in_video_iterator(
        self,
        inference_session: Sam2VideoInferenceSession,
        start_frame_idx: int | None = ...,
        max_frame_num_to_track: int | None = ...,
        reverse: bool = ...,
    ) -> Iterator[Sam2VideoSegmentationOutput]: ...

__all__ = [
    "Sam2VideoConfig",
    "Sam2VideoInferenceSession",
    "Sam2VideoMaskDecoderConfig",
    "Sam2VideoModel",
    "Sam2VideoPreTrainedModel",
    "Sam2VideoProcessor",
    "Sam2VideoPromptEncoderConfig",
]
