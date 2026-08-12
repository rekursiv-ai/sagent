from typing import Any
from torch import Tensor, nn

import torch

from ..sam2_video.configuration_sam2_video import (
    Sam2VideoConfig,
    Sam2VideoMaskDecoderConfig,
    Sam2VideoPromptEncoderConfig,
)
from ..sam2_video.modeling_sam2_video import (
    Sam2VideoAttention,
    Sam2VideoFeedForward,
    Sam2VideoInferenceSession,
    Sam2VideoLayerNorm,
    Sam2VideoMemoryAttention,
    Sam2VideoMemoryEncoder,
    Sam2VideoMemoryFuserCXBlock,
    Sam2VideoModel,
    Sam2VideoPositionEmbeddingSine,
    Sam2VideoPreTrainedModel,
    Sam2VideoTwoWayAttentionBlock,
    Sam2VideoVisionEncoderOutput,
    Sam2VideoVisionRotaryEmbedding,
)
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...processing_utils import Unpack
from ...pytorch_utils import compile_compatible_method_lru_cache
from ...utils import auto_docstring

class EdgeTamVideoPromptEncoderConfig(Sam2VideoPromptEncoderConfig): ...
class EdgeTamVideoMaskDecoderConfig(Sam2VideoMaskDecoderConfig): ...

class EdgeTamVideoConfig(Sam2VideoConfig):
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
        memory_attention_mlp_hidden_size=...,
        memory_attention_mlp_hidden_act=...,
        memory_attention_dropout=...,
        memory_attention_rope_theta=...,
        memory_attention_rope_feat_sizes=...,
        memory_attention_rope_k_sizes=...,
        memory_attention_rope_dropout=...,
        perceiver_resampler_num_latents=...,
        perceiver_resampler_num_latents_2d=...,
        perceiver_resampler_hidden_size=...,
        perceiver_resampler_mlp_intermediate_size=...,
        perceiver_resampler_num_attention_heads=...,
        perceiver_resampler_attention_head_dim=...,
        perceiver_resampler_num_layers=...,
        perceiver_resampler_hidden_dropout=...,
        perceiver_resampler_attention_dropout=...,
        memory_encoder_hidden_size=...,
        memory_encoder_output_channels=...,
        mask_downsampler_embed_dim=...,
        memory_fuser_intermediate_dim=...,
        mask_downsampler_kernel_size=...,
        mask_downsampler_stride=...,
        mask_downsampler_padding=...,
        mask_downsampler_total_stride=...,
        mask_downsampler_hidden_act=...,
        memory_fuser_num_layers=...,
        memory_fuser_embed_dim=...,
        memory_fuser_kernel_size=...,
        memory_fuser_padding=...,
        memory_fuser_layer_scale_init_value=...,
        memory_fuser_hidden_act=...,
        **kwargs,
    ) -> None: ...

class EdgeTamVideoLayerNorm(Sam2VideoLayerNorm): ...
class EdgeTamVideoMemoryFuserCXBlock(Sam2VideoMemoryFuserCXBlock): ...
class EdgeTamVideoVisionEncoderOutput(Sam2VideoVisionEncoderOutput): ...

class EdgeTamVideoVisionRotaryEmbedding(Sam2VideoVisionRotaryEmbedding):
    def __init__(
        self,
        config: EdgeTamVideoConfig,
        end_x: int | None = ...,
        end_y: int | None = ...,
    ) -> None: ...

class EdgeTamVideoAttention(Sam2VideoAttention): ...

def apply_rotary_pos_emb_2d_self_attn(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]: ...
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

class EdgeTamVideoTwoWayAttentionBlock(Sam2VideoTwoWayAttentionBlock): ...

class EdgeTamVideoPositionEmbeddingSine(Sam2VideoPositionEmbeddingSine):
    @compile_compatible_method_lru_cache(maxsize=2)
    def forward(self, **super_kwargs):  # -> Tensor:
        ...

class EdgeTamVideoMemoryEncoder(Sam2VideoMemoryEncoder): ...
class EdgeTamVideoFeedForward(Sam2VideoFeedForward): ...
class EdgeTamVideoPreTrainedModel(Sam2VideoPreTrainedModel): ...
class EdgeTamVideoInferenceSession(Sam2VideoInferenceSession): ...

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

class EdgeTamVideoMemoryAttention(Sam2VideoMemoryAttention):
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

@auto_docstring
class EdgeTamVideoModel(Sam2VideoModel):
    _tied_weights_keys = ...
    _keys_to_ignore_on_load_missing = ...
    _keys_to_ignore_on_load_unexpected = ...
    _can_record_outputs = ...
    def __init__(self, config: EdgeTamVideoConfig) -> None: ...

__all__ = [
    "EdgeTamVideoConfig",
    "EdgeTamVideoInferenceSession",
    "EdgeTamVideoMaskDecoderConfig",
    "EdgeTamVideoModel",
    "EdgeTamVideoPreTrainedModel",
    "EdgeTamVideoPromptEncoderConfig",
]
