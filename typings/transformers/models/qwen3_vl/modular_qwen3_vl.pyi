from torch import nn

import torch

from ..qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLCausalLMOutputWithPast,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModel,
    Qwen2_5_VLVisionBlock,
)
from ..qwen2_vl.modeling_qwen2_vl import (
    PatchEmbed,
    Qwen2VLModelOutputWithPast,
    Qwen2VLPreTrainedModel,
    TransformersKwargs,
    VisionAttention,
    VisionRotaryEmbedding,
)
from ..qwen2_vl.processing_qwen2_vl import Qwen2VLImagesKwargs, Qwen2VLProcessor
from ..qwen3.modeling_qwen3 import Qwen3Attention, Qwen3DecoderLayer, Qwen3Model
from ...cache_utils import Cache
from ...configuration_utils import PretrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import BaseModelOutputWithPast
from ...modeling_rope_utils import dynamic_rope_update
from ...processing_utils import ProcessingKwargs, Unpack, VideosKwargs
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring
from ...utils.generic import check_model_inputs
from ...video_utils import VideoInput

"""PyTorch Qwen3-VL model."""
logger = ...

class Qwen3VLVisionConfig(PretrainedConfig):
    model_type = ...
    base_config_key = ...
    def __init__(
        self,
        depth=...,
        hidden_size=...,
        hidden_act=...,
        intermediate_size=...,
        num_heads=...,
        in_channels=...,
        patch_size=...,
        spatial_merge_size=...,
        temporal_patch_size=...,
        out_hidden_size=...,
        num_position_embeddings=...,
        deepstack_visual_indexes=...,
        initializer_range=...,
        **kwargs,
    ) -> None: ...

class Qwen3VLTextConfig(PretrainedConfig):
    model_type = ...
    base_config_key = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        intermediate_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        num_key_value_heads=...,
        head_dim=...,
        hidden_act=...,
        max_position_embeddings=...,
        initializer_range=...,
        rms_norm_eps=...,
        use_cache=...,
        tie_word_embeddings=...,
        rope_theta=...,
        rope_scaling=...,
        attention_bias=...,
        attention_dropout=...,
        **kwargs,
    ) -> None: ...

class Qwen3VLConfig(PretrainedConfig):
    model_type = ...
    sub_configs = ...
    keys_to_ignore_at_inference = ...
    def __init__(
        self,
        text_config=...,
        vision_config=...,
        image_token_id=...,
        video_token_id=...,
        vision_start_token_id=...,
        vision_end_token_id=...,
        tie_word_embeddings=...,
        **kwargs,
    ) -> None: ...

class Qwen3VLVisionMLP(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_state):  # -> Any:
        ...

class Qwen3VLVisionPatchEmbed(PatchEmbed):
    def __init__(self, config) -> None: ...

class Qwen3VLVisionRotaryEmbedding(VisionRotaryEmbedding): ...

class Qwen3VLVisionPatchMerger(nn.Module):
    def __init__(
        self, config: Qwen3VLVisionConfig, use_postshuffle_norm=...
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class Qwen3VLVisionAttention(VisionAttention):
    def __init__(self, config: Qwen3VLVisionConfig) -> None: ...

class Qwen3VLVisionBlock(Qwen2_5_VLVisionBlock):
    def __init__(self, config, attn_implementation: str = ...) -> None: ...

class Qwen3VLTextRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor
    def __init__(self, config: Qwen3VLTextConfig, device=...) -> None: ...
    def apply_interleaved_mrope(self, freqs, mrope_section): ...
    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):  # -> tuple[Tensor, Tensor]:
        ...

class Qwen3VLTextAttention(Qwen3Attention):
    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class Qwen3VLTextDecoderLayer(Qwen3DecoderLayer):
    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor: ...

class Qwen3VLModelOutputWithPast(Qwen2VLModelOutputWithPast): ...

class Qwen3VLPreTrainedModel(Qwen2VLPreTrainedModel):
    config: Qwen3VLConfig
    _no_split_modules = ...
    _can_record_outputs = ...

class Qwen3VLVisionModel(Qwen3VLPreTrainedModel):
    config: Qwen3VLVisionConfig
    _no_split_modules = ...
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor: ...
    def fast_pos_embed_interpolate(self, grid_thw):  # -> Tensor:
        ...
    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs
    ) -> torch.Tensor: ...

@auto_docstring(custom_intro=...)
class Qwen3VLTextModel(Qwen3VLPreTrainedModel, Qwen3Model):
    config: Qwen3VLTextConfig
    _no_split_modules = ...
    def __init__(self, config: Qwen3VLTextConfig) -> None: ...
    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        visual_pos_masks: torch.Tensor | None = ...,
        deepstack_visual_embeds: list[torch.Tensor] | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple | BaseModelOutputWithPast: ...

@auto_docstring
class Qwen3VLModel(Qwen2_5_VLModel):
    config: Qwen3VLConfig
    _checkpoint_conversion_mapping = ...
    _no_split_modules = ...
    def __init__(self, config) -> None: ...
    def get_rope_index(
        self,
        input_ids: torch.LongTensor | None = ...,
        image_grid_thw: torch.LongTensor | None = ...,
        video_grid_thw: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = ...,
    ):  # -> tuple[tuple[Tensor, ...], Any]:
        ...
    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor | None = ...,
    ):  # -> tuple[tuple[Tensor, ...], Any]:
        ...
    @auto_docstring
    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        pixel_values: torch.Tensor | None = ...,
        pixel_values_videos: torch.FloatTensor | None = ...,
        image_grid_thw: torch.LongTensor | None = ...,
        video_grid_thw: torch.LongTensor | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Qwen3VLModelOutputWithPast: ...

class Qwen3VLCausalLMOutputWithPast(Qwen2_5_VLCausalLMOutputWithPast): ...

class Qwen3VLForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    config: Qwen3VLConfig
    _checkpoint_conversion_mapping = ...
    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        pixel_values: torch.Tensor | None = ...,
        pixel_values_videos: torch.FloatTensor | None = ...,
        image_grid_thw: torch.LongTensor | None = ...,
        video_grid_thw: torch.LongTensor | None = ...,
        cache_position: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Qwen3VLCausalLMOutputWithPast: ...
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=...,
        attention_mask=...,
        inputs_embeds=...,
        cache_position=...,
        position_ids=...,
        use_cache=...,
        pixel_values=...,
        pixel_values_videos=...,
        image_grid_thw=...,
        video_grid_thw=...,
        **kwargs,
    ):  # -> dict[Any, Any]:
        ...

class Qwen3VLVideosProcessorKwargs(VideosKwargs, total=False): ...
class Qwen3VLImagesKwargs(Qwen2VLImagesKwargs): ...

class Qwen3VLProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: Qwen3VLImagesKwargs
    videos_kwargs: Qwen3VLVideosProcessorKwargs
    _defaults = ...

class Qwen3VLProcessor(Qwen2VLProcessor):
    def __init__(
        self,
        image_processor=...,
        tokenizer=...,
        video_processor=...,
        chat_template=...,
        **kwargs,
    ) -> None: ...
    def __call__(
        self,
        images: ImageInput = ...,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput] = ...,
        videos: VideoInput = ...,
        **kwargs: Unpack[Qwen3VLProcessorKwargs],
    ) -> BatchFeature: ...

__all__ = [
    "Qwen3VLConfig",
    "Qwen3VLForConditionalGeneration",
    "Qwen3VLModel",
    "Qwen3VLPreTrainedModel",
    "Qwen3VLProcessor",
    "Qwen3VLTextConfig",
    "Qwen3VLTextModel",
    "Qwen3VLVisionModel",
]
