from torch import nn
from transformers.models.llava_next.modeling_llava_next import (
    LlavaNextCausalLMOutputWithPast,
    LlavaNextForConditionalGeneration,
    LlavaNextModel,
    LlavaNextModelOutputWithPast,
    LlavaNextMultiModalProjector,
    TransformersKwargs,
)

import torch

from ...cache_utils import Cache
from ...configuration_utils import PretrainedConfig
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...processing_utils import Unpack

logger = ...

class LlavaNextVideoConfig(PretrainedConfig):
    model_type = ...
    attribute_map = ...
    sub_configs = ...
    def __init__(
        self,
        vision_config=...,
        text_config=...,
        image_token_index=...,
        projector_hidden_act=...,
        multimodal_projector_bias=...,
        vision_feature_select_strategy=...,
        vision_feature_layer=...,
        image_grid_pinpoints=...,
        tie_word_embeddings=...,
        video_token_index=...,
        spatial_pool_mode=...,
        spatial_pool_stride=...,
        image_seq_length=...,
        video_seq_length=...,
        **kwargs,
    ) -> None: ...

class LlavaNextVideoModelOutputWithPast(LlavaNextModelOutputWithPast):
    video_hidden_states: torch.FloatTensor | None = ...

class LlavaNextVideoCausalLMOutputWithPast(LlavaNextCausalLMOutputWithPast):
    video_hidden_states: torch.FloatTensor | None = ...

class LlavaNextVideoPooler(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, image_features):  # -> Any:
        ...

class LlavaNextVideoMultiModalProjector(LlavaNextMultiModalProjector): ...

class LlavaNextVideoModel(LlavaNextModel):
    def __init__(self, config: LlavaNextVideoConfig, **super_kwargs) -> None: ...
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_sizes: torch.Tensor,
        vision_feature_layer: int | list[int] | None = ...,
        vision_feature_select_strategy: str | None = ...,
    ):  # -> list[Any]:
        ...
    def get_video_features(
        self,
        pixel_values: torch.FloatTensor,
        vision_feature_layer: int | list[int] | None = ...,
        vision_feature_select_strategy: str | None = ...,
    ):  # -> tuple[Tensor, ...]:
        ...
    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor | None = ...,
        video_features: torch.FloatTensor | None = ...,
    ):  # -> tuple[Any, Any]:
        ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        pixel_values_videos: torch.FloatTensor | None = ...,
        image_sizes: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        vision_feature_layer: int | list[int] | None = ...,
        vision_feature_select_strategy: str | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple | LlavaNextVideoModelOutputWithPast: ...

class LlavaNextVideoForConditionalGeneration(LlavaNextForConditionalGeneration):
    def get_video_features(
        self,
        pixel_values: torch.FloatTensor,
        vision_feature_layer: int | list[int] | None = ...,
        vision_feature_select_strategy: str | None = ...,
    ):  # -> Any:
        ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        pixel_values_videos: torch.FloatTensor | None = ...,
        image_sizes: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        vision_feature_layer: int | list[int] | None = ...,
        vision_feature_select_strategy: str | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | LlavaNextVideoCausalLMOutputWithPast: ...
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=...,
        inputs_embeds=...,
        pixel_values=...,
        pixel_values_videos=...,
        image_sizes=...,
        attention_mask=...,
        cache_position=...,
        logits_to_keep=...,
        **kwargs,
    ):  # -> dict[Any, Any]:
        ...

__all__ = [
    "LlavaNextVideoConfig",
    "LlavaNextVideoForConditionalGeneration",
    "LlavaNextVideoModel",
    "LlavaNextVideoPreTrainedModel",
]
