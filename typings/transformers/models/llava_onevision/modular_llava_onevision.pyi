from typing import Any
from transformers.models.llava_next.image_processing_llava_next_fast import (
    LlavaNextImageProcessorFast,
)
from transformers.models.llava_next_video.modeling_llava_next_video import (
    LlavaNextVideoCausalLMOutputWithPast,
    LlavaNextVideoForConditionalGeneration,
    LlavaNextVideoModel,
    LlavaNextVideoModelOutputWithPast,
    LlavaNextVideoPreTrainedModel,
    TransformersKwargs,
)

import torch

from ...cache_utils import Cache
from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import DefaultFastImageProcessorKwargs
from ...image_utils import ImageInput
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...processing_utils import Unpack
from ...utils import auto_docstring, can_return_tuple

logger = ...

class LlavaOnevisionFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    image_grid_pinpoints: list[list[int]] | None

class LlavaOnevisionImageProcessorFast(LlavaNextImageProcessorFast):
    resample = ...
    image_mean = ...
    image_std = ...
    size = ...
    crop_size = ...
    default_to_square = ...
    do_resize = ...
    do_center_crop = ...
    do_rescale = ...
    do_normalize = ...
    do_convert_rgb = ...
    do_pad = ...
    image_grid_pinpoints = ...
    model_input_names = ...
    def pad_to_square(
        self,
        images: torch.Tensor,
        background_color: int | tuple[int, int, int] = ...,
    ) -> torch.Tensor: ...
    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        **kwargs: Unpack[LlavaOnevisionFastImageProcessorKwargs],
    ) -> BatchFeature: ...

class LlavaOnevisionModelOutputWithPast(LlavaNextVideoModelOutputWithPast): ...
class LlavaOnevisionCausalLMOutputWithPast(LlavaNextVideoCausalLMOutputWithPast): ...
class LlavaOnevisionPreTrainedModel(LlavaNextVideoPreTrainedModel): ...

class LlavaOnevisionModel(LlavaNextVideoModel):
    def __init__(self, config) -> None: ...
    def pack_image_features(
        self, image_features, image_sizes, image_newline=..., vision_aspect_ratio=...
    ):  # -> tuple[list[Any], Tensor]:
        ...
    def apply_pooling(self, image_features):  # -> Tensor:
        ...
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_sizes: torch.Tensor,
        vision_feature_layer: int | list[int] | None = ...,
        vision_feature_select_strategy: str | None = ...,
        vision_aspect_ratio: str | None = ...,
        batch_num_images: torch.LongTensor | None = ...,
    ):  # -> list[Any]:
        ...
    def get_video_features(
        self,
        pixel_values: torch.FloatTensor,
        vision_feature_layer: int | list[int],
        vision_feature_select_strategy: str,
    ):  # -> Tensor:
        ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        image_sizes: torch.LongTensor | None = ...,
        pixel_values_videos: torch.FloatTensor | None = ...,
        image_sizes_videos: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        vision_feature_layer: int | list[int] | None = ...,
        vision_feature_select_strategy: str | None = ...,
        vision_aspect_ratio: str | None = ...,
        batch_num_images: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple | LlavaOnevisionModelOutputWithPast: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple | LlavaOnevisionModelOutputWithPast: ...

class LlavaOnevisionForConditionalGeneration(LlavaNextVideoForConditionalGeneration):
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        image_sizes: torch.LongTensor | None = ...,
        pixel_values_videos: torch.FloatTensor | None = ...,
        image_sizes_videos: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        vision_feature_layer: int | list[int] | None = ...,
        vision_feature_select_strategy: str | None = ...,
        vision_aspect_ratio: str | None = ...,
        batch_num_images: torch.LongTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | LlavaOnevisionCausalLMOutputWithPast: ...
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=...,
        inputs_embeds=...,
        pixel_values=...,
        image_sizes=...,
        pixel_values_videos=...,
        image_sizes_videos=...,
        attention_mask=...,
        cache_position=...,
        logits_to_keep=...,
        **kwargs,
    ):  # -> dict[Any, Any]:
        ...

__all__ = [
    "LlavaOnevisionForConditionalGeneration",
    "LlavaOnevisionImageProcessorFast",
    "LlavaOnevisionModel",
    "LlavaOnevisionPreTrainedModel",
]
