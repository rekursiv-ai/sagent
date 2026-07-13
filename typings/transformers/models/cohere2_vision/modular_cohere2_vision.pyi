from functools import lru_cache

from torch import nn
from transformers.models.aya_vision.modeling_aya_vision import (
    AyaVisionCausalLMOutputWithPast,
    AyaVisionForConditionalGeneration,
    AyaVisionModel,
    AyaVisionModelOutputWithPast,
)
from transformers.models.got_ocr2.image_processing_got_ocr2_fast import (
    GotOcr2ImageProcessorFast,
)

import torch

from .configuration_cohere2_vision import Cohere2VisionConfig
from ...cache_utils import Cache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import check_model_inputs

"""PyTorch AyaVision model."""
logger = ...

class Cohere2VisionMultiModalProjector(nn.Module):
    def __init__(self, config: Cohere2VisionConfig) -> None: ...
    def pixel_shuffle(self, image_features): ...
    def forward(self, image_features):  # -> Any:
        ...

class Cohere2VisionModelOutputWithPast(AyaVisionModelOutputWithPast): ...
class Cohere2VisionCausalLMOutputWithPast(AyaVisionCausalLMOutputWithPast): ...

class Cohere2VisionModel(AyaVisionModel):
    _checkpoint_conversion_mapping = ...
    def get_image_features(self, pixel_values: torch.FloatTensor):  # -> Any:
        ...
    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple | Cohere2VisionModelOutputWithPast: ...

class Cohere2VisionForConditionalGeneration(AyaVisionForConditionalGeneration):
    _checkpoint_conversion_mapping = ...
    def get_image_features(self, pixel_values: torch.FloatTensor):  # -> Any:
        ...
    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        image_sizes: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Cohere2VisionCausalLMOutputWithPast: ...

@lru_cache(maxsize=10)
def get_all_supported_aspect_ratios(max_image_tiles: int) -> list[tuple[int, int]]: ...
def get_optimal_tiled_canvas(
    original_image_size: tuple[int, int],
    target_tile_size: tuple[int, int],
    min_image_tiles: int,
    max_image_tiles: int,
) -> tuple[int, int]: ...

@auto_docstring
class Cohere2VisionImageProcessorFast(GotOcr2ImageProcessorFast):
    size = ...
    min_patches = ...
    max_patches = ...
    crop_to_patches = ...
    patch_size = ...

__all__ = [
    "Cohere2VisionForConditionalGeneration",
    "Cohere2VisionImageProcessorFast",
    "Cohere2VisionModel",
    "Cohere2VisionPreTrainedModel",
]
