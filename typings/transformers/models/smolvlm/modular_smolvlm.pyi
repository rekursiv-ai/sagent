import torch

from ..idefics3.configuration_idefics3 import Idefics3Config, Idefics3VisionConfig
from ..idefics3.image_processing_idefics3 import Idefics3ImageProcessor
from ..idefics3.image_processing_idefics3_fast import Idefics3ImageProcessorFast
from ..idefics3.modeling_idefics3 import (
    Idefics3BaseModelOutputWithPast,
    Idefics3ForConditionalGeneration,
    Idefics3Model,
    Idefics3PreTrainedModel,
    Idefics3VisionTransformer,
)
from ...cache_utils import Cache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...processing_utils import Unpack
from ...utils import auto_docstring, can_return_tuple

logger = ...

class SmolVLMVisionConfig(Idefics3VisionConfig): ...
class SmolVLMPreTrainedModel(Idefics3PreTrainedModel): ...
class SmolVLMVisionTransformer(Idefics3VisionTransformer): ...
class SmolVLMConfig(Idefics3Config): ...
class SmolVLMImageProcessor(Idefics3ImageProcessor): ...
class SmolVLMImageProcessorFast(Idefics3ImageProcessorFast): ...
class SmolVLMBaseModelOutputWithPast(Idefics3BaseModelOutputWithPast): ...

class SmolVLMModel(Idefics3Model):
    def inputs_merger(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.Tensor,
        image_hidden_states: torch.Tensor,
    ):  # -> Tensor:
        ...
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        pixel_attention_mask: torch.LongTensor | None = ...,
    ):  # -> Any:
        ...
    @can_return_tuple
    @auto_docstring(custom_intro=...)
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        pixel_attention_mask: torch.BoolTensor | None = ...,
        image_hidden_states: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple | SmolVLMBaseModelOutputWithPast: ...

class SmolVLMForConditionalGeneration(Idefics3ForConditionalGeneration):
    def __init__(self, config) -> None: ...
    def forward(self, **super_kwargs):  # -> None:
        ...

__all__ = [
    "SmolVLMConfig",
    "SmolVLMForConditionalGeneration",
    "SmolVLMImageProcessor",
    "SmolVLMImageProcessorFast",
    "SmolVLMModel",
    "SmolVLMPreTrainedModel",
    "SmolVLMVisionConfig",
    "SmolVLMVisionTransformer",
]
