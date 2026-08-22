from typing import Any

from torch import nn

import torch

from .configuration_mistral3 import Mistral3Config
from ..llava.modeling_llava import (
    LlavaCausalLMOutputWithPast,
    LlavaForConditionalGeneration,
    LlavaModel,
    LlavaModelOutputWithPast,
    LlavaPreTrainedModel,
    TransformersKwargs,
)
from ..mistral.modeling_mistral import MistralRMSNorm
from ...cache_utils import Cache
from ...processing_utils import Unpack

logger = ...

class Mistral3RMSNorm(MistralRMSNorm): ...

class Mistral3PatchMerger(nn.Module):
    def __init__(self, config: Mistral3Config) -> None: ...
    def forward(
        self, image_features: torch.Tensor, image_sizes: torch.Tensor
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class Mistral3MultiModalProjector(nn.Module):
    def __init__(self, config: Mistral3Config) -> None: ...
    def forward(
        self, image_features: torch.Tensor, image_sizes: torch.Tensor
    ):  # -> Any:
        ...

class Mistral3CausalLMOutputWithPast(LlavaCausalLMOutputWithPast): ...
class Mistral3ModelOutputWithPast(LlavaModelOutputWithPast): ...
class Mistral3PreTrainedModel(LlavaPreTrainedModel): ...

class Mistral3Model(LlavaModel):
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_sizes: torch.Tensor,
        vision_feature_layer: int | list[int] | None = ...,
        **kwargs,
    ):  # -> tuple[Tensor, ...]:
        ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        vision_feature_layer: int | list[int] | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        image_sizes: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Mistral3ModelOutputWithPast: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple | Mistral3ModelOutputWithPast: ...

class Mistral3ForConditionalGeneration(LlavaForConditionalGeneration):
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_sizes: torch.Tensor,
        vision_feature_layer: int | list[int] | None = ...,
        **kwargs,
    ):  # -> tuple[Tensor, ...] | list[Any]:
        ...
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
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        image_sizes: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Mistral3CausalLMOutputWithPast: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple | Mistral3CausalLMOutputWithPast: ...

__all__ = [
    "Mistral3ForConditionalGeneration",
    "Mistral3Model",
    "Mistral3PreTrainedModel",
]
