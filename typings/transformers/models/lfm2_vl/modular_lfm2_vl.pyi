from torch import nn

import torch

from .configuration_lfm2_vl import Lfm2VlConfig
from ..llava.modeling_llava import (
    LlavaCausalLMOutputWithPast,
    LlavaForConditionalGeneration,
    LlavaModel,
    LlavaModelOutputWithPast,
    LlavaPreTrainedModel,
)
from ...cache_utils import Cache
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple

"""PyTorch Lfm2-VL model."""
logger = ...

class Lfm2VlMultiModalProjector(nn.Module):
    def __init__(self, config: Lfm2VlConfig) -> None: ...
    def forward(self, image_features: torch.Tensor):  # -> Any:
        ...
    def pixel_unshuffle(self, hidden_states: torch.Tensor):  # -> Tensor:
        ...

class Lfm2VlPreTrainedModel(LlavaPreTrainedModel):
    _can_compile_fullgraph = ...

class Lfm2VlCausalLMOutputWithPast(LlavaCausalLMOutputWithPast): ...
class Lfm2VlModelOutputWithPast(LlavaModelOutputWithPast): ...

class Lfm2VlModel(LlavaModel):
    _checkpoint_conversion_mapping = ...
    def __init__(self, config: Lfm2VlConfig) -> None: ...
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        spatial_shapes: torch.Tensor,
        pixel_attention_mask: torch.Tensor,
        **kwargs,
    ) -> list[torch.Tensor]: ...
    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor,
    ):  # -> Any:
        ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        spatial_shapes: torch.Tensor | None = ...,
        pixel_attention_mask: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Lfm2VlModelOutputWithPast: ...

class Lfm2VlForConditionalGeneration(LlavaForConditionalGeneration):
    _checkpoint_conversion_mapping = ...
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        spatial_shapes: torch.Tensor,
        pixel_attention_mask: torch.Tensor,
        **kwargs,
    ):  # -> tuple[Tensor, ...] | list[Any]:
        ...
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        spatial_shapes: torch.Tensor | None = ...,
        pixel_attention_mask: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Lfm2VlCausalLMOutputWithPast: ...

__all__ = ["Lfm2VlForConditionalGeneration", "Lfm2VlModel", "Lfm2VlPreTrainedModel"]
