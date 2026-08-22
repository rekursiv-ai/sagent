from typing import Any

from transformers.models.instructblip.configuration_instructblip import (
    InstructBlipQFormerConfig,
    InstructBlipVisionConfig,
)
from transformers.models.instructblip.modeling_instructblip import (
    InstructBlipForConditionalGeneration,
    InstructBlipForConditionalGenerationModelOutput,
    InstructBlipModel,
    InstructBlipPreTrainedModel,
    InstructBlipQFormerModel,
    InstructBlipVisionModel,
    TransformersKwargs,
)

import torch

from ...configuration_utils import PretrainedConfig
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...processing_utils import Unpack

logger = ...

class InstructBlipVideoVisionConfig(InstructBlipVisionConfig): ...
class InstructBlipVideoQFormerConfig(InstructBlipQFormerConfig): ...

class InstructBlipVideoConfig(PretrainedConfig):
    def __init__(
        self,
        vision_config=...,
        qformer_config=...,
        text_config=...,
        num_query_tokens=...,
        video_token_index=...,
        **kwargs,
    ) -> None: ...
    @classmethod
    def from_vision_qformer_text_configs(
        cls,
        vision_config: InstructBlipVideoVisionConfig,
        qformer_config: InstructBlipVideoQFormerConfig,
        text_config: PretrainedConfig,
        **kwargs,
    ):  # -> Self:
        ...

class InstructBlipVideoPreTrainedModel(InstructBlipPreTrainedModel): ...
class InstructBlipVideoVisionModel(InstructBlipVisionModel): ...
class InstructBlipVideoQFormerModel(InstructBlipQFormerModel): ...
class InstructBlipVideoForConditionalGenerationModelOutput(
    InstructBlipForConditionalGenerationModelOutput
): ...

class InstructBlipVideoModel(InstructBlipModel):
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        qformer_input_ids: torch.FloatTensor,
        qformer_attention_mask: torch.LongTensor | None = ...,
        input_ids: torch.FloatTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        decoder_input_ids: torch.LongTensor | None = ...,
        decoder_attention_mask: torch.LongTensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        interpolate_pos_encoding: bool = ...,
        use_cache: bool | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple | InstructBlipVideoForConditionalGenerationModelOutput: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple | InstructBlipVideoForConditionalGenerationModelOutput: ...

class InstructBlipVideoForConditionalGeneration(InstructBlipForConditionalGeneration):
    def get_video_features(
        self,
        pixel_values: torch.FloatTensor,
        qformer_input_ids: torch.LongTensor,
        qformer_attention_mask: torch.LongTensor | None = ...,
        interpolate_pos_encoding: bool | None = ...,
        return_dict: bool | None = ...,
    ):  # -> tuple[Any, Any, Any] | Any:
        ...
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        qformer_input_ids: torch.LongTensor,
        qformer_attention_mask: torch.LongTensor | None = ...,
        interpolate_pos_encoding: bool | None = ...,
        return_dict: bool | None = ...,
    ):  # -> None:
        ...
    def get_placeholder_mask(
        self, input_ids: torch.LongTensor, inputs_embeds: torch.FloatTensor
    ):  # -> Any:
        ...
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        qformer_input_ids: torch.FloatTensor,
        qformer_attention_mask: torch.LongTensor | None = ...,
        input_ids: torch.FloatTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        decoder_input_ids: torch.LongTensor | None = ...,
        decoder_attention_mask: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        labels: torch.LongTensor | None = ...,
        return_dict: bool | None = ...,
        interpolate_pos_encoding: bool = ...,
        use_cache: bool | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | InstructBlipVideoForConditionalGenerationModelOutput: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple | InstructBlipVideoForConditionalGenerationModelOutput: ...
    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.FloatTensor,
        qformer_input_ids: torch.LongTensor | None = ...,
        qformer_attention_mask: torch.LongTensor | None = ...,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        interpolate_pos_encoding: bool = ...,
        **generate_kwargs,
    ) -> torch.LongTensor: ...

__all__ = [
    "InstructBlipVideoConfig",
    "InstructBlipVideoForConditionalGeneration",
    "InstructBlipVideoModel",
    "InstructBlipVideoPreTrainedModel",
    "InstructBlipVideoQFormerConfig",
    "InstructBlipVideoQFormerModel",
    "InstructBlipVideoVisionConfig",
    "InstructBlipVideoVisionModel",
]
