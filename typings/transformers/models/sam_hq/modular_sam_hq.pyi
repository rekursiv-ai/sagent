from dataclasses import dataclass
from typing import Any

from torch import nn
from transformers.modeling_outputs import ModelOutput
from transformers.utils.generic import TransformersKwargs, check_model_inputs

import torch

from ..sam.configuration_sam import (
    SamConfig,
    SamMaskDecoderConfig,
    SamPromptEncoderConfig,
    SamVisionConfig,
)
from ..sam.modeling_sam import (
    SamFeedForward,
    SamImageSegmentationOutput,
    SamLayerNorm,
    SamModel,
    SamPreTrainedModel,
    SamTwoWayTransformer,
    SamVisionAttention,
    SamVisionEncoder,
    SamVisionEncoderOutput,
    SamVisionLayer,
    SamVisionModel,
)
from ...processing_utils import Unpack
from ...utils import auto_docstring

logger = ...

class SamHQPromptEncoderConfig(SamPromptEncoderConfig): ...
class SamHQVisionConfig(SamVisionConfig): ...

class SamHQMaskDecoderConfig(SamMaskDecoderConfig):
    def __init__(self, vit_dim=..., **super_kwargs) -> None: ...

class SamHQConfig(SamConfig): ...

class SamHQVisionEncoderOutput(SamVisionEncoderOutput):
    intermediate_embeddings: list[torch.FloatTensor] | None = ...

@dataclass
class SamHQMMaskDecoderOutputs(ModelOutput):
    masks: torch.FloatTensor
    iou_scores: torch.FloatTensor | None = ...
    mask_decoder_attentions: torch.FloatTensor | None = ...

class SamHQImageSegmentationOutput(SamImageSegmentationOutput): ...
class SamHQVisionAttention(SamVisionAttention): ...
class SamHQVisionLayer(SamVisionLayer): ...
class SamHQPreTrainedModel(SamPreTrainedModel): ...

class SamHQVisionEncoder(SamVisionEncoder, SamHQPreTrainedModel):
    _can_record_outputs = ...
    @check_model_inputs
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | SamHQVisionEncoderOutput: ...

class SamHQLayerNorm(SamLayerNorm): ...
class SamHQTwoWayTransformer(SamTwoWayTransformer): ...
class SamHQFeedForward(SamFeedForward): ...

class SamHQMaskDecoder(nn.Module):
    def __init__(self, config: SamHQMaskDecoderConfig) -> None: ...
    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_positional_embeddings: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        hq_token_only: bool,
        intermediate_embeddings: list[torch.Tensor] | None = ...,
        attention_similarity: torch.Tensor | None = ...,
        target_embedding: torch.Tensor | None = ...,
    ) -> SamHQMMaskDecoderOutputs: ...
    def __call__(self, *args: Any, **kwargs: Any) -> SamHQMMaskDecoderOutputs: ...

class SamHQVisionModel(SamVisionModel): ...

@auto_docstring(custom_intro=...)
class SamHQModel(SamModel):
    _tied_weights_keys = ...
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config) -> None: ...
    @torch.no_grad()
    def get_image_embeddings(self, pixel_values):  # -> tuple[Any, Any]:
        ...
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        input_points: torch.FloatTensor | None = ...,
        input_labels: torch.LongTensor | None = ...,
        input_boxes: torch.FloatTensor | None = ...,
        input_masks: torch.LongTensor | None = ...,
        image_embeddings: torch.FloatTensor | None = ...,
        multimask_output: bool = ...,
        hq_token_only: bool = ...,
        attention_similarity: torch.FloatTensor | None = ...,
        target_embedding: torch.FloatTensor | None = ...,
        intermediate_embeddings: list[torch.FloatTensor] | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> list[dict[str, torch.Tensor]]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> list[dict[str, torch.Tensor]]: ...

__all__ = [
    "SamHQConfig",
    "SamHQMaskDecoderConfig",
    "SamHQModel",
    "SamHQPreTrainedModel",
    "SamHQPromptEncoderConfig",
    "SamHQVisionConfig",
    "SamHQVisionModel",
]
