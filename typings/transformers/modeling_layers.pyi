from torch import nn

import torch

from .cache_utils import Cache
from .modeling_outputs import (
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from .processing_utils import Unpack
from .utils import TransformersKwargs, auto_docstring, can_return_tuple

logger = ...

class GradientCheckpointingLayer(nn.Module):
    gradient_checkpointing = ...
    def __call__(self, *args, **kwargs):  # -> Any:
        ...

@auto_docstring
class GenericForSequenceClassification:
    base_model_prefix = ...
    def __init__(self, config) -> None: ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> SequenceClassifierOutputWithPast: ...

@auto_docstring
class GenericForQuestionAnswering:
    base_model_prefix = ...
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> Any:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        start_positions: torch.LongTensor | None = ...,
        end_positions: torch.LongTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> QuestionAnsweringModelOutput: ...

@auto_docstring
class GenericForTokenClassification:
    base_model_prefix = ...
    def __init__(self, config) -> None: ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> TokenClassifierOutput: ...
