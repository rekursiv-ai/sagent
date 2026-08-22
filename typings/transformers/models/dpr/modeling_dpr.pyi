from dataclasses import dataclass
from typing import Any

from torch import Tensor

import torch

from .configuration_dpr import DPRConfig
from ...modeling_outputs import BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring

"""PyTorch DPR model for Open Domain Question Answering."""
logger = ...

@dataclass
@auto_docstring(custom_intro=...)
class DPRContextEncoderOutput(ModelOutput):
    pooler_output: torch.FloatTensor
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class DPRQuestionEncoderOutput(ModelOutput):
    pooler_output: torch.FloatTensor
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class DPRReaderOutput(ModelOutput):
    start_logits: torch.FloatTensor
    end_logits: torch.FloatTensor | None = ...
    relevance_logits: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...

@auto_docstring
class DPRPreTrainedModel(PreTrainedModel):
    _supports_sdpa = ...

class DPREncoder(DPRPreTrainedModel):
    base_model_prefix = ...
    def __init__(self, config: DPRConfig) -> None: ...
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = ...,
        token_type_ids: Tensor | None = ...,
        inputs_embeds: Tensor | None = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ) -> BaseModelOutputWithPooling | tuple[Tensor, ...]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> BaseModelOutputWithPooling | tuple[Tensor, ...]: ...
    @property
    def embeddings_size(self) -> int: ...

class DPRSpanPredictor(DPRPreTrainedModel):
    base_model_prefix = ...
    def __init__(self, config: DPRConfig) -> None: ...
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        inputs_embeds: Tensor | None = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ) -> DPRReaderOutput | tuple[Tensor, ...]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> DPRReaderOutput | tuple[Tensor, ...]: ...

class DPRPretrainedContextEncoder(DPRPreTrainedModel):
    config: DPRConfig
    load_tf_weights = ...
    base_model_prefix = ...

class DPRPretrainedQuestionEncoder(DPRPreTrainedModel):
    config: DPRConfig
    load_tf_weights = ...
    base_model_prefix = ...

class DPRPretrainedReader(DPRPreTrainedModel):
    config: DPRConfig
    load_tf_weights = ...
    base_model_prefix = ...

@auto_docstring(custom_intro=...)
class DPRContextEncoder(DPRPretrainedContextEncoder):
    def __init__(self, config: DPRConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: Tensor | None = ...,
        attention_mask: Tensor | None = ...,
        token_type_ids: Tensor | None = ...,
        inputs_embeds: Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> DPRContextEncoderOutput | tuple[Tensor, ...]: ...

@auto_docstring(custom_intro=...)
class DPRQuestionEncoder(DPRPretrainedQuestionEncoder):
    def __init__(self, config: DPRConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: Tensor | None = ...,
        attention_mask: Tensor | None = ...,
        token_type_ids: Tensor | None = ...,
        inputs_embeds: Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> DPRQuestionEncoderOutput | tuple[Tensor, ...]: ...

@auto_docstring(custom_intro=...)
class DPRReader(DPRPretrainedReader):
    def __init__(self, config: DPRConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: Tensor | None = ...,
        attention_mask: Tensor | None = ...,
        inputs_embeds: Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> DPRReaderOutput | tuple[Tensor, ...]: ...

__all__ = [
    "DPRContextEncoder",
    "DPRPreTrainedModel",
    "DPRPretrainedContextEncoder",
    "DPRPretrainedQuestionEncoder",
    "DPRPretrainedReader",
    "DPRQuestionEncoder",
    "DPRReader",
]
