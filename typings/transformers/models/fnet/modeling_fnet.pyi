from dataclasses import dataclass
from typing import Any

from torch import nn

import torch

from .configuration_fnet import FNetConfig
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    ModelOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring

"""PyTorch FNet model."""
logger = ...

def two_dim_matmul(x, matrix_dim_one, matrix_dim_two):  # -> Tensor:
    ...
def fftn(x): ...

class FNetEmbeddings(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, input_ids=..., token_type_ids=..., position_ids=..., inputs_embeds=...
    ):  # -> Any:
        ...

class FNetBasicFourierTransform(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> tuple[Tensor | Any]:
        ...

class FNetBasicOutput(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states, input_tensor):  # -> Any:
        ...

class FNetFourierTransform(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> tuple[Any]:
        ...

class FNetIntermediate(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class FNetOutput(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class FNetLayer(GradientCheckpointingLayer):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> tuple[Tensor]:
        ...
    def feed_forward_chunk(self, fourier_output):  # -> Any:
        ...

class FNetEncoder(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states, output_hidden_states=..., return_dict=...
    ):  # -> tuple[Any | tuple[Any, ...] | tuple[()], ...] | BaseModelOutput:
        ...

class FNetPooler(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class FNetPredictionHeadTransform(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class FNetLMPredictionHead(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class FNetOnlyMLMHead(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, sequence_output):  # -> Any:
        ...

class FNetOnlyNSPHead(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, pooled_output):  # -> Any:
        ...

class FNetPreTrainingHeads(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, sequence_output, pooled_output):  # -> tuple[Any, Any]:
        ...

@auto_docstring
class FNetPreTrainedModel(PreTrainedModel):
    config: FNetConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...

@dataclass
@auto_docstring(
    custom_intro="""
    Output type of [`FNetForPreTraining`].
    """
)
class FNetForPreTrainingOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    prediction_logits: torch.FloatTensor | None = ...
    seq_relationship_logits: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...

@auto_docstring
class FNetModel(FNetPreTrainedModel):
    def __init__(self, config, add_pooling_layer=...) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutput: ...

@auto_docstring(custom_intro=...)
class FNetForPreTraining(FNetPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    def get_output_embeddings(self):  # -> Linear:
        ...
    def set_output_embeddings(self, new_embeddings):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        next_sentence_label: torch.Tensor | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | FNetForPreTrainingOutput: ...

@auto_docstring
class FNetForMaskedLM(FNetPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    def get_output_embeddings(self):  # -> Linear:
        ...
    def set_output_embeddings(self, new_embeddings):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | MaskedLMOutput: ...

@auto_docstring(custom_intro=...)
class FNetForNextSentencePrediction(FNetPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        **kwargs,
    ) -> tuple | NextSentencePredictorOutput: ...

@auto_docstring(custom_intro=...)
class FNetForSequenceClassification(FNetPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | SequenceClassifierOutput: ...

@auto_docstring
class FNetForMultipleChoice(FNetPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | MultipleChoiceModelOutput: ...

@auto_docstring
class FNetForTokenClassification(FNetPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | TokenClassifierOutput: ...

@auto_docstring
class FNetForQuestionAnswering(FNetPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        start_positions: torch.Tensor | None = ...,
        end_positions: torch.Tensor | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | QuestionAnsweringModelOutput: ...

__all__ = [
    "FNetForMaskedLM",
    "FNetForMultipleChoice",
    "FNetForNextSentencePrediction",
    "FNetForPreTraining",
    "FNetForQuestionAnswering",
    "FNetForSequenceClassification",
    "FNetForTokenClassification",
    "FNetLayer",
    "FNetModel",
    "FNetPreTrainedModel",
]
