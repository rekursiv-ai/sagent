from typing import Any

from torch import nn, tensor

import torch

from .configuration_ernie_m import ErnieMConfig
from ....cache_utils import Cache
from ....modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ....modeling_utils import PreTrainedModel
from ....utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from ....utils.deprecation import deprecate_kwarg

"""PyTorch ErnieM model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
_TOKENIZER_FOR_DOC = ...

class ErnieMEmbeddings(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        inputs_embeds: torch.LongTensor | None = ...,
        past_key_values_length: int = ...,
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class ErnieMSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=...) -> None: ...
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        encoder_hidden_states: torch.FloatTensor | None = ...,
        encoder_attention_mask: torch.FloatTensor | None = ...,
        past_key_values: Cache | None = ...,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.Tensor]: ...

class ErnieMAttention(nn.Module):
    def __init__(self, config, position_embedding_type=...) -> None: ...
    def prune_heads(self, heads):  # -> None:
        ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        encoder_hidden_states: torch.FloatTensor | None = ...,
        encoder_attention_mask: torch.FloatTensor | None = ...,
        past_key_values: Cache | None = ...,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.Tensor]: ...

class ErnieMEncoderLayer(nn.Module):
    def __init__(self, config) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        past_key_values: Cache | None = ...,
        output_attentions: bool | None = ...,
    ):  # -> tuple[Tensor, Any] | Tensor:
        ...

class ErnieMEncoder(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        input_embeds: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        past_key_values: Cache | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor] | BaseModelOutputWithPastAndCrossAttentions: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor] | BaseModelOutputWithPastAndCrossAttentions: ...

class ErnieMPooler(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class ErnieMPreTrainedModel(PreTrainedModel):
    config: ErnieMConfig
    base_model_prefix = ...

ERNIE_M_START_DOCSTRING = ...
ERNIE_M_INPUTS_DOCSTRING = ...

@add_start_docstrings(
    ...,
    ERNIE_M_START_DOCSTRING,
)
class ErnieMModel(ErnieMPreTrainedModel):
    def __init__(self, config, add_pooling_layer=...) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @add_start_docstrings_to_model_forward(
        ERNIE_M_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: tensor | None = ...,
        position_ids: tensor | None = ...,
        attention_mask: tensor | None = ...,
        head_mask: tensor | None = ...,
        inputs_embeds: tensor | None = ...,
        past_key_values: tuple[tuple[tensor]] | None = ...,
        use_cache: bool | None = ...,
        output_hidden_states: bool | None = ...,
        output_attentions: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.FloatTensor] | BaseModelOutputWithPoolingAndCrossAttentions: ...

@add_start_docstrings(
    ...,
    ERNIE_M_START_DOCSTRING,
)
class ErnieMForSequenceClassification(ErnieMPreTrainedModel):
    def __init__(self, config) -> None: ...
    @add_start_docstrings_to_model_forward(
        ERNIE_M_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool | None = ...,
        output_hidden_states: bool | None = ...,
        output_attentions: bool | None = ...,
        return_dict: bool | None = ...,
        labels: torch.Tensor | None = ...,
    ) -> tuple[torch.FloatTensor] | SequenceClassifierOutput: ...

@add_start_docstrings(
    ...,
    ERNIE_M_START_DOCSTRING,
)
class ErnieMForMultipleChoice(ErnieMPreTrainedModel):
    def __init__(self, config) -> None: ...
    @add_start_docstrings_to_model_forward(
        ERNIE_M_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.FloatTensor] | MultipleChoiceModelOutput: ...

@add_start_docstrings(
    ...,
    ERNIE_M_START_DOCSTRING,
)
class ErnieMForTokenClassification(ErnieMPreTrainedModel):
    def __init__(self, config) -> None: ...
    @add_start_docstrings_to_model_forward(
        ERNIE_M_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        output_hidden_states: bool | None = ...,
        output_attentions: bool | None = ...,
        return_dict: bool | None = ...,
        labels: torch.Tensor | None = ...,
    ) -> tuple[torch.FloatTensor] | TokenClassifierOutput: ...

@add_start_docstrings(
    ...,
    ERNIE_M_START_DOCSTRING,
)
class ErnieMForQuestionAnswering(ErnieMPreTrainedModel):
    def __init__(self, config) -> None: ...
    @add_start_docstrings_to_model_forward(
        ERNIE_M_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        start_positions: torch.Tensor | None = ...,
        end_positions: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.FloatTensor] | QuestionAnsweringModelOutput: ...

@add_start_docstrings(
    ...,
    ERNIE_M_START_DOCSTRING,
)
class ErnieMForInformationExtraction(ErnieMPreTrainedModel):
    def __init__(self, config) -> None: ...
    @add_start_docstrings_to_model_forward(
        ERNIE_M_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
    )
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        start_positions: torch.Tensor | None = ...,
        end_positions: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.FloatTensor] | QuestionAnsweringModelOutput: ...

__all__ = [
    "ErnieMForInformationExtraction",
    "ErnieMForMultipleChoice",
    "ErnieMForQuestionAnswering",
    "ErnieMForSequenceClassification",
    "ErnieMForTokenClassification",
    "ErnieMModel",
    "ErnieMPreTrainedModel",
]
