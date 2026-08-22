from dataclasses import dataclass
from typing import Any

from torch import nn

import torch

from .configuration_nezha import NezhaConfig
from ....cache_utils import Cache
from ....modeling_layers import GradientCheckpointingLayer
from ....modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ....modeling_utils import PreTrainedModel
from ....utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ....utils.deprecation import deprecate_kwarg

"""PyTorch Nezha model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...

def load_tf_weights_in_nezha(model, config, tf_checkpoint_path): ...

class NezhaRelativePositionsEncoding(nn.Module):
    def __init__(self, length, depth, max_relative_position=...) -> None: ...
    def forward(self, length):  # -> Tensor:
        ...

class NezhaEmbeddings(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class NezhaSelfAttention(nn.Module):
    def __init__(self, config) -> None: ...
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

class NezhaSelfOutput(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class NezhaAttention(nn.Module):
    def __init__(self, config) -> None: ...
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

class NezhaIntermediate(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class NezhaOutput(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class NezhaLayer(GradientCheckpointingLayer):
    def __init__(self, config) -> None: ...
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
    def feed_forward_chunk(self, attention_output):  # -> Any:
        ...

class NezhaEncoder(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        encoder_hidden_states: torch.FloatTensor | None = ...,
        encoder_attention_mask: torch.FloatTensor | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor] | BaseModelOutputWithPastAndCrossAttentions: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor] | BaseModelOutputWithPastAndCrossAttentions: ...

class NezhaPooler(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class NezhaPredictionHeadTransform(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class NezhaLMPredictionHead(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class NezhaOnlyMLMHead(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class NezhaOnlyNSPHead(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, pooled_output):  # -> Any:
        ...

class NezhaPreTrainingHeads(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, sequence_output, pooled_output):  # -> tuple[Any, Any]:
        ...

class NezhaPreTrainedModel(PreTrainedModel):
    config: NezhaConfig
    load_tf_weights = ...
    base_model_prefix = ...
    supports_gradient_checkpointing = ...

@dataclass
class NezhaForPreTrainingOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    prediction_logits: torch.FloatTensor | None = ...
    seq_relationship_logits: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...

NEZHA_START_DOCSTRING = ...
NEZHA_INPUTS_DOCSTRING = ...

@add_start_docstrings(
    ...,
    NEZHA_START_DOCSTRING,
)
class NezhaModel(NezhaPreTrainedModel):
    def __init__(self, config, add_pooling_layer=...) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @add_start_docstrings_to_model_forward(
        NEZHA_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        encoder_hidden_states: torch.Tensor | None = ...,
        encoder_attention_mask: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor] | BaseModelOutputWithPoolingAndCrossAttentions: ...

@add_start_docstrings(
    ...,
    NEZHA_START_DOCSTRING,
)
class NezhaForPreTraining(NezhaPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    def get_output_embeddings(self):  # -> Linear:
        ...
    def set_output_embeddings(self, new_embeddings):  # -> None:
        ...
    @add_start_docstrings_to_model_forward(
        NEZHA_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @replace_return_docstrings(
        output_type=NezhaForPreTrainingOutput, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        next_sentence_label: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor] | NezhaForPreTrainingOutput: ...

@add_start_docstrings(..., NEZHA_START_DOCSTRING)
class NezhaForMaskedLM(NezhaPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    def get_output_embeddings(self):  # -> Linear:
        ...
    def set_output_embeddings(self, new_embeddings):  # -> None:
        ...
    @add_start_docstrings_to_model_forward(
        NEZHA_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        encoder_hidden_states: torch.Tensor | None = ...,
        encoder_attention_mask: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor] | MaskedLMOutput: ...
    def prepare_inputs_for_generation(
        self, input_ids, attention_mask=..., **model_kwargs
    ):  # -> dict[str, Tensor | Any]:
        ...

@add_start_docstrings(
    ...,
    NEZHA_START_DOCSTRING,
)
class NezhaForNextSentencePrediction(NezhaPreTrainedModel):
    def __init__(self, config) -> None: ...
    @add_start_docstrings_to_model_forward(
        NEZHA_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @replace_return_docstrings(
        output_type=NextSentencePredictorOutput, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor] | NextSentencePredictorOutput: ...

@add_start_docstrings(
    ...,
    NEZHA_START_DOCSTRING,
)
class NezhaForSequenceClassification(NezhaPreTrainedModel):
    def __init__(self, config) -> None: ...
    @add_start_docstrings_to_model_forward(
        NEZHA_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor] | SequenceClassifierOutput: ...

@add_start_docstrings(
    ...,
    NEZHA_START_DOCSTRING,
)
class NezhaForMultipleChoice(NezhaPreTrainedModel):
    def __init__(self, config) -> None: ...
    @add_start_docstrings_to_model_forward(
        NEZHA_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
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
        token_type_ids: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor] | MultipleChoiceModelOutput: ...

@add_start_docstrings(
    ...,
    NEZHA_START_DOCSTRING,
)
class NezhaForTokenClassification(NezhaPreTrainedModel):
    def __init__(self, config) -> None: ...
    @add_start_docstrings_to_model_forward(
        NEZHA_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor] | TokenClassifierOutput: ...

@add_start_docstrings(
    ...,
    NEZHA_START_DOCSTRING,
)
class NezhaForQuestionAnswering(NezhaPreTrainedModel):
    def __init__(self, config) -> None: ...
    @add_start_docstrings_to_model_forward(
        NEZHA_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        start_positions: torch.Tensor | None = ...,
        end_positions: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor] | QuestionAnsweringModelOutput: ...

__all__ = [
    "NezhaForMaskedLM",
    "NezhaForMultipleChoice",
    "NezhaForNextSentencePrediction",
    "NezhaForPreTraining",
    "NezhaForQuestionAnswering",
    "NezhaForSequenceClassification",
    "NezhaForTokenClassification",
    "NezhaModel",
    "NezhaPreTrainedModel",
]
