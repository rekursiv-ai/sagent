from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from .configuration_flaubert import FlaubertConfig
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFMultipleChoiceModelOutput,
    TFQuestionAnsweringModelOutput,
    TFSequenceClassifierOutput,
    TFTokenClassifierOutput,
)
from ...modeling_tf_utils import (
    TFModelInputType,
    TFMultipleChoiceLoss,
    TFPreTrainedModel,
    TFQuestionAnsweringLoss,
    TFSequenceClassificationLoss,
    TFTokenClassificationLoss,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)

"""
TF 2.0 Flaubert model.
"""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
FLAUBERT_START_DOCSTRING = ...
FLAUBERT_INPUTS_DOCSTRING = ...

def get_masks(slen, lengths, causal, padding_mask=...):  # -> tuple[Any, Any]:
    ...

class TFFlaubertPreTrainedModel(TFPreTrainedModel):
    config_class = FlaubertConfig
    base_model_prefix = ...
    @property
    def dummy_inputs(self):  # -> dict[str, Any]:
        ...

@add_start_docstrings(
    ...,
    FLAUBERT_START_DOCSTRING,
)
class TFFlaubertModel(TFFlaubertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(FLAUBERT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: np.ndarray | tf.Tensor | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        langs: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        lengths: np.ndarray | tf.Tensor | None = ...,
        cache: dict[str, tf.Tensor] | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool | None = ...,
    ) -> tuple | TFBaseModelOutput: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFFlaubertMultiHeadAttention(keras.layers.Layer):
    NEW_ID = ...
    def __init__(self, n_heads, dim, config, **kwargs) -> None: ...
    def prune_heads(self, heads): ...
    def call(
        self, input, mask, kv, cache, head_mask, output_attentions, training=...
    ):  # -> tuple[Any, Any] | tuple[Any]:
        ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFFlaubertTransformerFFN(keras.layers.Layer):
    def __init__(self, in_dim, dim_hidden, out_dim, config, **kwargs) -> None: ...
    def call(self, input, training=...): ...
    def build(self, input_shape=...):  # -> None:
        ...

@keras_serializable
class TFFlaubertMainLayer(keras.layers.Layer):
    config_class = FlaubertConfig
    def __init__(self, config, **kwargs) -> None: ...
    def build(self, input_shape=...):  # -> None:
        ...
    def get_input_embeddings(self):  # -> TFSharedEmbeddings:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @unpack_inputs
    def call(
        self,
        input_ids: np.ndarray | tf.Tensor | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        langs: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        lengths: np.ndarray | tf.Tensor | None = ...,
        cache: dict[str, tf.Tensor] | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool | None = ...,
    ) -> tuple | TFBaseModelOutput: ...

class TFFlaubertPredLayer(keras.layers.Layer):
    def __init__(self, config, input_embeddings, **kwargs) -> None: ...
    def build(self, input_shape):  # -> None:
        ...
    def get_output_embeddings(self):  # -> Any:
        ...
    def set_output_embeddings(self, value):  # -> None:
        ...
    def get_bias(self):  # -> dict[str, Any]:
        ...
    def set_bias(self, value):  # -> None:
        ...
    def call(self, hidden_states): ...

@dataclass
class TFFlaubertWithLMHeadModelOutput(ModelOutput):
    logits: tf.Tensor | None = ...
    hidden_states: tuple[tf.Tensor] | None = ...
    attentions: tuple[tf.Tensor] | None = ...

@add_start_docstrings(
    ...,
    FLAUBERT_START_DOCSTRING,
)
class TFFlaubertWithLMHeadModel(TFFlaubertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    def get_lm_head(self):  # -> TFFlaubertPredLayer:
        ...
    def get_prefix_bias_name(self): ...
    def prepare_inputs_for_generation(
        self, inputs, **kwargs
    ):  # -> dict[str, Any | None]:
        ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(FLAUBERT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFFlaubertWithLMHeadModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: np.ndarray | tf.Tensor | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        langs: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        lengths: np.ndarray | tf.Tensor | None = ...,
        cache: dict[str, tf.Tensor] | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool | None = ...,
    ) -> tuple | TFFlaubertWithLMHeadModelOutput: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    FLAUBERT_START_DOCSTRING,
)
class TFFlaubertForSequenceClassification(
    TFFlaubertPreTrainedModel, TFSequenceClassificationLoss
):
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        FLAUBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        langs: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        lengths: np.ndarray | tf.Tensor | None = ...,
        cache: dict[str, tf.Tensor] | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: np.ndarray | tf.Tensor | None = ...,
        training: bool = ...,
    ) -> TFSequenceClassifierOutput | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    FLAUBERT_START_DOCSTRING,
)
class TFFlaubertForQuestionAnsweringSimple(
    TFFlaubertPreTrainedModel, TFQuestionAnsweringLoss
):
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        FLAUBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFQuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        langs: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        lengths: np.ndarray | tf.Tensor | None = ...,
        cache: dict[str, tf.Tensor] | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        start_positions: np.ndarray | tf.Tensor | None = ...,
        end_positions: np.ndarray | tf.Tensor | None = ...,
        training: bool = ...,
    ) -> TFQuestionAnsweringModelOutput | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    FLAUBERT_START_DOCSTRING,
)
class TFFlaubertForTokenClassification(
    TFFlaubertPreTrainedModel, TFTokenClassificationLoss
):
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        FLAUBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFTokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        langs: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        lengths: np.ndarray | tf.Tensor | None = ...,
        cache: dict[str, tf.Tensor] | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: np.ndarray | tf.Tensor | None = ...,
        training: bool = ...,
    ) -> TFTokenClassifierOutput | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    FLAUBERT_START_DOCSTRING,
)
class TFFlaubertForMultipleChoice(TFFlaubertPreTrainedModel, TFMultipleChoiceLoss):
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    @property
    def dummy_inputs(self):  # -> dict[str, Any]:
        ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        FLAUBERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFMultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        langs: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        lengths: np.ndarray | tf.Tensor | None = ...,
        cache: dict[str, tf.Tensor] | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: np.ndarray | tf.Tensor | None = ...,
        training: bool = ...,
    ) -> TFMultipleChoiceModelOutput | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

__all__ = [
    "TFFlaubertForMultipleChoice",
    "TFFlaubertForQuestionAnsweringSimple",
    "TFFlaubertForSequenceClassification",
    "TFFlaubertForTokenClassification",
    "TFFlaubertModel",
    "TFFlaubertPreTrainedModel",
    "TFFlaubertWithLMHeadModel",
]
