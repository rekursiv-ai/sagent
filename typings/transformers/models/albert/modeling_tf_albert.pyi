from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from .configuration_albert import AlbertConfig
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFBaseModelOutputWithPooling,
    TFMaskedLMOutput,
    TFMultipleChoiceModelOutput,
    TFQuestionAnsweringModelOutput,
    TFSequenceClassifierOutput,
    TFTokenClassifierOutput,
)
from ...modeling_tf_utils import (
    TFMaskedLanguageModelingLoss,
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
    replace_return_docstrings,
)

"""TF 2.0 ALBERT model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...

class TFAlbertPreTrainingLoss:
    def hf_compute_loss(self, labels: tf.Tensor, logits: tf.Tensor) -> tf.Tensor: ...

class TFAlbertEmbeddings(keras.layers.Layer):
    def __init__(self, config: AlbertConfig, **kwargs) -> None: ...
    def build(self, input_shape=...):  # -> None:
        ...
    def call(
        self,
        input_ids: tf.Tensor | None = ...,
        position_ids: tf.Tensor | None = ...,
        token_type_ids: tf.Tensor | None = ...,
        inputs_embeds: tf.Tensor | None = ...,
        past_key_values_length=...,
        training: bool = ...,
    ) -> tf.Tensor: ...

class TFAlbertAttention(keras.layers.Layer):
    def __init__(self, config: AlbertConfig, **kwargs) -> None: ...
    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor: ...
    def call(
        self,
        input_tensor: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = ...,
    ) -> tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFAlbertLayer(keras.layers.Layer):
    def __init__(self, config: AlbertConfig, **kwargs) -> None: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = ...,
    ) -> tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFAlbertLayerGroup(keras.layers.Layer):
    def __init__(self, config: AlbertConfig, **kwargs) -> None: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        output_hidden_states: bool,
        training: bool = ...,
    ) -> TFBaseModelOutput | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFAlbertTransformer(keras.layers.Layer):
    def __init__(self, config: AlbertConfig, **kwargs) -> None: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        training: bool = ...,
    ) -> TFBaseModelOutput | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFAlbertPreTrainedModel(TFPreTrainedModel):
    config_class = AlbertConfig
    base_model_prefix = ...

class TFAlbertMLMHead(keras.layers.Layer):
    def __init__(
        self, config: AlbertConfig, input_embeddings: keras.layers.Layer, **kwargs
    ) -> None: ...
    def build(self, input_shape=...):  # -> None:
        ...
    def get_output_embeddings(self) -> keras.layers.Layer: ...
    def set_output_embeddings(self, value: tf.Variable):  # -> None:
        ...
    def get_bias(self) -> dict[str, tf.Variable]: ...
    def set_bias(self, value: tf.Variable):  # -> None:
        ...
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor: ...

@keras_serializable
class TFAlbertMainLayer(keras.layers.Layer):
    config_class = AlbertConfig
    def __init__(
        self, config: AlbertConfig, add_pooling_layer: bool = ..., **kwargs
    ) -> None: ...
    def get_input_embeddings(self) -> keras.layers.Layer: ...
    def set_input_embeddings(self, value: tf.Variable):  # -> None:
        ...
    @unpack_inputs
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> TFBaseModelOutputWithPooling | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

@dataclass
class TFAlbertForPreTrainingOutput(ModelOutput):
    loss: tf.Tensor | None = ...
    prediction_logits: tf.Tensor | None = ...
    sop_logits: tf.Tensor | None = ...
    hidden_states: tuple[tf.Tensor] | None = ...
    attentions: tuple[tf.Tensor] | None = ...

ALBERT_START_DOCSTRING = ...
ALBERT_INPUTS_DOCSTRING = ...

@add_start_docstrings(
    ...,
    ALBERT_START_DOCSTRING,
)
class TFAlbertModel(TFAlbertPreTrainedModel):
    def __init__(self, config: AlbertConfig, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        ALBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool | None = ...,
    ) -> TFBaseModelOutputWithPooling | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    ALBERT_START_DOCSTRING,
)
class TFAlbertForPreTraining(TFAlbertPreTrainedModel, TFAlbertPreTrainingLoss):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config: AlbertConfig, *inputs, **kwargs) -> None: ...
    def get_lm_head(self) -> keras.layers.Layer: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        ALBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @replace_return_docstrings(
        output_type=TFAlbertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: np.ndarray | tf.Tensor | None = ...,
        sentence_order_label: np.ndarray | tf.Tensor | None = ...,
        training: bool | None = ...,
    ) -> TFAlbertForPreTrainingOutput | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFAlbertSOPHead(keras.layers.Layer):
    def __init__(self, config: AlbertConfig, **kwargs) -> None: ...
    def call(self, pooled_output: tf.Tensor, training: bool) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(..., ALBERT_START_DOCSTRING)
class TFAlbertForMaskedLM(TFAlbertPreTrainedModel, TFMaskedLanguageModelingLoss):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config: AlbertConfig, *inputs, **kwargs) -> None: ...
    def get_lm_head(self) -> keras.layers.Layer: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        ALBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @replace_return_docstrings(
        output_type=TFMaskedLMOutput, config_class=_CONFIG_FOR_DOC
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: np.ndarray | tf.Tensor | None = ...,
        training: bool | None = ...,
    ) -> TFMaskedLMOutput | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    ALBERT_START_DOCSTRING,
)
class TFAlbertForSequenceClassification(
    TFAlbertPreTrainedModel, TFSequenceClassificationLoss
):
    _keys_to_ignore_on_load_unexpected = ...
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config: AlbertConfig, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        ALBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint="vumichien/albert-base-v2-imdb",
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="'LABEL_1'",
        expected_loss=0.12,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: np.ndarray | tf.Tensor | None = ...,
        training: bool | None = ...,
    ) -> TFSequenceClassifierOutput | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    ALBERT_START_DOCSTRING,
)
class TFAlbertForTokenClassification(
    TFAlbertPreTrainedModel, TFTokenClassificationLoss
):
    _keys_to_ignore_on_load_unexpected = ...
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config: AlbertConfig, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        ALBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
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
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: np.ndarray | tf.Tensor | None = ...,
        training: bool | None = ...,
    ) -> TFTokenClassifierOutput | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    ALBERT_START_DOCSTRING,
)
class TFAlbertForQuestionAnswering(TFAlbertPreTrainedModel, TFQuestionAnsweringLoss):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config: AlbertConfig, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        ALBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint="vumichien/albert-base-v2-squad2",
        output_type=TFQuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
        qa_target_start_index=12,
        qa_target_end_index=13,
        expected_output="'a nice puppet'",
        expected_loss=7.36,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        start_positions: np.ndarray | tf.Tensor | None = ...,
        end_positions: np.ndarray | tf.Tensor | None = ...,
        training: bool | None = ...,
    ) -> TFQuestionAnsweringModelOutput | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    ALBERT_START_DOCSTRING,
)
class TFAlbertForMultipleChoice(TFAlbertPreTrainedModel, TFMultipleChoiceLoss):
    _keys_to_ignore_on_load_unexpected = ...
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config: AlbertConfig, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        ALBERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
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
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: np.ndarray | tf.Tensor | None = ...,
        training: bool | None = ...,
    ) -> TFMultipleChoiceModelOutput | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

__all__ = [
    "TFAlbertForMaskedLM",
    "TFAlbertForMultipleChoice",
    "TFAlbertForPreTraining",
    "TFAlbertForQuestionAnswering",
    "TFAlbertForSequenceClassification",
    "TFAlbertForTokenClassification",
    "TFAlbertMainLayer",
    "TFAlbertModel",
    "TFAlbertPreTrainedModel",
]
