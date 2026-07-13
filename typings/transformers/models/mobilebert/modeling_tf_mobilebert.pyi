from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from .configuration_mobilebert import MobileBertConfig
from ...modeling_tf_outputs import (
    TFBaseModelOutputWithPooling,
    TFMaskedLMOutput,
    TFMultipleChoiceModelOutput,
    TFNextSentencePredictorOutput,
    TFQuestionAnsweringModelOutput,
    TFSequenceClassifierOutput,
    TFTokenClassifierOutput,
)
from ...modeling_tf_utils import (
    TFMaskedLanguageModelingLoss,
    TFModelInputType,
    TFMultipleChoiceLoss,
    TFNextSentencePredictionLoss,
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

"""TF 2.0 MobileBERT model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_TOKEN_CLASSIFICATION = ...
_TOKEN_CLASS_EXPECTED_OUTPUT = ...
_TOKEN_CLASS_EXPECTED_LOSS = ...
_CHECKPOINT_FOR_QA = ...
_QA_EXPECTED_OUTPUT = ...
_QA_EXPECTED_LOSS = ...
_QA_TARGET_START_INDEX = ...
_QA_TARGET_END_INDEX = ...
_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION = ...
_SEQ_CLASS_EXPECTED_OUTPUT = ...
_SEQ_CLASS_EXPECTED_LOSS = ...

class TFMobileBertPreTrainingLoss:
    def hf_compute_loss(self, labels: tf.Tensor, logits: tf.Tensor) -> tf.Tensor: ...

class TFMobileBertIntermediate(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
    def call(self, hidden_states): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFLayerNorm(keras.layers.LayerNormalization):
    def __init__(self, feat_size, *args, **kwargs) -> None: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFNoNorm(keras.layers.Layer):
    def __init__(self, feat_size, epsilon=..., **kwargs) -> None: ...
    def build(self, input_shape):  # -> None:
        ...
    def call(self, inputs: tf.Tensor): ...

NORM2FN = ...

class TFMobileBertEmbeddings(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
    def build(self, input_shape=...):  # -> None:
        ...
    def call(
        self,
        input_ids=...,
        position_ids=...,
        token_type_ids=...,
        inputs_embeds=...,
        training=...,
    ): ...

class TFMobileBertSelfAttention(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
    def transpose_for_scores(self, x, batch_size): ...
    def call(
        self,
        query_tensor,
        key_tensor,
        value_tensor,
        attention_mask,
        head_mask,
        output_attentions,
        training=...,
    ):  # -> tuple[Any, Any] | tuple[Any]:
        ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFMobileBertSelfOutput(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
    def call(self, hidden_states, residual_tensor, training=...): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFMobileBertAttention(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
    def prune_heads(self, heads): ...
    def call(
        self,
        query_tensor,
        key_tensor,
        value_tensor,
        layer_input,
        attention_mask,
        head_mask,
        output_attentions,
        training=...,
    ): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFOutputBottleneck(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
    def call(self, hidden_states, residual_tensor, training=...): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFMobileBertOutput(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
    def call(
        self, hidden_states, residual_tensor_1, residual_tensor_2, training=...
    ): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFBottleneckLayer(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
    def call(self, inputs): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFBottleneck(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
    def call(self, hidden_states):  # -> tuple[Any, ...] | tuple[Any, Any, Any, Any]:
        ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFFFNOutput(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
    def call(self, hidden_states, residual_tensor): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFFFNLayer(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
    def call(self, hidden_states): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFMobileBertLayer(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
    def call(
        self, hidden_states, attention_mask, head_mask, output_attentions, training=...
    ): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFMobileBertEncoder(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
    def call(
        self,
        hidden_states,
        attention_mask,
        head_mask,
        output_attentions,
        output_hidden_states,
        return_dict,
        training=...,
    ):  # -> tuple[Any | tuple[Any, ...] | tuple[()], ...] | TFBaseModelOutput:
        ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFMobileBertPooler(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
    def call(self, hidden_states): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFMobileBertPredictionHeadTransform(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
    def call(self, hidden_states): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFMobileBertLMPredictionHead(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
    def build(self, input_shape=...):  # -> None:
        ...
    def get_output_embeddings(self):  # -> Self:
        ...
    def set_output_embeddings(self, value):  # -> None:
        ...
    def get_bias(self):  # -> dict[str, Any]:
        ...
    def set_bias(self, value):  # -> None:
        ...
    def call(self, hidden_states): ...

class TFMobileBertMLMHead(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
    def call(self, sequence_output): ...
    def build(self, input_shape=...):  # -> None:
        ...

@keras_serializable
class TFMobileBertMainLayer(keras.layers.Layer):
    config_class = MobileBertConfig
    def __init__(self, config, add_pooling_layer=..., **kwargs) -> None: ...
    def get_input_embeddings(self):  # -> TFMobileBertEmbeddings:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @unpack_inputs
    def call(
        self,
        input_ids=...,
        attention_mask=...,
        token_type_ids=...,
        position_ids=...,
        head_mask=...,
        inputs_embeds=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
        training=...,
    ):  # -> TFBaseModelOutputWithPooling:
        ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFMobileBertPreTrainedModel(TFPreTrainedModel):
    config_class = MobileBertConfig
    base_model_prefix = ...

@dataclass
class TFMobileBertForPreTrainingOutput(ModelOutput):
    loss: tf.Tensor | None = ...
    prediction_logits: tf.Tensor | None = ...
    seq_relationship_logits: tf.Tensor | None = ...
    hidden_states: tuple[tf.Tensor] | None = ...
    attentions: tuple[tf.Tensor] | None = ...

MOBILEBERT_START_DOCSTRING = ...
MOBILEBERT_INPUTS_DOCSTRING = ...

@add_start_docstrings(
    ...,
    MOBILEBERT_START_DOCSTRING,
)
class TFMobileBertModel(TFMobileBertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
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
    ) -> tuple | TFBaseModelOutputWithPooling: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    MOBILEBERT_START_DOCSTRING,
)
class TFMobileBertForPreTraining(
    TFMobileBertPreTrainedModel, TFMobileBertPreTrainingLoss
):
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    def get_lm_head(self):  # -> TFMobileBertLMPredictionHead:
        ...
    def get_prefix_bias_name(self): ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @replace_return_docstrings(
        output_type=TFMobileBertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC
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
        next_sentence_label: np.ndarray | tf.Tensor | None = ...,
        training: bool | None = ...,
    ) -> tuple | TFMobileBertForPreTrainingOutput: ...
    def build(self, input_shape=...):  # -> None:
        ...
    def tf_to_pt_weight_rename(
        self, tf_weight
    ):  # -> tuple[Literal['cls.predictions.decoder.weight'], Literal['mobilebert.embeddings.word_embeddings.weight']] | tuple[Any]:
        ...

@add_start_docstrings(
    ...,
    MOBILEBERT_START_DOCSTRING,
)
class TFMobileBertForMaskedLM(
    TFMobileBertPreTrainedModel, TFMaskedLanguageModelingLoss
):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    def get_lm_head(self):  # -> TFMobileBertLMPredictionHead:
        ...
    def get_prefix_bias_name(self): ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFMaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="'paris'",
        expected_loss=0.57,
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
    ) -> tuple | TFMaskedLMOutput: ...
    def build(self, input_shape=...):  # -> None:
        ...
    def tf_to_pt_weight_rename(
        self, tf_weight
    ):  # -> tuple[Literal['cls.predictions.decoder.weight'], Literal['mobilebert.embeddings.word_embeddings.weight']] | tuple[Any]:
        ...

class TFMobileBertOnlyNSPHead(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
    def call(self, pooled_output): ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    MOBILEBERT_START_DOCSTRING,
)
class TFMobileBertForNextSentencePrediction(
    TFMobileBertPreTrainedModel, TFNextSentencePredictionLoss
):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @replace_return_docstrings(
        output_type=TFNextSentencePredictorOutput, config_class=_CONFIG_FOR_DOC
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
        next_sentence_label: np.ndarray | tf.Tensor | None = ...,
        training: bool | None = ...,
    ) -> tuple | TFNextSentencePredictorOutput: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    MOBILEBERT_START_DOCSTRING,
)
class TFMobileBertForSequenceClassification(
    TFMobileBertPreTrainedModel, TFSequenceClassificationLoss
):
    _keys_to_ignore_on_load_unexpected = ...
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION,
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_SEQ_CLASS_EXPECTED_OUTPUT,
        expected_loss=_SEQ_CLASS_EXPECTED_LOSS,
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
    ) -> tuple | TFSequenceClassifierOutput: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    MOBILEBERT_START_DOCSTRING,
)
class TFMobileBertForQuestionAnswering(
    TFMobileBertPreTrainedModel, TFQuestionAnsweringLoss
):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_QA,
        output_type=TFQuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
        qa_target_start_index=_QA_TARGET_START_INDEX,
        qa_target_end_index=_QA_TARGET_END_INDEX,
        expected_output=_QA_EXPECTED_OUTPUT,
        expected_loss=_QA_EXPECTED_LOSS,
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
    ) -> tuple | TFQuestionAnsweringModelOutput: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    MOBILEBERT_START_DOCSTRING,
)
class TFMobileBertForMultipleChoice(TFMobileBertPreTrainedModel, TFMultipleChoiceLoss):
    _keys_to_ignore_on_load_unexpected = ...
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
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
    ) -> tuple | TFMultipleChoiceModelOutput: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    MOBILEBERT_START_DOCSTRING,
)
class TFMobileBertForTokenClassification(
    TFMobileBertPreTrainedModel, TFTokenClassificationLoss
):
    _keys_to_ignore_on_load_unexpected = ...
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_TOKEN_CLASSIFICATION,
        output_type=TFTokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_TOKEN_CLASS_EXPECTED_OUTPUT,
        expected_loss=_TOKEN_CLASS_EXPECTED_LOSS,
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
    ) -> tuple | TFTokenClassifierOutput: ...
    def build(self, input_shape=...):  # -> None:
        ...

__all__ = [
    "TFMobileBertForMaskedLM",
    "TFMobileBertForMultipleChoice",
    "TFMobileBertForNextSentencePrediction",
    "TFMobileBertForPreTraining",
    "TFMobileBertForQuestionAnswering",
    "TFMobileBertForSequenceClassification",
    "TFMobileBertForTokenClassification",
    "TFMobileBertMainLayer",
    "TFMobileBertModel",
    "TFMobileBertPreTrainedModel",
]
