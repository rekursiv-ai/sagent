import numpy as np
import tensorflow as tf

from .configuration_convbert import ConvBertConfig
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
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
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)

"""TF 2.0 ConvBERT model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...

class TFConvBertEmbeddings(keras.layers.Layer):
    def __init__(self, config: ConvBertConfig, **kwargs) -> None: ...
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

class TFConvBertSelfAttention(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
    def transpose_for_scores(self, x, batch_size): ...
    def call(
        self, hidden_states, attention_mask, head_mask, output_attentions, training=...
    ):  # -> tuple[Any, Any] | tuple[Any]:
        ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFConvBertSelfOutput(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
    def call(self, hidden_states, input_tensor, training=...): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFConvBertAttention(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
    def prune_heads(self, heads): ...
    def call(
        self, input_tensor, attention_mask, head_mask, output_attentions, training=...
    ): ...
    def build(self, input_shape=...):  # -> None:
        ...

class GroupedLinearLayer(keras.layers.Layer):
    def __init__(
        self, input_size, output_size, num_groups, kernel_initializer, **kwargs
    ) -> None: ...
    def build(self, input_shape=...):  # -> None:
        ...
    def call(self, hidden_states): ...

class TFConvBertIntermediate(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
    def call(self, hidden_states): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFConvBertOutput(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
    def call(self, hidden_states, input_tensor, training=...): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFConvBertLayer(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
    def call(
        self, hidden_states, attention_mask, head_mask, output_attentions, training=...
    ): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFConvBertEncoder(keras.layers.Layer):
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

class TFConvBertPredictionHeadTransform(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
    def call(self, hidden_states): ...
    def build(self, input_shape=...):  # -> None:
        ...

@keras_serializable
class TFConvBertMainLayer(keras.layers.Layer):
    config_class = ConvBertConfig
    def __init__(self, config, **kwargs) -> None: ...
    def get_input_embeddings(self):  # -> TFConvBertEmbeddings:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def get_extended_attention_mask(self, attention_mask, input_shape, dtype): ...
    def get_head_mask(self, head_mask): ...
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
    ): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFConvBertPreTrainedModel(TFPreTrainedModel):
    config_class = ConvBertConfig
    base_model_prefix = ...

CONVBERT_START_DOCSTRING = ...
CONVBERT_INPUTS_DOCSTRING = ...

@add_start_docstrings(
    ...,
    CONVBERT_START_DOCSTRING,
)
class TFConvBertModel(TFConvBertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        CONVBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.array | tf.Tensor | None = ...,
        token_type_ids: np.array | tf.Tensor | None = ...,
        position_ids: np.array | tf.Tensor | None = ...,
        head_mask: np.array | tf.Tensor | None = ...,
        inputs_embeds: tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> TFBaseModelOutput | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFConvBertMaskedLMHead(keras.layers.Layer):
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

class TFConvBertGeneratorPredictions(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
    def call(self, generator_hidden_states, training=...): ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    CONVBERT_START_DOCSTRING,
)
class TFConvBertForMaskedLM(TFConvBertPreTrainedModel, TFMaskedLanguageModelingLoss):
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    def get_lm_head(self):  # -> TFConvBertMaskedLMHead:
        ...
    def get_prefix_bias_name(self): ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        CONVBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFMaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: tf.Tensor | None = ...,
        training: bool | None = ...,
    ) -> tuple | TFMaskedLMOutput: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFConvBertClassificationHead(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
    def call(self, hidden_states, **kwargs): ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    CONVBERT_START_DOCSTRING,
)
class TFConvBertForSequenceClassification(
    TFConvBertPreTrainedModel, TFSequenceClassificationLoss
):
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        CONVBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
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
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: tf.Tensor | None = ...,
        training: bool | None = ...,
    ) -> tuple | TFSequenceClassifierOutput: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    CONVBERT_START_DOCSTRING,
)
class TFConvBertForMultipleChoice(TFConvBertPreTrainedModel, TFMultipleChoiceLoss):
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        CONVBERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
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
        inputs_embeds: tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: tf.Tensor | None = ...,
        training: bool | None = ...,
    ) -> tuple | TFMultipleChoiceModelOutput: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    CONVBERT_START_DOCSTRING,
)
class TFConvBertForTokenClassification(
    TFConvBertPreTrainedModel, TFTokenClassificationLoss
):
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        CONVBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
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
        inputs_embeds: tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: tf.Tensor | None = ...,
        training: bool | None = ...,
    ) -> tuple | TFTokenClassifierOutput: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    CONVBERT_START_DOCSTRING,
)
class TFConvBertForQuestionAnswering(
    TFConvBertPreTrainedModel, TFQuestionAnsweringLoss
):
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        CONVBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
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
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        start_positions: tf.Tensor | None = ...,
        end_positions: tf.Tensor | None = ...,
        training: bool | None = ...,
    ) -> tuple | TFQuestionAnsweringModelOutput: ...
    def build(self, input_shape=...):  # -> None:
        ...

__all__ = [
    "TFConvBertForMaskedLM",
    "TFConvBertForMultipleChoice",
    "TFConvBertForQuestionAnswering",
    "TFConvBertForSequenceClassification",
    "TFConvBertForTokenClassification",
    "TFConvBertLayer",
    "TFConvBertModel",
    "TFConvBertPreTrainedModel",
]
