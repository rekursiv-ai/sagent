from dataclasses import dataclass

import enum

import numpy as np
import tensorflow as tf

from .configuration_tapas import TapasConfig
from ...modeling_tf_outputs import (
    TFBaseModelOutputWithPastAndCrossAttentions,
    TFBaseModelOutputWithPooling,
    TFMaskedLMOutput,
    TFSequenceClassifierOutput,
)
from ...modeling_tf_utils import (
    TFMaskedLanguageModelingLoss,
    TFModelInputType,
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_tensorflow_probability_available,
    replace_return_docstrings,
)

"""TF 2.0 TAPAS model."""
logger = ...
if is_tensorflow_probability_available():
    n = ...
else:
    _ = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
EPSILON_ZERO_DIVISION = ...
CLOSE_ENOUGH_TO_LOG_ZERO = ...

@dataclass
class TFTableQuestionAnsweringOutput(ModelOutput):
    loss: tf.Tensor | None = ...
    logits: tf.Tensor | None = ...
    logits_aggregation: tf.Tensor | None = ...
    hidden_states: tuple[tf.Tensor] | None = ...
    attentions: tuple[tf.Tensor] | None = ...

class TFTapasEmbeddings(keras.layers.Layer):
    def __init__(self, config: TapasConfig, **kwargs) -> None: ...
    def build(self, input_shape=...):  # -> None:
        ...
    def call(
        self,
        input_ids: tf.Tensor | None = ...,
        position_ids: tf.Tensor | None = ...,
        token_type_ids: tf.Tensor | None = ...,
        inputs_embeds: tf.Tensor | None = ...,
        training: bool = ...,
    ) -> tf.Tensor: ...

class TFTapasSelfAttention(keras.layers.Layer):
    def __init__(self, config: TapasConfig, **kwargs) -> None: ...
    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        encoder_hidden_states: tf.Tensor,
        encoder_attention_mask: tf.Tensor,
        past_key_value: tuple[tf.Tensor],
        output_attentions: bool,
        training: bool = ...,
    ) -> tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFTapasSelfOutput(keras.layers.Layer):
    def __init__(self, config: TapasConfig, **kwargs) -> None: ...
    def call(
        self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = ...
    ) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFTapasAttention(keras.layers.Layer):
    def __init__(self, config: TapasConfig, **kwargs) -> None: ...
    def prune_heads(self, heads): ...
    def call(
        self,
        input_tensor: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        encoder_hidden_states: tf.Tensor,
        encoder_attention_mask: tf.Tensor,
        past_key_value: tuple[tf.Tensor],
        output_attentions: bool,
        training: bool = ...,
    ) -> tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFTapasIntermediate(keras.layers.Layer):
    def __init__(self, config: TapasConfig, **kwargs) -> None: ...
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFTapasOutput(keras.layers.Layer):
    def __init__(self, config: TapasConfig, **kwargs) -> None: ...
    def call(
        self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = ...
    ) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFTapasLayer(keras.layers.Layer):
    def __init__(self, config: TapasConfig, **kwargs) -> None: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        encoder_hidden_states: tf.Tensor | None,
        encoder_attention_mask: tf.Tensor | None,
        past_key_value: tuple[tf.Tensor] | None,
        output_attentions: bool,
        training: bool = ...,
    ) -> tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFTapasEncoder(keras.layers.Layer):
    def __init__(self, config: TapasConfig, **kwargs) -> None: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        encoder_hidden_states: tf.Tensor | None,
        encoder_attention_mask: tf.Tensor | None,
        past_key_values: tuple[tuple[tf.Tensor]] | None,
        use_cache: bool | None,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        training: bool = ...,
    ) -> TFBaseModelOutputWithPastAndCrossAttentions | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFTapasPooler(keras.layers.Layer):
    def __init__(self, config: TapasConfig, **kwargs) -> None: ...
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFTapasPredictionHeadTransform(keras.layers.Layer):
    def __init__(self, config: TapasConfig, **kwargs) -> None: ...
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFTapasLMPredictionHead(keras.layers.Layer):
    def __init__(
        self, config: TapasConfig, input_embeddings: keras.layers.Layer, **kwargs
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

class TFTapasMLMHead(keras.layers.Layer):
    def __init__(
        self, config: TapasConfig, input_embeddings: keras.layers.Layer, **kwargs
    ) -> None: ...
    def call(self, sequence_output: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

@keras_serializable
class TFTapasMainLayer(keras.layers.Layer):
    config_class = TapasConfig
    def __init__(
        self, config: TapasConfig, add_pooling_layer: bool = ..., **kwargs
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

class TFTapasPreTrainedModel(TFPreTrainedModel):
    config_class = TapasConfig
    base_model_prefix = ...
    @property
    def input_signature(self):  # -> dict[str, Any]:
        ...

TAPAS_START_DOCSTRING = ...
TAPAS_INPUTS_DOCSTRING = ...

@add_start_docstrings(
    ...,
    TAPAS_START_DOCSTRING,
)
class TFTapasModel(TFTapasPreTrainedModel):
    def __init__(self, config: TapasConfig, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        TAPAS_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @replace_return_docstrings(
        output_type=TFBaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC
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

@add_start_docstrings(..., TAPAS_START_DOCSTRING)
class TFTapasForMaskedLM(TFTapasPreTrainedModel, TFMaskedLanguageModelingLoss):
    def __init__(self, config: TapasConfig, *inputs, **kwargs) -> None: ...
    def get_lm_head(self) -> keras.layers.Layer: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        TAPAS_INPUTS_DOCSTRING.format("batch_size, sequence_length")
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

class TFTapasComputeTokenLogits(keras.layers.Layer):
    def __init__(self, config: TapasConfig, **kwargs) -> None: ...
    def call(self, sequence_output: tf.Tensor) -> tf.Tensor: ...

class TFTapasComputeColumnLogits(keras.layers.Layer):
    def __init__(self, config: TapasConfig, **kwargs) -> None: ...
    def call(
        self, sequence_output, cell_index, cell_mask, allow_empty_column_selection
    ) -> tf.Tensor: ...

@add_start_docstrings(
    ...,
    TAPAS_START_DOCSTRING,
)
class TFTapasForQuestionAnswering(TFTapasPreTrainedModel):
    def __init__(self, config: TapasConfig, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        TAPAS_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @replace_return_docstrings(
        output_type=TFTableQuestionAnsweringOutput, config_class=_CONFIG_FOR_DOC
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        table_mask: np.ndarray | tf.Tensor | None = ...,
        aggregation_labels: np.ndarray | tf.Tensor | None = ...,
        float_answer: np.ndarray | tf.Tensor | None = ...,
        numeric_values: np.ndarray | tf.Tensor | None = ...,
        numeric_values_scale: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: np.ndarray | tf.Tensor | None = ...,
        training: bool | None = ...,
    ) -> TFTableQuestionAnsweringOutput | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    TAPAS_START_DOCSTRING,
)
class TFTapasForSequenceClassification(
    TFTapasPreTrainedModel, TFSequenceClassificationLoss
):
    def __init__(self, config: TapasConfig, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        TAPAS_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
    )
    @replace_return_docstrings(
        output_type=TFSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC
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

class AverageApproximationFunction(enum.StrEnum):
    RATIO = ...
    FIRST_ORDER = ...
    SECOND_ORDER = ...

class IndexMap:
    def __init__(self, indices, num_segments, batch_dims=...) -> None: ...
    def batch_shape(self): ...

class ProductIndexMap(IndexMap):
    def __init__(self, outer_index, inner_index) -> None: ...
    def project_outer(self, index):  # -> IndexMap:
        ...
    def project_inner(self, index):  # -> IndexMap:
        ...

def gather(values, index, name=...): ...
def flatten(index, name=...):  # -> IndexMap:
    ...
def range_index_map(batch_shape, num_segments, name=...):  # -> IndexMap:
    ...
def reduce_mean(values, index, name=...):  # -> tuple[Any, IndexMap]:
    ...
def reduce_sum(values, index, name=...):  # -> tuple[Any, IndexMap]:
    ...
def reduce_max(values, index, name=...):  # -> tuple[Any, IndexMap]:
    ...
def reduce_min(values, index, name=...):  # -> tuple[Any, IndexMap]:
    ...

__all__ = [
    "TFTapasForMaskedLM",
    "TFTapasForQuestionAnswering",
    "TFTapasForSequenceClassification",
    "TFTapasModel",
    "TFTapasPreTrainedModel",
]
