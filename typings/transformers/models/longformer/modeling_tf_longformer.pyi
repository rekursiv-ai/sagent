from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from .configuration_longformer import LongformerConfig
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
)

"""Tensorflow Longformer model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
LARGE_NEGATIVE = ...

@dataclass
class TFLongformerBaseModelOutput(ModelOutput):
    last_hidden_state: tf.Tensor | None = ...
    hidden_states: tuple[tf.Tensor, ...] | None = ...
    attentions: tuple[tf.Tensor, ...] | None = ...
    global_attentions: tuple[tf.Tensor, ...] | None = ...

@dataclass
class TFLongformerBaseModelOutputWithPooling(ModelOutput):
    last_hidden_state: tf.Tensor | None = ...
    pooler_output: tf.Tensor | None = ...
    hidden_states: tuple[tf.Tensor, ...] | None = ...
    attentions: tuple[tf.Tensor, ...] | None = ...
    global_attentions: tuple[tf.Tensor, ...] | None = ...

@dataclass
class TFLongformerMaskedLMOutput(ModelOutput):
    loss: tf.Tensor | None = ...
    logits: tf.Tensor | None = ...
    hidden_states: tuple[tf.Tensor, ...] | None = ...
    attentions: tuple[tf.Tensor, ...] | None = ...
    global_attentions: tuple[tf.Tensor, ...] | None = ...

@dataclass
class TFLongformerQuestionAnsweringModelOutput(ModelOutput):
    loss: tf.Tensor | None = ...
    start_logits: tf.Tensor | None = ...
    end_logits: tf.Tensor | None = ...
    hidden_states: tuple[tf.Tensor, ...] | None = ...
    attentions: tuple[tf.Tensor, ...] | None = ...
    global_attentions: tuple[tf.Tensor, ...] | None = ...

@dataclass
class TFLongformerSequenceClassifierOutput(ModelOutput):
    loss: tf.Tensor | None = ...
    logits: tf.Tensor | None = ...
    hidden_states: tuple[tf.Tensor, ...] | None = ...
    attentions: tuple[tf.Tensor, ...] | None = ...
    global_attentions: tuple[tf.Tensor, ...] | None = ...

@dataclass
class TFLongformerMultipleChoiceModelOutput(ModelOutput):
    loss: tf.Tensor | None = ...
    logits: tf.Tensor | None = ...
    hidden_states: tuple[tf.Tensor, ...] | None = ...
    attentions: tuple[tf.Tensor, ...] | None = ...
    global_attentions: tuple[tf.Tensor, ...] | None = ...

@dataclass
class TFLongformerTokenClassifierOutput(ModelOutput):
    loss: tf.Tensor | None = ...
    logits: tf.Tensor | None = ...
    hidden_states: tuple[tf.Tensor, ...] | None = ...
    attentions: tuple[tf.Tensor, ...] | None = ...
    global_attentions: tuple[tf.Tensor, ...] | None = ...

class TFLongformerLMHead(keras.layers.Layer):
    def __init__(self, config, input_embeddings, **kwargs) -> None: ...
    def build(self, input_shape=...):  # -> None:
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

class TFLongformerEmbeddings(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
    def build(self, input_shape=...):  # -> None:
        ...
    def create_position_ids_from_input_ids(
        self, input_ids, past_key_values_length=...
    ): ...
    def call(
        self,
        input_ids=...,
        position_ids=...,
        token_type_ids=...,
        inputs_embeds=...,
        past_key_values_length=...,
        training=...,
    ): ...

class TFLongformerIntermediate(keras.layers.Layer):
    def __init__(self, config: LongformerConfig, **kwargs) -> None: ...
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFLongformerOutput(keras.layers.Layer):
    def __init__(self, config: LongformerConfig, **kwargs) -> None: ...
    def call(
        self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = ...
    ) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFLongformerPooler(keras.layers.Layer):
    def __init__(self, config: LongformerConfig, **kwargs) -> None: ...
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFLongformerSelfOutput(keras.layers.Layer):
    def __init__(self, config: LongformerConfig, **kwargs) -> None: ...
    def call(
        self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = ...
    ) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFLongformerSelfAttention(keras.layers.Layer):
    def __init__(self, config, layer_id, **kwargs) -> None: ...
    def build(self, input_shape=...):  # -> None:
        ...
    def call(self, inputs, training=...):  # -> tuple[Any, Any, Any]:
        ...
    def reshape_and_transpose(self, vector, batch_size): ...

class TFLongformerAttention(keras.layers.Layer):
    def __init__(self, config, layer_id=..., **kwargs) -> None: ...
    def prune_heads(self, heads): ...
    def call(self, inputs, training=...): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFLongformerLayer(keras.layers.Layer):
    def __init__(self, config, layer_id=..., **kwargs) -> None: ...
    def call(self, inputs, training=...): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFLongformerEncoder(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
    def call(
        self,
        hidden_states,
        attention_mask=...,
        head_mask=...,
        padding_len=...,
        is_index_masked=...,
        is_index_global_attn=...,
        is_global_attn=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
        training=...,
    ):  # -> tuple[Any | tuple[Any, ...] | tuple[()], ...] | TFLongformerBaseModelOutput:
        ...
    def build(self, input_shape=...):  # -> None:
        ...

@keras_serializable
class TFLongformerMainLayer(keras.layers.Layer):
    config_class = LongformerConfig
    def __init__(self, config, add_pooling_layer=..., **kwargs) -> None: ...
    def get_input_embeddings(self):  # -> TFLongformerEmbeddings:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @unpack_inputs
    def call(
        self,
        input_ids=...,
        attention_mask=...,
        head_mask=...,
        global_attention_mask=...,
        token_type_ids=...,
        position_ids=...,
        inputs_embeds=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
        training=...,
    ):  # -> TFLongformerBaseModelOutputWithPooling:
        ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFLongformerPreTrainedModel(TFPreTrainedModel):
    config_class = LongformerConfig
    base_model_prefix = ...
    @property
    def input_signature(self):  # -> dict[str, Any]:
        ...

LONGFORMER_START_DOCSTRING = ...
LONGFORMER_INPUTS_DOCSTRING = ...

@add_start_docstrings(
    ...,
    LONGFORMER_START_DOCSTRING,
)
class TFLongformerModel(TFLongformerPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        LONGFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        global_attention_mask: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool | None = ...,
    ) -> TFLongformerBaseModelOutputWithPooling | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    LONGFORMER_START_DOCSTRING,
)
class TFLongformerForMaskedLM(
    TFLongformerPreTrainedModel, TFMaskedLanguageModelingLoss
):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    def get_lm_head(self):  # -> TFLongformerLMHead:
        ...
    def get_prefix_bias_name(self): ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        LONGFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint="allenai/longformer-base-4096",
        output_type=TFLongformerMaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        mask="<mask>",
        expected_output="' Paris'",
        expected_loss=0.44,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        global_attention_mask: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: np.ndarray | tf.Tensor | None = ...,
        training: bool | None = ...,
    ) -> TFLongformerMaskedLMOutput | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    LONGFORMER_START_DOCSTRING,
)
class TFLongformerForQuestionAnswering(
    TFLongformerPreTrainedModel, TFQuestionAnsweringLoss
):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        LONGFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint="allenai/longformer-large-4096-finetuned-triviaqa",
        output_type=TFLongformerQuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="' puppet'",
        expected_loss=0.96,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        global_attention_mask: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        start_positions: np.ndarray | tf.Tensor | None = ...,
        end_positions: np.ndarray | tf.Tensor | None = ...,
        training: bool | None = ...,
    ) -> TFLongformerQuestionAnsweringModelOutput | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFLongformerClassificationHead(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
    def call(self, hidden_states, training=...): ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    LONGFORMER_START_DOCSTRING,
)
class TFLongformerForSequenceClassification(
    TFLongformerPreTrainedModel, TFSequenceClassificationLoss
):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        LONGFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFLongformerSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        global_attention_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: np.ndarray | tf.Tensor | None = ...,
        training: bool | None = ...,
    ) -> TFLongformerSequenceClassifierOutput | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    LONGFORMER_START_DOCSTRING,
)
class TFLongformerForMultipleChoice(TFLongformerPreTrainedModel, TFMultipleChoiceLoss):
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    @property
    def input_signature(self):  # -> dict[str, Any]:
        ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        LONGFORMER_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFLongformerMultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        global_attention_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: np.ndarray | tf.Tensor | None = ...,
        training: bool | None = ...,
    ) -> TFLongformerMultipleChoiceModelOutput | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    LONGFORMER_START_DOCSTRING,
)
class TFLongformerForTokenClassification(
    TFLongformerPreTrainedModel, TFTokenClassificationLoss
):
    _keys_to_ignore_on_load_unexpected = ...
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        LONGFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFLongformerTokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        global_attention_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: np.array | tf.Tensor | None = ...,
        training: bool | None = ...,
    ) -> TFLongformerTokenClassifierOutput | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

__all__ = [
    "TFLongformerForMaskedLM",
    "TFLongformerForMultipleChoice",
    "TFLongformerForQuestionAnswering",
    "TFLongformerForSequenceClassification",
    "TFLongformerForTokenClassification",
    "TFLongformerModel",
    "TFLongformerPreTrainedModel",
    "TFLongformerSelfAttention",
]
