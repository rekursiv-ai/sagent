from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from .configuration_electra import ElectraConfig
from ...modeling_tf_outputs import (
    TFBaseModelOutputWithPastAndCrossAttentions,
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

"""TF Electra model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...

class TFElectraSelfAttention(keras.layers.Layer):
    def __init__(self, config: ElectraConfig, **kwargs) -> None: ...
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

class TFElectraSelfOutput(keras.layers.Layer):
    def __init__(self, config: ElectraConfig, **kwargs) -> None: ...
    def call(
        self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = ...
    ) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFElectraAttention(keras.layers.Layer):
    def __init__(self, config: ElectraConfig, **kwargs) -> None: ...
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

class TFElectraIntermediate(keras.layers.Layer):
    def __init__(self, config: ElectraConfig, **kwargs) -> None: ...
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFElectraOutput(keras.layers.Layer):
    def __init__(self, config: ElectraConfig, **kwargs) -> None: ...
    def call(
        self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = ...
    ) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFElectraLayer(keras.layers.Layer):
    def __init__(self, config: ElectraConfig, **kwargs) -> None: ...
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

class TFElectraEncoder(keras.layers.Layer):
    def __init__(self, config: ElectraConfig, **kwargs) -> None: ...
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

class TFElectraPooler(keras.layers.Layer):
    def __init__(self, config: ElectraConfig, **kwargs) -> None: ...
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFElectraEmbeddings(keras.layers.Layer):
    def __init__(self, config: ElectraConfig, **kwargs) -> None: ...
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

class TFElectraDiscriminatorPredictions(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
    def call(self, discriminator_hidden_states, training=...): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFElectraGeneratorPredictions(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
    def call(self, generator_hidden_states, training=...): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFElectraPreTrainedModel(TFPreTrainedModel):
    config_class = ElectraConfig
    base_model_prefix = ...
    _keys_to_ignore_on_load_unexpected = ...
    _keys_to_ignore_on_load_missing = ...

@keras_serializable
class TFElectraMainLayer(keras.layers.Layer):
    config_class = ElectraConfig
    def __init__(self, config, **kwargs) -> None: ...
    def get_input_embeddings(self):  # -> TFElectraEmbeddings:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def get_extended_attention_mask(
        self, attention_mask, input_shape, dtype, past_key_values_length=...
    ): ...
    def get_head_mask(self, head_mask): ...
    @unpack_inputs
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        encoder_hidden_states: np.ndarray | tf.Tensor | None = ...,
        encoder_attention_mask: np.ndarray | tf.Tensor | None = ...,
        past_key_values: tuple[tuple[np.ndarray | tf.Tensor]] | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool | None = ...,
    ) -> TFBaseModelOutputWithPastAndCrossAttentions | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

@dataclass
class TFElectraForPreTrainingOutput(ModelOutput):
    logits: tf.Tensor | None = ...
    hidden_states: tuple[tf.Tensor] | None = ...
    attentions: tuple[tf.Tensor] | None = ...

ELECTRA_START_DOCSTRING = ...
ELECTRA_INPUTS_DOCSTRING = ...

@add_start_docstrings(
    ...,
    ELECTRA_START_DOCSTRING,
)
class TFElectraModel(TFElectraPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        ELECTRA_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPastAndCrossAttentions,
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
        encoder_hidden_states: np.ndarray | tf.Tensor | None = ...,
        encoder_attention_mask: np.ndarray | tf.Tensor | None = ...,
        past_key_values: tuple[tuple[np.ndarray | tf.Tensor]] | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool | None = ...,
    ) -> TFBaseModelOutputWithPastAndCrossAttentions | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    ELECTRA_START_DOCSTRING,
)
class TFElectraForPreTraining(TFElectraPreTrainedModel):
    def __init__(self, config, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        ELECTRA_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @replace_return_docstrings(
        output_type=TFElectraForPreTrainingOutput, config_class=_CONFIG_FOR_DOC
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
    ) -> TFElectraForPreTrainingOutput | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFElectraMaskedLMHead(keras.layers.Layer):
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

@add_start_docstrings(
    ...,
    ELECTRA_START_DOCSTRING,
)
class TFElectraForMaskedLM(TFElectraPreTrainedModel, TFMaskedLanguageModelingLoss):
    def __init__(self, config, **kwargs) -> None: ...
    def get_lm_head(self):  # -> TFElectraMaskedLMHead:
        ...
    def get_prefix_bias_name(self): ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        ELECTRA_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint="google/electra-small-generator",
        output_type=TFMaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        mask="[MASK]",
        expected_output="'paris'",
        expected_loss=1.22,
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

class TFElectraClassificationHead(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
    def call(self, inputs, **kwargs): ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    ELECTRA_START_DOCSTRING,
)
class TFElectraForSequenceClassification(
    TFElectraPreTrainedModel, TFSequenceClassificationLoss
):
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        ELECTRA_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint="bhadresh-savani/electra-base-emotion",
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="'joy'",
        expected_loss=0.06,
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
    ELECTRA_START_DOCSTRING,
)
class TFElectraForMultipleChoice(TFElectraPreTrainedModel, TFMultipleChoiceLoss):
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        ELECTRA_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
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

@add_start_docstrings(
    ...,
    ELECTRA_START_DOCSTRING,
)
class TFElectraForTokenClassification(
    TFElectraPreTrainedModel, TFTokenClassificationLoss
):
    def __init__(self, config, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        ELECTRA_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=...,
        output_type=TFTokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=...,
        expected_loss=0.11,
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
    ELECTRA_START_DOCSTRING,
)
class TFElectraForQuestionAnswering(TFElectraPreTrainedModel, TFQuestionAnsweringLoss):
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        ELECTRA_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint="bhadresh-savani/electra-base-squad2",
        output_type=TFQuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
        qa_target_start_index=11,
        qa_target_end_index=12,
        expected_output="'a nice puppet'",
        expected_loss=2.64,
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

__all__ = [
    "TFElectraForMaskedLM",
    "TFElectraForMultipleChoice",
    "TFElectraForPreTraining",
    "TFElectraForQuestionAnswering",
    "TFElectraForSequenceClassification",
    "TFElectraForTokenClassification",
    "TFElectraModel",
    "TFElectraPreTrainedModel",
]
