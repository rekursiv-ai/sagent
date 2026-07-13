import numpy as np
import tensorflow as tf

from .configuration_roberta import RobertaConfig
from ...modeling_tf_outputs import (
    TFBaseModelOutputWithPastAndCrossAttentions,
    TFBaseModelOutputWithPoolingAndCrossAttentions,
    TFCausalLMOutputWithCrossAttentions,
    TFMaskedLMOutput,
    TFMultipleChoiceModelOutput,
    TFQuestionAnsweringModelOutput,
    TFSequenceClassifierOutput,
    TFTokenClassifierOutput,
)
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
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

"""TF 2.0 RoBERTa model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...

class TFRobertaEmbeddings(keras.layers.Layer):
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

class TFRobertaPooler(keras.layers.Layer):
    def __init__(self, config: RobertaConfig, **kwargs) -> None: ...
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFRobertaSelfAttention(keras.layers.Layer):
    def __init__(self, config: RobertaConfig, **kwargs) -> None: ...
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

class TFRobertaSelfOutput(keras.layers.Layer):
    def __init__(self, config: RobertaConfig, **kwargs) -> None: ...
    def call(
        self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = ...
    ) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFRobertaAttention(keras.layers.Layer):
    def __init__(self, config: RobertaConfig, **kwargs) -> None: ...
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

class TFRobertaIntermediate(keras.layers.Layer):
    def __init__(self, config: RobertaConfig, **kwargs) -> None: ...
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFRobertaOutput(keras.layers.Layer):
    def __init__(self, config: RobertaConfig, **kwargs) -> None: ...
    def call(
        self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = ...
    ) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFRobertaLayer(keras.layers.Layer):
    def __init__(self, config: RobertaConfig, **kwargs) -> None: ...
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

class TFRobertaEncoder(keras.layers.Layer):
    def __init__(self, config: RobertaConfig, **kwargs) -> None: ...
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

@keras_serializable
class TFRobertaMainLayer(keras.layers.Layer):
    config_class = RobertaConfig
    def __init__(self, config, add_pooling_layer=..., **kwargs) -> None: ...
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
        encoder_hidden_states: np.ndarray | tf.Tensor | None = ...,
        encoder_attention_mask: np.ndarray | tf.Tensor | None = ...,
        past_key_values: tuple[tuple[np.ndarray | tf.Tensor]] | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> TFBaseModelOutputWithPoolingAndCrossAttentions | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFRobertaPreTrainedModel(TFPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = ...

ROBERTA_START_DOCSTRING = ...
ROBERTA_INPUTS_DOCSTRING = ...

@add_start_docstrings(
    ...,
    ROBERTA_START_DOCSTRING,
)
class TFRobertaModel(TFRobertaPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPoolingAndCrossAttentions,
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
    ) -> tuple | TFBaseModelOutputWithPoolingAndCrossAttentions: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFRobertaLMHead(keras.layers.Layer):
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

@add_start_docstrings(..., ROBERTA_START_DOCSTRING)
class TFRobertaForMaskedLM(TFRobertaPreTrainedModel, TFMaskedLanguageModelingLoss):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    def get_lm_head(self):  # -> TFRobertaLMHead:
        ...
    def get_prefix_bias_name(self): ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFMaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        mask="<mask>",
        expected_output="' Paris'",
        expected_loss=0.1,
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

class TFRobertaForCausalLM(TFRobertaPreTrainedModel, TFCausalLanguageModelingLoss):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config: RobertaConfig, *inputs, **kwargs) -> None: ...
    def get_lm_head(self):  # -> TFRobertaLMHead:
        ...
    def get_prefix_bias_name(self): ...
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=..., attention_mask=..., **model_kwargs
    ):  # -> dict[str, Any | None]:
        ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFCausalLMOutputWithCrossAttentions,
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
        labels: np.ndarray | tf.Tensor | None = ...,
        training: bool | None = ...,
    ) -> TFCausalLMOutputWithCrossAttentions | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFRobertaClassificationHead(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
    def call(self, features, training=...): ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    ROBERTA_START_DOCSTRING,
)
class TFRobertaForSequenceClassification(
    TFRobertaPreTrainedModel, TFSequenceClassificationLoss
):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint="cardiffnlp/twitter-roberta-base-emotion",
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="'optimism'",
        expected_loss=0.08,
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
    ROBERTA_START_DOCSTRING,
)
class TFRobertaForMultipleChoice(TFRobertaPreTrainedModel, TFMultipleChoiceLoss):
    _keys_to_ignore_on_load_unexpected = ...
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        ROBERTA_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
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
    ROBERTA_START_DOCSTRING,
)
class TFRobertaForTokenClassification(
    TFRobertaPreTrainedModel, TFTokenClassificationLoss
):
    _keys_to_ignore_on_load_unexpected = ...
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint="ydshieh/roberta-large-ner-english",
        output_type=TFTokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=...,
        expected_loss=0.01,
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
    ROBERTA_START_DOCSTRING,
)
class TFRobertaForQuestionAnswering(TFRobertaPreTrainedModel, TFQuestionAnsweringLoss):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint="ydshieh/roberta-base-squad2",
        output_type=TFQuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="' puppet'",
        expected_loss=0.86,
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
    "TFRobertaForCausalLM",
    "TFRobertaForMaskedLM",
    "TFRobertaForMultipleChoice",
    "TFRobertaForQuestionAnswering",
    "TFRobertaForSequenceClassification",
    "TFRobertaForTokenClassification",
    "TFRobertaMainLayer",
    "TFRobertaModel",
    "TFRobertaPreTrainedModel",
]
