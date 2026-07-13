import numpy as np
import tensorflow as tf

from .configuration_layoutlm import LayoutLMConfig
from ...modeling_tf_outputs import (
    TFBaseModelOutputWithPastAndCrossAttentions,
    TFBaseModelOutputWithPoolingAndCrossAttentions,
    TFMaskedLMOutput,
    TFQuestionAnsweringModelOutput,
    TFSequenceClassifierOutput,
    TFTokenClassifierOutput,
)
from ...modeling_tf_utils import (
    TFMaskedLanguageModelingLoss,
    TFModelInputType,
    TFPreTrainedModel,
    TFQuestionAnsweringLoss,
    TFSequenceClassificationLoss,
    TFTokenClassificationLoss,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)

"""TF 2.0 LayoutLM model."""
logger = ...
_CONFIG_FOR_DOC = ...

class TFLayoutLMEmbeddings(keras.layers.Layer):
    def __init__(self, config: LayoutLMConfig, **kwargs) -> None: ...
    def build(self, input_shape=...):  # -> None:
        ...
    def call(
        self,
        input_ids: tf.Tensor | None = ...,
        bbox: tf.Tensor | None = ...,
        position_ids: tf.Tensor | None = ...,
        token_type_ids: tf.Tensor | None = ...,
        inputs_embeds: tf.Tensor | None = ...,
        training: bool = ...,
    ) -> tf.Tensor: ...

class TFLayoutLMSelfAttention(keras.layers.Layer):
    def __init__(self, config: LayoutLMConfig, **kwargs) -> None: ...
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

class TFLayoutLMSelfOutput(keras.layers.Layer):
    def __init__(self, config: LayoutLMConfig, **kwargs) -> None: ...
    def call(
        self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = ...
    ) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFLayoutLMAttention(keras.layers.Layer):
    def __init__(self, config: LayoutLMConfig, **kwargs) -> None: ...
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

class TFLayoutLMIntermediate(keras.layers.Layer):
    def __init__(self, config: LayoutLMConfig, **kwargs) -> None: ...
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFLayoutLMOutput(keras.layers.Layer):
    def __init__(self, config: LayoutLMConfig, **kwargs) -> None: ...
    def call(
        self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = ...
    ) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFLayoutLMLayer(keras.layers.Layer):
    def __init__(self, config: LayoutLMConfig, **kwargs) -> None: ...
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

class TFLayoutLMEncoder(keras.layers.Layer):
    def __init__(self, config: LayoutLMConfig, **kwargs) -> None: ...
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

class TFLayoutLMPooler(keras.layers.Layer):
    def __init__(self, config: LayoutLMConfig, **kwargs) -> None: ...
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFLayoutLMPredictionHeadTransform(keras.layers.Layer):
    def __init__(self, config: LayoutLMConfig, **kwargs) -> None: ...
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFLayoutLMLMPredictionHead(keras.layers.Layer):
    def __init__(
        self, config: LayoutLMConfig, input_embeddings: keras.layers.Layer, **kwargs
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

class TFLayoutLMMLMHead(keras.layers.Layer):
    def __init__(
        self, config: LayoutLMConfig, input_embeddings: keras.layers.Layer, **kwargs
    ) -> None: ...
    def call(self, sequence_output: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

@keras_serializable
class TFLayoutLMMainLayer(keras.layers.Layer):
    config_class = LayoutLMConfig
    def __init__(
        self, config: LayoutLMConfig, add_pooling_layer: bool = ..., **kwargs
    ) -> None: ...
    def get_input_embeddings(self) -> keras.layers.Layer: ...
    def set_input_embeddings(self, value: tf.Variable):  # -> None:
        ...
    @unpack_inputs
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        bbox: np.ndarray | tf.Tensor | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        encoder_hidden_states: np.ndarray | tf.Tensor | None = ...,
        encoder_attention_mask: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> TFBaseModelOutputWithPoolingAndCrossAttentions | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFLayoutLMPreTrainedModel(TFPreTrainedModel):
    config_class = LayoutLMConfig
    base_model_prefix = ...
    @property
    def input_signature(self):  # -> dict[str, Any]:
        ...

LAYOUTLM_START_DOCSTRING = ...
LAYOUTLM_INPUTS_DOCSTRING = ...

@add_start_docstrings(
    ...,
    LAYOUTLM_START_DOCSTRING,
)
class TFLayoutLMModel(TFLayoutLMPreTrainedModel):
    def __init__(self, config: LayoutLMConfig, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        LAYOUTLM_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @replace_return_docstrings(
        output_type=TFBaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        bbox: np.ndarray | tf.Tensor | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        encoder_hidden_states: np.ndarray | tf.Tensor | None = ...,
        encoder_attention_mask: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool | None = ...,
    ) -> TFBaseModelOutputWithPoolingAndCrossAttentions | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    LAYOUTLM_START_DOCSTRING,
)
class TFLayoutLMForMaskedLM(TFLayoutLMPreTrainedModel, TFMaskedLanguageModelingLoss):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config: LayoutLMConfig, *inputs, **kwargs) -> None: ...
    def get_lm_head(self) -> keras.layers.Layer: ...
    def get_prefix_bias_name(self) -> str: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        LAYOUTLM_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @replace_return_docstrings(
        output_type=TFMaskedLMOutput, config_class=_CONFIG_FOR_DOC
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        bbox: np.ndarray | tf.Tensor | None = ...,
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
    LAYOUTLM_START_DOCSTRING,
)
class TFLayoutLMForSequenceClassification(
    TFLayoutLMPreTrainedModel, TFSequenceClassificationLoss
):
    _keys_to_ignore_on_load_unexpected = ...
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config: LayoutLMConfig, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        LAYOUTLM_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @replace_return_docstrings(
        output_type=TFSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        bbox: np.ndarray | tf.Tensor | None = ...,
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
    LAYOUTLM_START_DOCSTRING,
)
class TFLayoutLMForTokenClassification(
    TFLayoutLMPreTrainedModel, TFTokenClassificationLoss
):
    _keys_to_ignore_on_load_unexpected = ...
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config: LayoutLMConfig, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        LAYOUTLM_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @replace_return_docstrings(
        output_type=TFTokenClassifierOutput, config_class=_CONFIG_FOR_DOC
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        bbox: np.ndarray | tf.Tensor | None = ...,
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
    LAYOUTLM_START_DOCSTRING,
)
class TFLayoutLMForQuestionAnswering(
    TFLayoutLMPreTrainedModel, TFQuestionAnsweringLoss
):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config: LayoutLMConfig, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        LAYOUTLM_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @replace_return_docstrings(
        output_type=TFQuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        bbox: np.ndarray | tf.Tensor | None = ...,
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
    "TFLayoutLMForMaskedLM",
    "TFLayoutLMForQuestionAnswering",
    "TFLayoutLMForSequenceClassification",
    "TFLayoutLMForTokenClassification",
    "TFLayoutLMMainLayer",
    "TFLayoutLMModel",
    "TFLayoutLMPreTrainedModel",
]
