import numpy as np
import tensorflow as tf

from .configuration_roformer import RoFormerConfig
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFBaseModelOutputWithPooling,
    TFCausalLMOutput,
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

"""TF 2.0 RoFormer model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...

class TFRoFormerSinusoidalPositionalEmbedding(keras.layers.Layer):
    def __init__(self, num_positions: int, embedding_dim: int, **kwargs) -> None: ...
    def build(self, input_shape: tf.TensorShape):  # -> None:
        ...
    def call(self, input_shape: tf.TensorShape, past_key_values_length: int = ...): ...

class TFRoFormerEmbeddings(keras.layers.Layer):
    def __init__(self, config: RoFormerConfig, **kwargs) -> None: ...
    def build(self, input_shape=...):  # -> None:
        ...
    def call(
        self,
        input_ids: tf.Tensor | None = ...,
        token_type_ids: tf.Tensor | None = ...,
        inputs_embeds: tf.Tensor | None = ...,
        training: bool = ...,
    ) -> tf.Tensor: ...

class TFRoFormerSelfAttention(keras.layers.Layer):
    def __init__(self, config: RoFormerConfig, **kwargs) -> None: ...
    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        sinusoidal_pos: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = ...,
    ) -> tuple[tf.Tensor]: ...
    @staticmethod
    def apply_rotary_position_embeddings(
        sinusoidal_pos, query_layer, key_layer, value_layer=...
    ):  # -> tuple[Any, Any, Any] | tuple[Any, Any]:
        ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFRoFormerSelfOutput(keras.layers.Layer):
    def __init__(self, config: RoFormerConfig, **kwargs) -> None: ...
    def call(
        self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = ...
    ) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFRoFormerAttention(keras.layers.Layer):
    def __init__(self, config: RoFormerConfig, **kwargs) -> None: ...
    def prune_heads(self, heads): ...
    def call(
        self,
        input_tensor: tf.Tensor,
        attention_mask: tf.Tensor,
        sinusoidal_pos: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = ...,
    ) -> tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFRoFormerIntermediate(keras.layers.Layer):
    def __init__(self, config: RoFormerConfig, **kwargs) -> None: ...
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFRoFormerOutput(keras.layers.Layer):
    def __init__(self, config: RoFormerConfig, **kwargs) -> None: ...
    def call(
        self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = ...
    ) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFRoFormerLayer(keras.layers.Layer):
    def __init__(self, config: RoFormerConfig, **kwargs) -> None: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        sinusoidal_pos: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = ...,
    ) -> tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFRoFormerEncoder(keras.layers.Layer):
    def __init__(self, config: RoFormerConfig, **kwargs) -> None: ...
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

class TFRoFormerPredictionHeadTransform(keras.layers.Layer):
    def __init__(self, config: RoFormerConfig, **kwargs) -> None: ...
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFRoFormerLMPredictionHead(keras.layers.Layer):
    def __init__(
        self, config: RoFormerConfig, input_embeddings: keras.layers.Layer, **kwargs
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

class TFRoFormerMLMHead(keras.layers.Layer):
    def __init__(
        self, config: RoFormerConfig, input_embeddings: keras.layers.Layer, **kwargs
    ) -> None: ...
    def call(self, sequence_output: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

@keras_serializable
class TFRoFormerMainLayer(keras.layers.Layer):
    config_class = RoFormerConfig
    def __init__(
        self, config: RoFormerConfig, add_pooling_layer: bool = ..., **kwargs
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
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> TFBaseModelOutput | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFRoFormerPreTrainedModel(TFPreTrainedModel):
    config_class = RoFormerConfig
    base_model_prefix = ...

ROFORMER_START_DOCSTRING = ...
ROFORMER_INPUTS_DOCSTRING = ...

@add_start_docstrings(
    ...,
    ROFORMER_START_DOCSTRING,
)
class TFRoFormerModel(TFRoFormerPreTrainedModel):
    def __init__(self, config: RoFormerConfig, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        ROFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length")
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
    ROFORMER_START_DOCSTRING,
)
class TFRoFormerForMaskedLM(TFRoFormerPreTrainedModel, TFMaskedLanguageModelingLoss):
    def __init__(self, config: RoFormerConfig, *inputs, **kwargs) -> None: ...
    def get_lm_head(self) -> keras.layers.Layer: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        ROFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length")
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
    ROFORMER_START_DOCSTRING,
)
class TFRoFormerForCausalLM(TFRoFormerPreTrainedModel, TFCausalLanguageModelingLoss):
    def __init__(self, config: RoFormerConfig, *inputs, **kwargs) -> None: ...
    def get_lm_head(self) -> keras.layers.Layer: ...
    @unpack_inputs
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFCausalLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: np.ndarray | tf.Tensor | None = ...,
        training: bool | None = ...,
    ) -> TFCausalLMOutput | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFRoFormerClassificationHead(keras.layers.Layer):
    def __init__(self, config: RoFormerConfig, *inputs, **kwargs) -> None: ...
    def call(self, hidden_states: tf.Tensor, training: bool = ...) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    ROFORMER_START_DOCSTRING,
)
class TFRoFormerForSequenceClassification(
    TFRoFormerPreTrainedModel, TFSequenceClassificationLoss
):
    def __init__(self, config: RoFormerConfig, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        ROFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length")
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
    ROFORMER_START_DOCSTRING,
)
class TFRoFormerForMultipleChoice(TFRoFormerPreTrainedModel, TFMultipleChoiceLoss):
    def __init__(self, config: RoFormerConfig, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        ROFORMER_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
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
    ROFORMER_START_DOCSTRING,
)
class TFRoFormerForTokenClassification(
    TFRoFormerPreTrainedModel, TFTokenClassificationLoss
):
    def __init__(self, config: RoFormerConfig, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        ROFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length")
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
    ROFORMER_START_DOCSTRING,
)
class TFRoFormerForQuestionAnswering(
    TFRoFormerPreTrainedModel, TFQuestionAnsweringLoss
):
    def __init__(self, config: RoFormerConfig, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        ROFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length")
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
    "TFRoFormerForCausalLM",
    "TFRoFormerForMaskedLM",
    "TFRoFormerForMultipleChoice",
    "TFRoFormerForQuestionAnswering",
    "TFRoFormerForSequenceClassification",
    "TFRoFormerForTokenClassification",
    "TFRoFormerLayer",
    "TFRoFormerModel",
    "TFRoFormerPreTrainedModel",
]
