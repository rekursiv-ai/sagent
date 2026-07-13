import numpy as np
import tensorflow as tf

from .configuration_deberta_v2 import DebertaV2Config
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
    unpack_inputs,
)
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)

"""TF 2.0 DeBERTa-v2 model."""
logger = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...

class TFDebertaV2ContextPooler(keras.layers.Layer):
    def __init__(self, config: DebertaV2Config, **kwargs) -> None: ...
    def call(self, hidden_states, training: bool = ...): ...
    @property
    def output_dim(self) -> int: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFDebertaV2XSoftmax(keras.layers.Layer):
    def __init__(self, axis=..., **kwargs) -> None: ...
    def call(self, inputs: tf.Tensor, mask: tf.Tensor): ...

class TFDebertaV2StableDropout(keras.layers.Layer):
    def __init__(self, drop_prob, **kwargs) -> None: ...
    @tf.custom_gradient
    def xdropout(self, inputs):  # -> tuple[Any, Callable[..., Any]]:
        ...
    def call(
        self, inputs: tf.Tensor, training: tf.Tensor = ...
    ):  # -> tuple[Any, Callable[..., Any]]:
        ...

class TFDebertaV2SelfOutput(keras.layers.Layer):
    def __init__(self, config: DebertaV2Config, **kwargs) -> None: ...
    def call(self, hidden_states, input_tensor, training: bool = ...): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFDebertaV2Attention(keras.layers.Layer):
    def __init__(self, config: DebertaV2Config, **kwargs) -> None: ...
    def call(
        self,
        input_tensor: tf.Tensor,
        attention_mask: tf.Tensor,
        query_states: tf.Tensor | None = ...,
        relative_pos: tf.Tensor | None = ...,
        rel_embeddings: tf.Tensor | None = ...,
        output_attentions: bool = ...,
        training: bool = ...,
    ) -> tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFDebertaV2Intermediate(keras.layers.Layer):
    def __init__(self, config: DebertaV2Config, **kwargs) -> None: ...
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFDebertaV2Output(keras.layers.Layer):
    def __init__(self, config: DebertaV2Config, **kwargs) -> None: ...
    def call(
        self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = ...
    ) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFDebertaV2Layer(keras.layers.Layer):
    def __init__(self, config: DebertaV2Config, **kwargs) -> None: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        query_states: tf.Tensor | None = ...,
        relative_pos: tf.Tensor | None = ...,
        rel_embeddings: tf.Tensor | None = ...,
        output_attentions: bool = ...,
        training: bool = ...,
    ) -> tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFDebertaV2ConvLayer(keras.layers.Layer):
    def __init__(self, config: DebertaV2Config, **kwargs) -> None: ...
    def build(self, input_shape=...):  # -> None:
        ...
    def call(
        self,
        hidden_states: tf.Tensor,
        residual_states: tf.Tensor,
        input_mask: tf.Tensor,
        training: bool = ...,
    ) -> tf.Tensor: ...

class TFDebertaV2Encoder(keras.layers.Layer):
    def __init__(self, config: DebertaV2Config, **kwargs) -> None: ...
    def build(self, input_shape=...):  # -> None:
        ...
    def get_rel_embedding(self):  # -> None:
        ...
    def get_attention_mask(self, attention_mask): ...
    def get_rel_pos(
        self, hidden_states, query_states=..., relative_pos=...
    ):  # -> None:
        ...
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        query_states: tf.Tensor | None = ...,
        relative_pos: tf.Tensor | None = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
        training: bool = ...,
    ) -> TFBaseModelOutput | tuple[tf.Tensor]: ...

def make_log_bucket_position(relative_pos, bucket_size, max_position): ...
def build_relative_position(
    query_size, key_size, bucket_size=..., max_position=...
): ...
def c2p_dynamic_expand(c2p_pos, query_layer, relative_pos): ...
def p2c_dynamic_expand(c2p_pos, query_layer, key_layer): ...
def pos_dynamic_expand(pos_index, p2c_att, key_layer): ...
def take_along_axis(x, indices): ...

class TFDebertaV2DisentangledSelfAttention(keras.layers.Layer):
    def __init__(self, config: DebertaV2Config, **kwargs) -> None: ...
    def transpose_for_scores(
        self, tensor: tf.Tensor, attention_heads: int
    ) -> tf.Tensor: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        query_states: tf.Tensor | None = ...,
        relative_pos: tf.Tensor | None = ...,
        rel_embeddings: tf.Tensor | None = ...,
        output_attentions: bool = ...,
        training: bool = ...,
    ) -> tuple[tf.Tensor]: ...
    def disentangled_att_bias(
        self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor
    ):  # -> Literal[0]:
        ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFDebertaV2Embeddings(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
    def build(self, input_shape=...):  # -> None:
        ...
    def call(
        self,
        input_ids: tf.Tensor | None = ...,
        position_ids: tf.Tensor | None = ...,
        token_type_ids: tf.Tensor | None = ...,
        inputs_embeds: tf.Tensor | None = ...,
        mask: tf.Tensor | None = ...,
        training: bool = ...,
    ) -> tf.Tensor: ...

class TFDebertaV2PredictionHeadTransform(keras.layers.Layer):
    def __init__(self, config: DebertaV2Config, **kwargs) -> None: ...
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFDebertaV2LMPredictionHead(keras.layers.Layer):
    def __init__(
        self, config: DebertaV2Config, input_embeddings: keras.layers.Layer, **kwargs
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

class TFDebertaV2OnlyMLMHead(keras.layers.Layer):
    def __init__(
        self, config: DebertaV2Config, input_embeddings: keras.layers.Layer, **kwargs
    ) -> None: ...
    def call(self, sequence_output: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFDebertaV2MainLayer(keras.layers.Layer):
    config_class = DebertaV2Config
    def __init__(self, config: DebertaV2Config, **kwargs) -> None: ...
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
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> TFBaseModelOutput | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFDebertaV2PreTrainedModel(TFPreTrainedModel):
    config_class = DebertaV2Config
    base_model_prefix = ...

DEBERTA_START_DOCSTRING = ...
DEBERTA_INPUTS_DOCSTRING = ...

@add_start_docstrings(
    ...,
    DEBERTA_START_DOCSTRING,
)
class TFDebertaV2Model(TFDebertaV2PreTrainedModel):
    def __init__(self, config: DebertaV2Config, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool | None = ...,
    ) -> TFBaseModelOutput | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(..., DEBERTA_START_DOCSTRING)
class TFDebertaV2ForMaskedLM(TFDebertaV2PreTrainedModel, TFMaskedLanguageModelingLoss):
    def __init__(self, config: DebertaV2Config, *inputs, **kwargs) -> None: ...
    def get_lm_head(self) -> keras.layers.Layer: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length")
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
    DEBERTA_START_DOCSTRING,
)
class TFDebertaV2ForSequenceClassification(
    TFDebertaV2PreTrainedModel, TFSequenceClassificationLoss
):
    def __init__(self, config: DebertaV2Config, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length")
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
    DEBERTA_START_DOCSTRING,
)
class TFDebertaV2ForTokenClassification(
    TFDebertaV2PreTrainedModel, TFTokenClassificationLoss
):
    def __init__(self, config: DebertaV2Config, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length")
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
    DEBERTA_START_DOCSTRING,
)
class TFDebertaV2ForQuestionAnswering(
    TFDebertaV2PreTrainedModel, TFQuestionAnsweringLoss
):
    def __init__(self, config: DebertaV2Config, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length")
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
    DEBERTA_START_DOCSTRING,
)
class TFDebertaV2ForMultipleChoice(TFDebertaV2PreTrainedModel, TFMultipleChoiceLoss):
    def __init__(self, config: DebertaV2Config, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        DEBERTA_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
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
    "TFDebertaV2ForMaskedLM",
    "TFDebertaV2ForMultipleChoice",
    "TFDebertaV2ForQuestionAnswering",
    "TFDebertaV2ForSequenceClassification",
    "TFDebertaV2ForTokenClassification",
    "TFDebertaV2Model",
    "TFDebertaV2PreTrainedModel",
]
