import numpy as np
import tensorflow as tf

from .configuration_esm import EsmConfig
from ...file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from ...modeling_tf_outputs import (
    TFBaseModelOutputWithPoolingAndCrossAttentions,
    TFMaskedLMOutput,
    TFSequenceClassifierOutput,
    TFTokenClassifierOutput,
)
from ...modeling_tf_utils import (
    TFMaskedLanguageModelingLoss,
    TFModelInputType,
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    TFTokenClassificationLoss,
    keras,
    unpack_inputs,
)

"""PyTorch ESM model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...

def rotate_half(x): ...
def apply_rotary_pos_emb(x, cos, sin): ...
def symmetrize(x): ...
def average_product_correct(x): ...

class TFRotaryEmbedding(keras.layers.Layer):
    def __init__(self, dim: int, name=...) -> None: ...
    def build(self, input_shape):  # -> None:
        ...
    def call(self, q: tf.Tensor, k: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]: ...

class TFEsmContactPredictionHead(keras.layers.Layer):
    def __init__(
        self, in_features: int, bias=..., eos_idx: int = ..., name=...
    ) -> None: ...
    def build(self, input_shape=...):  # -> None:
        ...
    def call(self, tokens, attentions): ...

class TFEsmEmbeddings(keras.layers.Layer):
    def __init__(self, config, name=...) -> None: ...
    def call(
        self,
        input_ids=...,
        attention_mask=...,
        position_ids=...,
        inputs_embeds=...,
        past_key_values_length=...,
    ): ...
    def create_position_ids_from_inputs_embeds(self, inputs_embeds): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFEsmSelfAttention(keras.layers.Layer):
    def __init__(self, config, position_embedding_type=..., name=...) -> None: ...
    def transpose_for_scores(self, x: tf.Tensor) -> tf.Tensor: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor | None = ...,
        head_mask: tf.Tensor | None = ...,
        encoder_hidden_states: tf.Tensor | None = ...,
        encoder_attention_mask: tf.Tensor | None = ...,
        past_key_value: tuple[tuple[tf.Tensor]] | None = ...,
        output_attentions: bool | None = ...,
        training: bool = ...,
    ) -> tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFEsmSelfOutput(keras.layers.Layer):
    def __init__(self, config, name=...) -> None: ...
    def call(self, hidden_states, input_tensor, training=...): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFEsmAttention(keras.layers.Layer):
    def __init__(self, config, name=...) -> None: ...
    def prune_heads(self, heads): ...
    def call(
        self,
        hidden_states,
        attention_mask=...,
        head_mask=...,
        encoder_hidden_states=...,
        encoder_attention_mask=...,
        past_key_value=...,
        output_attentions=...,
        training=...,
    ): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFEsmIntermediate(keras.layers.Layer):
    def __init__(self, config: EsmConfig, **kwargs) -> None: ...
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFEsmOutput(keras.layers.Layer):
    def __init__(self, config, name=...) -> None: ...
    def call(self, hidden_states, input_tensor, training=...): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFEsmLayer(keras.layers.Layer):
    def __init__(self, config, name=...) -> None: ...
    def call(
        self,
        hidden_states,
        attention_mask=...,
        head_mask=...,
        encoder_hidden_states=...,
        encoder_attention_mask=...,
        past_key_value=...,
        output_attentions=...,
        training=...,
    ): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFEsmEncoder(keras.layers.Layer):
    def __init__(self, config, name=...) -> None: ...
    def call(
        self,
        hidden_states,
        attention_mask=...,
        head_mask=...,
        encoder_hidden_states=...,
        encoder_attention_mask=...,
        past_key_values=...,
        use_cache=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
        training=...,
    ):  # -> tuple[Any | tuple[()] | tuple[Any, ...], ...] | TFBaseModelOutputWithPastAndCrossAttentions:
        ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFEsmPooler(keras.layers.Layer):
    def __init__(self, config: EsmConfig, **kwargs) -> None: ...
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFEsmPreTrainedModel(TFPreTrainedModel):
    config_class = EsmConfig
    base_model_prefix = ...

ESM_START_DOCSTRING = ...
ESM_INPUTS_DOCSTRING = ...

@add_start_docstrings(
    ...,
    ESM_START_DOCSTRING,
)
class TFEsmMainLayer(keras.layers.Layer):
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config, add_pooling_layer=..., name=..., **kwargs) -> None: ...
    def build(self, input_shape=...):  # -> None:
        ...
    def get_input_embeddings(self): ...
    def set_input_embeddings(self, value: tf.Variable):  # -> None:
        ...
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
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
    def predict_contacts(self, tokens, attention_mask): ...

@add_start_docstrings(
    ...,
    ESM_START_DOCSTRING,
)
class TFEsmModel(TFEsmPreTrainedModel):
    def __init__(
        self, config: EsmConfig, add_pooling_layer=..., *inputs, **kwargs
    ) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        ESM_INPUTS_DOCSTRING.format("batch_size, sequence_length")
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
    ) -> TFBaseModelOutputWithPoolingAndCrossAttentions | tuple[tf.Tensor]: ...
    def predict_contacts(self, tokens, attention_mask): ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    """ESM Model with a `language modeling` head on top.""", ESM_START_DOCSTRING
)
class TFEsmForMaskedLM(TFEsmPreTrainedModel, TFMaskedLanguageModelingLoss):
    _keys_to_ignore_on_load_missing = ...
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config) -> None: ...
    def get_output_embeddings(self):  # -> None:
        ...
    def set_output_embeddings(self, new_embeddings):  # -> None:
        ...
    def get_lm_head(self):  # -> TFEsmLMHead:
        ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        ESM_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFMaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        mask="<mask>",
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        encoder_hidden_states: np.ndarray | tf.Tensor | None = ...,
        encoder_attention_mask: np.ndarray | tf.Tensor | None = ...,
        labels: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> TFMaskedLMOutput | tuple[tf.Tensor]: ...
    def predict_contacts(self, tokens, attention_mask): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFEsmLMHead(keras.layers.Layer):
    def __init__(self, config, name=...) -> None: ...
    def build(self, input_shape=...):  # -> None:
        ...
    def get_bias(self):  # -> dict[str, Any]:
        ...
    def call(self, features): ...

@add_start_docstrings(
    ...,
    ESM_START_DOCSTRING,
)
class TFEsmForSequenceClassification(
    TFEsmPreTrainedModel, TFSequenceClassificationLoss
):
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        ESM_INPUTS_DOCSTRING.format("batch_size, sequence_length")
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
        position_ids: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        labels: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> TFSequenceClassifierOutput | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    ESM_START_DOCSTRING,
)
class TFEsmForTokenClassification(TFEsmPreTrainedModel, TFTokenClassificationLoss):
    _keys_to_ignore_on_load_unexpected = ...
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        ESM_INPUTS_DOCSTRING.format("batch_size, sequence_length")
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
        position_ids: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        labels: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> TFTokenClassifierOutput | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFEsmClassificationHead(keras.layers.Layer):
    def __init__(self, config, name=...) -> None: ...
    def call(self, features, training=...): ...
    def build(self, input_shape=...):  # -> None:
        ...

def create_position_ids_from_input_ids(
    input_ids, padding_idx, past_key_values_length=...
): ...

__all__ = [
    "TFEsmForMaskedLM",
    "TFEsmForSequenceClassification",
    "TFEsmForTokenClassification",
    "TFEsmModel",
    "TFEsmPreTrainedModel",
]
