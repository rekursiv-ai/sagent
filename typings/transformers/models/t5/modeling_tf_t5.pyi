import numpy as np
import tensorflow as tf

from .configuration_t5 import T5Config
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFSeq2SeqLMOutput,
    TFSeq2SeqModelOutput,
)
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFModelInputType,
    TFPreTrainedModel,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)

"""TF 2.0 T5 model."""
logger = ...
_CONFIG_FOR_DOC = ...

class TFT5LayerNorm(keras.layers.Layer):
    def __init__(self, hidden_size, epsilon=..., **kwargs) -> None: ...
    def build(self, input_shape):  # -> None:
        ...
    def call(self, hidden_states): ...

class TFT5DenseActDense(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
    def call(self, hidden_states, training=...): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFT5DenseGatedActDense(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
    def call(self, hidden_states, training=...): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFT5LayerFF(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
    def call(self, hidden_states, training=...): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFT5Attention(keras.layers.Layer):
    NEW_ID = ...
    def __init__(self, config, has_relative_attention_bias=..., **kwargs) -> None: ...
    def build(self, input_shape=...):  # -> None:
        ...
    def prune_heads(self, heads): ...
    def compute_bias(self, query_length, key_length): ...
    def call(
        self,
        hidden_states,
        mask=...,
        key_value_states=...,
        position_bias=...,
        past_key_value=...,
        layer_head_mask=...,
        query_length=...,
        use_cache=...,
        training=...,
        output_attentions=...,
    ):  # -> tuple[Any, tuple[Any, Any] | None, Any, Any] | tuple[Any, tuple[Any, Any] | None, Any]:
        ...

class TFT5LayerSelfAttention(keras.layers.Layer):
    def __init__(self, config, has_relative_attention_bias=..., **kwargs) -> None: ...
    def call(
        self,
        hidden_states,
        attention_mask=...,
        position_bias=...,
        layer_head_mask=...,
        past_key_value=...,
        use_cache=...,
        output_attentions=...,
        training=...,
    ): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFT5LayerCrossAttention(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
    def call(
        self,
        hidden_states,
        key_value_states,
        attention_mask=...,
        position_bias=...,
        layer_head_mask=...,
        past_key_value=...,
        query_length=...,
        use_cache=...,
        output_attentions=...,
        training=...,
    ): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFT5Block(keras.layers.Layer):
    def __init__(self, config, has_relative_attention_bias=..., **kwargs) -> None: ...
    def call(
        self,
        hidden_states,
        attention_mask=...,
        position_bias=...,
        encoder_hidden_states=...,
        encoder_attention_mask=...,
        encoder_decoder_position_bias=...,
        layer_head_mask=...,
        encoder_layer_head_mask=...,
        past_key_value=...,
        use_cache=...,
        output_attentions=...,
        training=...,
    ): ...
    def build(self, input_shape=...):  # -> None:
        ...

@keras_serializable
class TFT5MainLayer(keras.layers.Layer):
    config_class = T5Config
    def __init__(self, config, embed_tokens=..., **kwargs) -> None: ...
    @unpack_inputs
    def call(
        self,
        input_ids=...,
        attention_mask=...,
        encoder_hidden_states=...,
        encoder_attention_mask=...,
        inputs_embeds=...,
        head_mask=...,
        encoder_head_mask=...,
        past_key_values=...,
        use_cache=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
        training=...,
    ) -> tuple: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFT5PreTrainedModel(TFPreTrainedModel):
    config_class = T5Config
    base_model_prefix = ...
    _keys_to_ignore_on_load_unexpected = ...
    def get_input_embeddings(self): ...
    def set_input_embeddings(self, value):  # -> None:
        ...

T5_START_DOCSTRING = ...
T5_INPUTS_DOCSTRING = ...
T5_ENCODER_INPUTS_DOCSTRING = ...
_HEAD_MASK_WARNING_MSG = ...

@add_start_docstrings(
    ...,
    T5_START_DOCSTRING,
)
class TFT5Model(TFT5PreTrainedModel):
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    def get_encoder(self):  # -> TFT5MainLayer:
        ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=TFSeq2SeqModelOutput, config_class=_CONFIG_FOR_DOC
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        decoder_input_ids: np.ndarray | tf.Tensor | None = ...,
        decoder_attention_mask: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        decoder_head_mask: np.ndarray | tf.Tensor | None = ...,
        encoder_outputs: np.ndarray | tf.Tensor | None = ...,
        past_key_values: tuple[tuple[np.ndarray | tf.Tensor]] | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        decoder_inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool | None = ...,
    ) -> tuple | TFSeq2SeqModelOutput: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    """T5 Model with a `language modeling` head on top.""", T5_START_DOCSTRING
)
class TFT5ForConditionalGeneration(TFT5PreTrainedModel, TFCausalLanguageModelingLoss):
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    def get_output_embeddings(self): ...
    def set_output_embeddings(self, value):  # -> None:
        ...
    def get_encoder(self):  # -> TFT5MainLayer:
        ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=TFSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        decoder_input_ids: np.ndarray | tf.Tensor | None = ...,
        decoder_attention_mask: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        decoder_head_mask: np.ndarray | tf.Tensor | None = ...,
        encoder_outputs: np.ndarray | tf.Tensor | None = ...,
        past_key_values: tuple[tuple[np.ndarray | tf.Tensor]] | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        decoder_inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        labels: np.ndarray | tf.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool | None = ...,
    ) -> tuple | TFSeq2SeqLMOutput: ...
    def serving_output(self, output):  # -> TFSeq2SeqLMOutput:
        ...
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=...,
        attention_mask=...,
        decoder_attention_mask=...,
        head_mask=...,
        decoder_head_mask=...,
        use_cache=...,
        encoder_outputs=...,
        **kwargs,
    ):  # -> dict[str, Any | None]:
        ...
    def prepare_decoder_input_ids_from_labels(self, labels: tf.Tensor): ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    T5_START_DOCSTRING,
)
class TFT5EncoderModel(TFT5PreTrainedModel):
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    def get_encoder(self):  # -> TFT5MainLayer:
        ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(T5_ENCODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=TFBaseModelOutput, config_class=_CONFIG_FOR_DOC
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool | None = ...,
    ) -> tuple | TFBaseModelOutput: ...
    def build(self, input_shape=...):  # -> None:
        ...

__all__ = [
    "TFT5EncoderModel",
    "TFT5ForConditionalGeneration",
    "TFT5Model",
    "TFT5PreTrainedModel",
]
