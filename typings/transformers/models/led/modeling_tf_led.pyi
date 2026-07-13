from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from .configuration_led import LEDConfig
from ...modeling_tf_utils import (
    TFModelInputType,
    TFPreTrainedModel,
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

"""TF 2.0 LED model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
LARGE_NEGATIVE = ...

def shift_tokens_right(
    input_ids: tf.Tensor, pad_token_id: int, decoder_start_token_id: int
): ...

class TFLEDLearnedPositionalEmbedding(keras.layers.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, **kwargs) -> None: ...
    def call(self, input_shape: tf.TensorShape, past_key_values_length: int = ...): ...

class TFLEDEncoderSelfAttention(keras.layers.Layer):
    def __init__(self, config, layer_id, **kwargs) -> None: ...
    def build(self, input_shape=...):  # -> None:
        ...
    def call(self, inputs, training=...):  # -> tuple[Any, Any, Any]:
        ...
    def reshape_and_transpose(self, vector, batch_size): ...

class TFLEDEncoderAttention(keras.layers.Layer):
    def __init__(self, config, layer_id, **kwargs) -> None: ...
    def call(self, inputs, training=...): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFLEDDecoderAttention(keras.layers.Layer):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = ...,
        is_decoder: bool = ...,
        bias: bool = ...,
        **kwargs,
    ) -> None: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        key_value_states: tf.Tensor | None = ...,
        past_key_value: tuple[tuple[tf.Tensor]] | None = ...,
        attention_mask: tf.Tensor | None = ...,
        layer_head_mask: tf.Tensor | None = ...,
        training=...,
    ) -> tuple[tf.Tensor, tf.Tensor | None]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFLEDEncoderLayer(keras.layers.Layer):
    def __init__(self, config: LEDConfig, layer_id: int, **kwargs) -> None: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        layer_head_mask: tf.Tensor,
        is_index_masked: tf.Tensor,
        is_index_global_attn: tf.Tensor,
        is_global_attn: bool,
        training=...,
    ): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFLEDDecoderLayer(keras.layers.Layer):
    def __init__(self, config: LEDConfig, **kwargs) -> None: ...
    def call(
        self,
        hidden_states,
        attention_mask: tf.Tensor | None = ...,
        encoder_hidden_states: tf.Tensor | None = ...,
        encoder_attention_mask: tf.Tensor | None = ...,
        layer_head_mask: tf.Tensor | None = ...,
        encoder_layer_head_mask: tf.Tensor | None = ...,
        past_key_value: tuple[tf.Tensor] | None = ...,
        training=...,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tuple[tuple[tf.Tensor]]]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFLEDPreTrainedModel(TFPreTrainedModel):
    config_class = LEDConfig
    base_model_prefix = ...
    @property
    def input_signature(self):  # -> dict[str, Any]:
        ...

@dataclass
class TFLEDEncoderBaseModelOutput(ModelOutput):
    last_hidden_state: tf.Tensor | None = ...
    hidden_states: tuple[tf.Tensor, ...] | None = ...
    attentions: tuple[tf.Tensor, ...] | None = ...
    global_attentions: tuple[tf.Tensor, ...] | None = ...

@dataclass
class TFLEDSeq2SeqModelOutput(ModelOutput):
    last_hidden_state: tf.Tensor | None = ...
    past_key_values: list[tf.Tensor] | None = ...
    decoder_hidden_states: tuple[tf.Tensor, ...] | None = ...
    decoder_attentions: tuple[tf.Tensor, ...] | None = ...
    cross_attentions: tuple[tf.Tensor, ...] | None = ...
    encoder_last_hidden_state: tf.Tensor | None = ...
    encoder_hidden_states: tuple[tf.Tensor, ...] | None = ...
    encoder_attentions: tuple[tf.Tensor, ...] | None = ...
    encoder_global_attentions: tuple[tf.Tensor, ...] | None = ...

@dataclass
class TFLEDSeq2SeqLMOutput(ModelOutput):
    loss: tf.Tensor | None = ...
    logits: tf.Tensor | None = ...
    past_key_values: list[tf.Tensor] | None = ...
    decoder_hidden_states: tuple[tf.Tensor, ...] | None = ...
    decoder_attentions: tuple[tf.Tensor, ...] | None = ...
    cross_attentions: tuple[tf.Tensor, ...] | None = ...
    encoder_last_hidden_state: tf.Tensor | None = ...
    encoder_hidden_states: tuple[tf.Tensor, ...] | None = ...
    encoder_attentions: tuple[tf.Tensor, ...] | None = ...
    encoder_global_attentions: tuple[tf.Tensor, ...] | None = ...

LED_START_DOCSTRING = ...
LED_INPUTS_DOCSTRING = ...

@keras_serializable
class TFLEDEncoder(keras.layers.Layer):
    config_class = LEDConfig
    def __init__(
        self,
        config: LEDConfig,
        embed_tokens: keras.layers.Embedding | None = ...,
        **kwargs,
    ) -> None: ...
    def get_embed_tokens(self):  # -> None:
        ...
    def set_embed_tokens(self, embed_tokens):  # -> None:
        ...
    @unpack_inputs
    def call(
        self,
        input_ids=...,
        inputs_embeds=...,
        attention_mask=...,
        global_attention_mask=...,
        head_mask=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
        training=...,
    ):  # -> tuple[Any | tuple[Any, ...] | tuple[()], ...] | TFLEDEncoderBaseModelOutput:
        ...
    @tf.function
    def compute_hidden_states(self, hidden_states, padding_len): ...
    def build(self, input_shape=...):  # -> None:
        ...

@keras_serializable
class TFLEDDecoder(keras.layers.Layer):
    config_class = LEDConfig
    def __init__(
        self,
        config: LEDConfig,
        embed_tokens: keras.layers.Embedding | None = ...,
        **kwargs,
    ) -> None: ...
    def set_embed_tokens(self, embed_tokens):  # -> None:
        ...
    @unpack_inputs
    def call(
        self,
        input_ids=...,
        inputs_embeds=...,
        attention_mask=...,
        encoder_hidden_states=...,
        encoder_attention_mask=...,
        head_mask=...,
        encoder_head_mask=...,
        past_key_values=...,
        use_cache=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
        training=...,
    ):  # -> tuple[Any | tuple[()] | tuple[Any] | tuple[Any, ...], ...] | TFBaseModelOutputWithPastAndCrossAttentions:
        ...
    def build(self, input_shape=...):  # -> None:
        ...

@keras_serializable
class TFLEDMainLayer(keras.layers.Layer):
    config_class = LEDConfig
    def __init__(self, config: LEDConfig, **kwargs) -> None: ...
    def get_input_embeddings(self): ...
    def set_input_embeddings(self, new_embeddings):  # -> None:
        ...
    @unpack_inputs
    def call(
        self,
        input_ids=...,
        attention_mask=...,
        decoder_input_ids=...,
        decoder_attention_mask=...,
        head_mask=...,
        decoder_head_mask=...,
        encoder_outputs: tuple | TFLEDEncoderBaseModelOutput | None = ...,
        global_attention_mask=...,
        past_key_values=...,
        inputs_embeds=...,
        decoder_inputs_embeds=...,
        use_cache=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
        training=...,
        **kwargs,
    ):  # -> TFLEDSeq2SeqModelOutput:
        ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    LED_START_DOCSTRING,
)
class TFLEDModel(TFLEDPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    def get_encoder(self):  # -> TFLEDEncoder:
        ...
    def get_decoder(self):  # -> TFLEDDecoder:
        ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        LED_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFLEDSeq2SeqModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: tf.Tensor | None = ...,
        decoder_input_ids: tf.Tensor | None = ...,
        decoder_attention_mask: tf.Tensor | None = ...,
        head_mask: tf.Tensor | None = ...,
        decoder_head_mask: tf.Tensor | None = ...,
        encoder_outputs: tf.Tensor | None = ...,
        global_attention_mask: tf.Tensor | None = ...,
        past_key_values: tuple[tuple[tf.Tensor]] | None = ...,
        inputs_embeds: tf.Tensor | None = ...,
        decoder_inputs_embeds: tf.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
        **kwargs,
    ) -> tuple[tf.Tensor] | TFLEDSeq2SeqModelOutput: ...
    def serving_output(self, output):  # -> TFLEDSeq2SeqModelOutput:
        ...
    def build(self, input_shape=...):  # -> None:
        ...

class BiasLayer(keras.layers.Layer):
    def __init__(self, shape, initializer, trainable, name, **kwargs) -> None: ...
    def call(self, x): ...

@add_start_docstrings(
    ...,
    LED_START_DOCSTRING,
)
class TFLEDForConditionalGeneration(TFLEDPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    def get_decoder(self):  # -> TFLEDDecoder:
        ...
    def get_encoder(self):  # -> TFLEDEncoder:
        ...
    def get_bias(self):  # -> dict[str, Any]:
        ...
    def set_bias(self, value):  # -> None:
        ...
    def get_output_embeddings(self): ...
    def set_output_embeddings(self, value):  # -> None:
        ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(LED_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=TFLEDSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        decoder_input_ids: np.ndarray | tf.Tensor | None = ...,
        decoder_attention_mask: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        decoder_head_mask: np.ndarray | tf.Tensor | None = ...,
        encoder_outputs: TFLEDEncoderBaseModelOutput | None = ...,
        global_attention_mask: np.ndarray | tf.Tensor | None = ...,
        past_key_values: tuple[tuple[np.ndarray | tf.Tensor]] | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        decoder_inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: tf.Tensor | None = ...,
        training: bool = ...,
    ) -> tuple[tf.Tensor] | TFLEDSeq2SeqLMOutput: ...
    def serving_output(self, output):  # -> TFLEDSeq2SeqLMOutput:
        ...
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=...,
        attention_mask=...,
        head_mask=...,
        decoder_head_mask=...,
        use_cache=...,
        encoder_outputs=...,
        **kwargs,
    ):  # -> dict[str, Any | None]:
        ...
    def prepare_decoder_input_ids_from_labels(self, labels: tf.Tensor): ...
    def hf_compute_loss(self, labels, logits): ...
    def build(self, input_shape=...):  # -> None:
        ...

__all__ = ["TFLEDForConditionalGeneration", "TFLEDModel", "TFLEDPreTrainedModel"]
