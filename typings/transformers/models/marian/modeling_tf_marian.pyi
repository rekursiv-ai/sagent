import numpy as np
import tensorflow as tf

from .configuration_marian import MarianConfig
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFSeq2SeqLMOutput,
    TFSeq2SeqModelOutput,
)
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFPreTrainedModel,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ...utils import (
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)

"""TF 2.0 Marian model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
LARGE_NEGATIVE = ...

def shift_tokens_right(
    input_ids: tf.Tensor, pad_token_id: int, decoder_start_token_id: int
): ...

class TFMarianSinusoidalPositionalEmbedding(keras.layers.Layer):
    def __init__(self, num_positions: int, embedding_dim: int, **kwargs) -> None: ...
    def build(self, input_shape: tf.TensorShape):  # -> None:
        ...
    def call(
        self,
        input_shape: tf.TensorShape,
        past_key_values_length: int = ...,
        position_ids: tf.Tensor | None = ...,
    ): ...

class TFMarianAttention(keras.layers.Layer):
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
        training: bool | None = ...,
    ) -> tuple[tf.Tensor, tf.Tensor | None]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFMarianEncoderLayer(keras.layers.Layer):
    def __init__(self, config: MarianConfig, **kwargs) -> None: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: np.ndarray | tf.Tensor | None,
        layer_head_mask: tf.Tensor | None,
        training: bool | None = ...,
    ) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFMarianDecoderLayer(keras.layers.Layer):
    def __init__(self, config: MarianConfig, **kwargs) -> None: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        encoder_hidden_states: np.ndarray | tf.Tensor | None = ...,
        encoder_attention_mask: np.ndarray | tf.Tensor | None = ...,
        layer_head_mask: tf.Tensor | None = ...,
        cross_attn_layer_head_mask: tf.Tensor | None = ...,
        past_key_value: tuple[tuple[np.ndarray | tf.Tensor]] | None = ...,
        training: bool | None = ...,
    ) -> tuple[tf.Tensor, tf.Tensor, tuple[tuple[tf.Tensor]]]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFMarianPreTrainedModel(TFPreTrainedModel):
    config_class = MarianConfig
    base_model_prefix = ...

MARIAN_START_DOCSTRING = ...
MARIAN_GENERATION_EXAMPLE = ...
MARIAN_INPUTS_DOCSTRING = ...

@keras_serializable
class TFMarianEncoder(keras.layers.Layer):
    config_class = MarianConfig
    def __init__(
        self,
        config: MarianConfig,
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
        input_ids: tf.Tensor | None = ...,
        inputs_embeds: tf.Tensor | None = ...,
        attention_mask: tf.Tensor | None = ...,
        head_mask: tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ):  # -> tuple[Any | tuple[Any, ...] | tuple[()], ...] | TFBaseModelOutput:
        ...
    def build(self, input_shape=...):  # -> None:
        ...

@keras_serializable
class TFMarianDecoder(keras.layers.Layer):
    config_class = MarianConfig
    def __init__(
        self,
        config: MarianConfig,
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
        input_ids: tf.Tensor | None = ...,
        inputs_embeds: tf.Tensor | None = ...,
        attention_mask: tf.Tensor | None = ...,
        position_ids: tf.Tensor | None = ...,
        encoder_hidden_states: tf.Tensor | None = ...,
        encoder_attention_mask: tf.Tensor | None = ...,
        head_mask: tf.Tensor | None = ...,
        cross_attn_head_mask: tf.Tensor | None = ...,
        past_key_values: tuple[tuple[tf.Tensor]] | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ):  # -> tuple[Any, tuple[()] | tuple[Any, ...] | None, tuple[Any, ...] | Any | tuple[()] | None, tuple[()] | tuple[Any, ...] | None, tuple[()] | tuple[Any, ...] | None] | TFBaseModelOutputWithPastAndCrossAttentions:
        ...
    def build(self, input_shape=...):  # -> None:
        ...

@keras_serializable
class TFMarianMainLayer(keras.layers.Layer):
    config_class = MarianConfig
    def __init__(self, config: MarianConfig, **kwargs) -> None: ...
    def get_input_embeddings(self): ...
    def set_input_embeddings(self, new_embeddings):  # -> None:
        ...
    @unpack_inputs
    def call(
        self,
        input_ids: tf.Tensor | None = ...,
        attention_mask: tf.Tensor | None = ...,
        decoder_input_ids: tf.Tensor | None = ...,
        decoder_attention_mask: tf.Tensor | None = ...,
        decoder_position_ids: tf.Tensor | None = ...,
        head_mask: tf.Tensor | None = ...,
        decoder_head_mask: tf.Tensor | None = ...,
        cross_attn_head_mask: tf.Tensor | None = ...,
        encoder_outputs: tuple | TFBaseModelOutput | None = ...,
        past_key_values: tuple[tuple[tf.Tensor]] | None = ...,
        inputs_embeds: tf.Tensor | None = ...,
        decoder_inputs_embeds: tf.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
        **kwargs,
    ):  # -> TFSeq2SeqModelOutput:
        ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    MARIAN_START_DOCSTRING,
)
class TFMarianModel(TFMarianPreTrainedModel):
    def __init__(self, config: MarianConfig, *inputs, **kwargs) -> None: ...
    def get_encoder(self):  # -> TFMarianEncoder:
        ...
    def get_decoder(self):  # -> TFMarianDecoder:
        ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        MARIAN_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFSeq2SeqModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: tf.Tensor | None = ...,
        attention_mask: tf.Tensor | None = ...,
        decoder_input_ids: tf.Tensor | None = ...,
        decoder_attention_mask: tf.Tensor | None = ...,
        decoder_position_ids: tf.Tensor | None = ...,
        head_mask: tf.Tensor | None = ...,
        decoder_head_mask: tf.Tensor | None = ...,
        cross_attn_head_mask: tf.Tensor | None = ...,
        encoder_outputs: tf.Tensor | None = ...,
        past_key_values: tuple[tuple[tf.Tensor]] | None = ...,
        inputs_embeds: tf.Tensor | None = ...,
        decoder_inputs_embeds: tf.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
        **kwargs,
    ) -> tuple[tf.Tensor] | TFSeq2SeqModelOutput: ...
    def serving_output(self, output):  # -> TFSeq2SeqModelOutput:
        ...
    def build(self, input_shape=...):  # -> None:
        ...

class BiasLayer(keras.layers.Layer):
    def __init__(self, shape, initializer, trainable, name, **kwargs) -> None: ...
    def call(self, x): ...

@add_start_docstrings(
    ...,
    MARIAN_START_DOCSTRING,
)
class TFMarianMTModel(TFMarianPreTrainedModel, TFCausalLanguageModelingLoss):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    def get_decoder(self):  # -> TFMarianDecoder:
        ...
    def get_encoder(self):  # -> TFMarianEncoder:
        ...
    def get_output_embeddings(self): ...
    def set_output_embeddings(self, value):  # -> None:
        ...
    def get_bias(self):  # -> dict[str, Any]:
        ...
    def set_bias(self, value):  # -> None:
        ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(MARIAN_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=TFSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC
    )
    @add_end_docstrings(MARIAN_GENERATION_EXAMPLE)
    def call(
        self,
        input_ids: tf.Tensor | None = ...,
        attention_mask: tf.Tensor | None = ...,
        decoder_input_ids: tf.Tensor | None = ...,
        decoder_attention_mask: tf.Tensor | None = ...,
        decoder_position_ids: tf.Tensor | None = ...,
        head_mask: tf.Tensor | None = ...,
        decoder_head_mask: tf.Tensor | None = ...,
        cross_attn_head_mask: tf.Tensor | None = ...,
        encoder_outputs: TFBaseModelOutput | None = ...,
        past_key_values: tuple[tuple[tf.Tensor]] | None = ...,
        inputs_embeds: tf.Tensor | None = ...,
        decoder_inputs_embeds: tf.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: tf.Tensor | None = ...,
        training: bool = ...,
    ) -> tuple[tf.Tensor] | TFSeq2SeqLMOutput: ...
    def serving_output(self, output):  # -> TFSeq2SeqLMOutput:
        ...
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=...,
        attention_mask=...,
        decoder_attention_mask=...,
        head_mask=...,
        decoder_head_mask=...,
        cross_attn_head_mask=...,
        use_cache=...,
        encoder_outputs=...,
        **kwargs,
    ):  # -> dict[str, Any | None]:
        ...
    def prepare_decoder_input_ids_from_labels(self, labels: tf.Tensor): ...
    def build(self, input_shape=...):  # -> None:
        ...

__all__ = ["TFMarianMTModel", "TFMarianModel", "TFMarianPreTrainedModel"]
