import numpy as np
import tensorflow as tf

from .configuration_speech_to_text import Speech2TextConfig
from ...modeling_tf_outputs import TFSeq2SeqLMOutput, TFSeq2SeqModelOutput
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFModelInputType,
    TFPreTrainedModel,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)

"""TensorFlow Speech2Text model."""
logger = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
LARGE_NEGATIVE = ...

def shift_tokens_right(
    input_ids: tf.Tensor, pad_token_id: int, decoder_start_token_id: int
): ...

class TFConv1dSubsampler(keras.layers.Layer):
    def __init__(self, config: Speech2TextConfig, **kwargs) -> None: ...
    def call(self, input_features: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFSpeech2TextSinusoidalPositionalEmbedding(keras.layers.Layer):
    def __init__(
        self,
        num_positions: int,
        embedding_dim: int,
        padding_idx: int | None = ...,
        **kwargs,
    ) -> None: ...
    def call(
        self, input_ids: tf.Tensor, past_key_values_length: int = ...
    ) -> tf.Tensor: ...
    @staticmethod
    def create_position_ids_from_input_ids(
        input_ids: tf.Tensor, padding_idx: int, past_key_values_length: int | None = ...
    ) -> tf.Tensor: ...

class TFSpeech2TextAttention(keras.layers.Layer):
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

class TFSpeech2TextEncoderLayer(keras.layers.Layer):
    def __init__(self, config: Speech2TextConfig, **kwargs) -> None: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        layer_head_mask: tf.Tensor,
        training: bool = ...,
    ):  # -> tuple[Any, Any]:
        ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFSpeech2TextDecoderLayer(keras.layers.Layer):
    def __init__(self, config: Speech2TextConfig, **kwargs) -> None: ...
    def call(
        self,
        hidden_states,
        attention_mask: tf.Tensor | None = ...,
        encoder_hidden_states: tf.Tensor | None = ...,
        encoder_attention_mask: tf.Tensor | None = ...,
        layer_head_mask: tf.Tensor | None = ...,
        cross_attn_layer_head_mask: tf.Tensor | None = ...,
        past_key_value: tuple[tf.Tensor] | None = ...,
        training=...,
    ) -> tuple[tf.Tensor, tf.Tensor, tuple[tuple[tf.Tensor]]]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFSpeech2TextPreTrainedModel(TFPreTrainedModel):
    config_class = Speech2TextConfig
    base_model_prefix = ...
    main_input_name = ...
    _keys_to_ignore_on_load_unexpected = ...
    @property
    def input_signature(self):  # -> dict[str, Any]:
        ...

SPEECH_TO_TEXT_START_DOCSTRING = ...
SPEECH_TO_TEXT_INPUTS_DOCSTRING = ...

@keras_serializable
class TFSpeech2TextEncoder(keras.layers.Layer):
    config_class = Speech2TextConfig
    def __init__(self, config: Speech2TextConfig, **kwargs) -> None: ...
    @unpack_inputs
    def call(
        self,
        input_features=...,
        attention_mask=...,
        head_mask=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
        training=...,
    ):  # -> tuple[Any | tuple[Any, ...] | tuple[()], ...] | TFBaseModelOutput:
        ...
    def build(self, input_shape=...):  # -> None:
        ...

@keras_serializable
class TFSpeech2TextDecoder(keras.layers.Layer):
    config_class = Speech2TextConfig
    def __init__(self, config: Speech2TextConfig, **kwargs) -> None: ...
    def get_embed_tokens(self):  # -> TFSharedEmbeddings:
        ...
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
        cross_attn_head_mask=...,
        past_key_values=...,
        use_cache=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
        training=...,
    ):  # -> tuple[Any, tuple[()] | tuple[Any, ...] | None, tuple[Any, ...] | Any | tuple[()] | None, tuple[()] | tuple[Any, ...] | None, tuple[()] | tuple[Any, ...] | None] | TFBaseModelOutputWithPastAndCrossAttentions:
        ...
    def build(self, input_shape=...):  # -> None:
        ...

@keras_serializable
class TFSpeech2TextMainLayer(keras.layers.Layer):
    config_class = Speech2TextConfig
    def __init__(self, config: Speech2TextConfig, **kwargs) -> None: ...
    def get_input_embeddings(self):  # -> TFSharedEmbeddings:
        ...
    def set_input_embeddings(self, new_embeddings):  # -> None:
        ...
    @unpack_inputs
    def call(
        self,
        input_features=...,
        attention_mask=...,
        decoder_input_ids=...,
        decoder_attention_mask=...,
        head_mask=...,
        decoder_head_mask=...,
        cross_attn_head_mask=...,
        encoder_outputs=...,
        past_key_values=...,
        decoder_inputs_embeds=...,
        use_cache=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
        training=...,
        **kwargs,
    ):  # -> TFSeq2SeqModelOutput:
        ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    SPEECH_TO_TEXT_START_DOCSTRING,
)
class TFSpeech2TextModel(TFSpeech2TextPreTrainedModel):
    def __init__(self, config: Speech2TextConfig, *inputs, **kwargs) -> None: ...
    def get_encoder(self):  # -> TFSpeech2TextEncoder:
        ...
    def get_decoder(self):  # -> TFSpeech2TextDecoder:
        ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(SPEECH_TO_TEXT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFSeq2SeqModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_features: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        decoder_input_ids: np.ndarray | tf.Tensor | None = ...,
        decoder_attention_mask: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        decoder_head_mask: np.ndarray | tf.Tensor | None = ...,
        cross_attn_head_mask: np.ndarray | tf.Tensor | None = ...,
        encoder_outputs: np.ndarray | tf.Tensor | None = ...,
        past_key_values: tuple[tuple[np.ndarray | tf.Tensor]] | None = ...,
        decoder_inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
        **kwargs,
    ) -> tuple | TFSeq2SeqModelOutput: ...
    def serving_output(self, output):  # -> TFSeq2SeqModelOutput:
        ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    SPEECH_TO_TEXT_START_DOCSTRING,
)
class TFSpeech2TextForConditionalGeneration(
    TFSpeech2TextPreTrainedModel, TFCausalLanguageModelingLoss
):
    def __init__(self, config: Speech2TextConfig) -> None: ...
    def get_encoder(self):  # -> TFSpeech2TextEncoder:
        ...
    def get_decoder(self):  # -> TFSpeech2TextDecoder:
        ...
    def resize_token_embeddings(self, new_num_tokens: int) -> tf.Variable: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(SPEECH_TO_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=TFSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC
    )
    def call(
        self,
        input_features: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        decoder_input_ids: np.ndarray | tf.Tensor | None = ...,
        decoder_attention_mask: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        decoder_head_mask: np.ndarray | tf.Tensor | None = ...,
        cross_attn_head_mask: np.ndarray | tf.Tensor | None = ...,
        encoder_outputs: np.ndarray | tf.Tensor | None = ...,
        past_key_values: tuple[tuple[np.ndarray | tf.Tensor]] | None = ...,
        decoder_inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        labels: np.ndarray | tf.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool | None = ...,
        **kwargs,
    ) -> tuple | TFSeq2SeqLMOutput: ...
    def serving_output(self, output):  # -> TFSeq2SeqLMOutput:
        ...
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=...,
        attention_mask=...,
        head_mask=...,
        decoder_head_mask=...,
        cross_attn_head_mask=...,
        use_cache=...,
        encoder_outputs=...,
        **kwargs,
    ):  # -> dict[str, Any | None]:
        ...
    def build(self, input_shape=...):  # -> None:
        ...
    def tf_to_pt_weight_rename(
        self, tf_weight
    ):  # -> tuple[Literal['lm_head.weight'], Literal['model.decoder.embed_tokens.weight']] | tuple[Any]:
        ...

__all__ = [
    "TFSpeech2TextForConditionalGeneration",
    "TFSpeech2TextModel",
    "TFSpeech2TextPreTrainedModel",
]
