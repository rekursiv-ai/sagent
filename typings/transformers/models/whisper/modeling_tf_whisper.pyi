import numpy as np
import tensorflow as tf

from .configuration_whisper import WhisperConfig
from ...generation.configuration_utils import GenerationConfig
from ...generation.tf_logits_process import TFLogitsProcessorList
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
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)

"""TensorFlow Whisper model."""
logger = ...
_CONFIG_FOR_DOC = ...
LARGE_NEGATIVE = ...

def sinusoidal_embedding_init(shape, dtype=...) -> tf.Tensor: ...
def shift_tokens_right(
    input_ids: tf.Tensor, pad_token_id: int, decoder_start_token_id: int
): ...

class TFWhisperPositionalEmbedding(keras.layers.Layer):
    def __init__(
        self,
        num_positions: int,
        embedding_dim: int,
        padding_idx: int | None = ...,
        embedding_initializer=...,
        **kwargs,
    ) -> None: ...
    def build(self, input_shape):  # -> None:
        ...
    def call(self, input_ids, past_key_values_length=...): ...

class TFWhisperAttention(keras.layers.Layer):
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

class TFWhisperEncoderLayer(keras.layers.Layer):
    def __init__(self, config: WhisperConfig, **kwargs) -> None: ...
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

class TFWhisperDecoderLayer(keras.layers.Layer):
    def __init__(self, config: WhisperConfig, **kwargs) -> None: ...
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

class TFWhisperPreTrainedModel(TFPreTrainedModel):
    config_class = WhisperConfig
    base_model_prefix = ...
    main_input_name = ...
    @property
    def dummy_inputs(self) -> dict[str, tf.Tensor]: ...
    @property
    def input_signature(self):  # -> dict[str, Any]:
        ...

WHISPER_START_DOCSTRING = ...
WHISPER_INPUTS_DOCSTRING = ...

@keras_serializable
class TFWhisperEncoder(keras.layers.Layer):
    config_class = WhisperConfig
    def __init__(self, config: WhisperConfig, **kwargs) -> None: ...
    @unpack_inputs
    def call(
        self,
        input_features=...,
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
class TFWhisperDecoder(keras.layers.Layer):
    config_class = WhisperConfig
    def __init__(self, config: WhisperConfig, **kwargs) -> None: ...
    @unpack_inputs
    def call(
        self,
        input_ids=...,
        attention_mask=...,
        position_ids=...,
        encoder_hidden_states=...,
        head_mask=...,
        cross_attn_head_mask=...,
        past_key_values=...,
        inputs_embeds=...,
        use_cache=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
        training=...,
    ): ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    WHISPER_START_DOCSTRING,
)
@keras_serializable
class TFWhisperMainLayer(keras.layers.Layer):
    config_class = WhisperConfig
    def __init__(self, config: WhisperConfig, **kwargs) -> None: ...
    def get_input_embeddings(self): ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def get_encoder(self):  # -> TFWhisperEncoder:
        ...
    @add_start_docstrings_to_model_forward(WHISPER_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=TFSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC
    )
    @unpack_inputs
    def call(
        self,
        input_features=...,
        decoder_input_ids=...,
        decoder_attention_mask=...,
        decoder_position_ids=...,
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
    ):  # -> TFSeq2SeqModelOutput:
        ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    WHISPER_START_DOCSTRING,
)
class TFWhisperModel(TFWhisperPreTrainedModel):
    def __init__(self, config: WhisperConfig, **kwargs) -> None: ...
    def get_input_embeddings(self): ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def get_encoder(self):  # -> TFWhisperEncoder:
        ...
    def get_decoder(self):  # -> TFWhisperDecoder:
        ...
    def decoder(self):  # -> TFWhisperDecoder:
        ...
    def encoder(self):  # -> TFWhisperEncoder:
        ...
    @add_start_docstrings_to_model_forward(WHISPER_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=TFSeq2SeqModelOutput, config_class=_CONFIG_FOR_DOC
    )
    @unpack_inputs
    def call(
        self,
        input_features: TFModelInputType | None = ...,
        decoder_input_ids: np.ndarray | tf.Tensor | None = ...,
        decoder_attention_mask: np.ndarray | tf.Tensor | None = ...,
        decoder_position_ids: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        decoder_head_mask: np.ndarray | tf.Tensor | None = ...,
        cross_attn_head_mask: np.ndarray | tf.Tensor | None = ...,
        encoder_outputs: tuple[tuple[np.ndarray | tf.Tensor]] | None = ...,
        past_key_values: tuple[tuple[np.ndarray | tf.Tensor]] | None = ...,
        decoder_inputs_embeds: tuple[np.ndarray | tf.Tensor] | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> tuple[tf.Tensor] | TFSeq2SeqModelOutput: ...
    def serving_output(self, output):  # -> TFSeq2SeqModelOutput:
        ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    WHISPER_START_DOCSTRING,
)
class TFWhisperForConditionalGeneration(
    TFWhisperPreTrainedModel, TFCausalLanguageModelingLoss
):
    base_model_prefix = ...
    _keys_to_ignore_on_load_missing = ...
    _keys_to_ignore_on_save = ...
    def __init__(self, config: WhisperConfig, **kwargs) -> None: ...
    def get_encoder(self):  # -> TFWhisperEncoder:
        ...
    def get_decoder(self): ...
    def get_output_embeddings(self): ...
    def set_output_embeddings(self, value):  # -> None:
        ...
    def resize_token_embeddings(
        self, new_num_tokens: int
    ) -> keras.layers.Embedding: ...
    @add_start_docstrings_to_model_forward(WHISPER_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=TFSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC
    )
    @unpack_inputs
    def call(
        self,
        input_features: TFModelInputType | None = ...,
        decoder_input_ids: np.ndarray | tf.Tensor | None = ...,
        decoder_attention_mask: np.ndarray | tf.Tensor | None = ...,
        decoder_position_ids: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        decoder_head_mask: np.ndarray | tf.Tensor | None = ...,
        cross_attn_head_mask: np.ndarray | tf.Tensor | None = ...,
        encoder_outputs: tuple[tuple[np.ndarray | tf.Tensor]] | None = ...,
        past_key_values: tuple[tuple[np.ndarray | tf.Tensor]] | None = ...,
        decoder_inputs_embeds: tuple[np.ndarray | tf.Tensor] | None = ...,
        labels: np.ndarray | tf.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> tuple[tf.Tensor] | TFSeq2SeqLMOutput: ...
    def generate(
        self,
        inputs: tf.Tensor | None = ...,
        generation_config: GenerationConfig | None = ...,
        logits_processor: TFLogitsProcessorList | None = ...,
        seed: list[int] | None = ...,
        return_timestamps: bool | None = ...,
        task: str | None = ...,
        language: str | None = ...,
        is_multilingual: bool | None = ...,
        prompt_ids: tf.Tensor | None = ...,
        return_token_timestamps=...,
        **kwargs,
    ): ...
    def serving_output(self, output):  # -> TFSeq2SeqLMOutput:
        ...
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=...,
        use_cache=...,
        encoder_outputs=...,
        attention_mask=...,
        decoder_attention_mask=...,
        **kwargs,
    ):  # -> dict[str, Any | None]:
        ...
    def build(self, input_shape=...):  # -> None:
        ...

__all__ = [
    "TFWhisperForConditionalGeneration",
    "TFWhisperModel",
    "TFWhisperPreTrainedModel",
]
