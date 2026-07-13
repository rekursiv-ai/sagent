import tensorflow as tf

from .configuration_blip import BlipTextConfig
from ...modeling_tf_outputs import TFBaseModelOutputWithPoolingAndCrossAttentions
from ...modeling_tf_utils import (
    TFModelInputType,
    TFPreTrainedModel,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ...utils import add_start_docstrings_to_model_forward

logger = ...
BLIP_TEXT_INPUTS_DOCSTRING = ...

class TFBlipTextEmbeddings(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
    def call(
        self,
        input_ids=...,
        position_ids=...,
        inputs_embeds=...,
        past_key_values_length=...,
        training=...,
    ): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFBlipTextSelfAttention(keras.layers.Layer):
    def __init__(self, config, is_cross_attention, **kwargs) -> None: ...
    def transpose_for_scores(self, x): ...
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
    ):  # -> tuple[Any, ...]:
        ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFBlipTextSelfOutput(keras.layers.Layer):
    def __init__(self, config: BlipTextConfig, **kwargs) -> None: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        input_tensor: tf.Tensor,
        training: bool | None = ...,
    ) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFBlipTextAttention(keras.layers.Layer):
    def __init__(self, config, is_cross_attention=..., **kwargs) -> None: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor | None = ...,
        head_mask: tf.Tensor | None = ...,
        encoder_hidden_states: tf.Tensor | None = ...,
        encoder_attention_mask: tf.Tensor | None = ...,
        past_key_value: tuple[tuple[tf.Tensor]] | None = ...,
        output_attentions: bool | None = ...,
        training: bool | None = ...,
    ): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFBlipTextIntermediate(keras.layers.Layer):
    def __init__(self, config: BlipTextConfig, **kwargs) -> None: ...
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFBlipTextOutput(keras.layers.Layer):
    def __init__(self, config: BlipTextConfig, **kwargs) -> None: ...
    def call(
        self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = ...
    ) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFBlipTextLayer(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
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

@keras_serializable
class TFBlipTextEncoder(keras.layers.Layer):
    config_class = BlipTextConfig
    def __init__(self, config, name=..., **kwargs) -> None: ...
    @unpack_inputs
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

class TFBlipTextPooler(keras.layers.Layer):
    def __init__(self, config: BlipTextConfig, **kwargs) -> None: ...
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFBlipTextPredictionHeadTransform(keras.layers.Layer):
    def __init__(self, config: BlipTextConfig, **kwargs) -> None: ...
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFBlipTextLMPredictionHead(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
    def build(self, input_shape=...):  # -> None:
        ...
    def call(self, hidden_states): ...

class TFBlipTextOnlyMLMHead(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
    def call(self, sequence_output: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFBlipTextPreTrainedModel(TFPreTrainedModel):
    config_class = BlipTextConfig
    base_model_prefix = ...
    _keys_to_ignore_on_load_missing = ...

class TFBlipTextModel(TFBlipTextPreTrainedModel):
    def __init__(self, config, add_pooling_layer=..., name=..., **kwargs) -> None: ...
    def get_input_embeddings(self): ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @tf.function
    def get_extended_attention_mask(
        self, attention_mask: tf.Tensor, input_shape: tuple[int], is_decoder: bool
    ) -> tf.Tensor: ...
    @add_start_docstrings_to_model_forward(BLIP_TEXT_INPUTS_DOCSTRING)
    @unpack_inputs
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: tf.Tensor | None = ...,
        position_ids: tf.Tensor | None = ...,
        head_mask: tf.Tensor | None = ...,
        inputs_embeds: tf.Tensor | None = ...,
        encoder_embeds: tf.Tensor | None = ...,
        encoder_hidden_states: tf.Tensor | None = ...,
        encoder_attention_mask: tf.Tensor | None = ...,
        past_key_values: tuple[tuple[tf.Tensor]] | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        is_decoder: bool = ...,
        training: bool = ...,
    ) -> tuple[tf.Tensor] | TFBaseModelOutputWithPoolingAndCrossAttentions: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFBlipTextLMHeadModel(TFBlipTextPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = ...
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config, **kwargs) -> None: ...
    def get_output_embeddings(self): ...
    def set_output_embeddings(self, new_embeddings):  # -> None:
        ...
    @add_start_docstrings_to_model_forward(BLIP_TEXT_INPUTS_DOCSTRING)
    @unpack_inputs
    def call(
        self,
        input_ids=...,
        attention_mask=...,
        position_ids=...,
        head_mask=...,
        inputs_embeds=...,
        encoder_hidden_states=...,
        encoder_attention_mask=...,
        labels=...,
        past_key_values=...,
        use_cache=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
        return_logits=...,
        is_decoder=...,
        training=...,
    ):  # -> TFCausalLMOutputWithCrossAttentions:
        ...
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=..., attention_mask=..., **model_kwargs
    ):  # -> dict[str, Any | bool | None]:
        ...
    def build(self, input_shape=...):  # -> None:
        ...

__all__ = ["TFBlipTextLMHeadModel", "TFBlipTextModel", "TFBlipTextPreTrainedModel"]
