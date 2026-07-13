from dataclasses import dataclass

import tensorflow as tf

from .configuration_idefics import IdeficsVisionConfig
from ...modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPooling
from ...modeling_tf_utils import TFPreTrainedModel
from ...utils import ModelOutput

"""TF IdeficsVision model: a copy of CLIPVisionModel using a simpler config object"""
logger = ...

@dataclass
class TFIdeficsVisionModelOutput(ModelOutput):
    image_embeds: tf.Tensor | None = ...
    last_hidden_state: tf.Tensor | None = ...
    hidden_states: tuple[tf.Tensor] | None = ...
    attentions: tuple[tf.Tensor] | None = ...

class TFIdeficsVisionEmbeddings(tf.keras.layers.Layer):
    def __init__(self, config: IdeficsVisionConfig, **kwargs) -> None: ...
    def interpolate_pos_encoding(
        self, embeddings: tf.Tensor, height: int, width: int
    ) -> tf.Tensor: ...
    def call(
        self, pixel_values: tf.Tensor, interpolate_pos_encoding: bool = ...
    ) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFIdeficsVisionAttention(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor | None = ...,
        causal_attention_mask: tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
    ) -> tuple[tf.Tensor, tf.Tensor | None, tuple[tf.Tensor] | None]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFIdeficsVisionMLP(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFIdeficsVisionEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, config: IdeficsVisionConfig, **kwargs) -> None: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        causal_attention_mask: tf.Tensor,
        output_attentions: bool | None = ...,
    ) -> tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFIdeficsVisionEncoder(tf.keras.layers.Layer):
    def __init__(self, config: IdeficsVisionConfig, **kwargs) -> None: ...
    def call(
        self,
        inputs_embeds,
        attention_mask: tf.Tensor | None = ...,
        causal_attention_mask: tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool | None = ...,
    ) -> tuple | TFBaseModelOutput: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFIdeficsVisionTransformer(TFPreTrainedModel):
    def __init__(self, config: IdeficsVisionConfig, **kwargs) -> None: ...
    def call(
        self,
        pixel_values: tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        interpolate_pos_encoding: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool | None = ...,
    ) -> tuple | TFBaseModelOutputWithPooling: ...
    def build(self, input_shape=...):  # -> None:
        ...
