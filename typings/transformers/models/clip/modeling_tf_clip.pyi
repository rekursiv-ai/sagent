from dataclasses import dataclass
from typing import Any

import numpy as np
import tensorflow as tf

from .configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from ...modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPooling
from ...modeling_tf_utils import (
    TFModelInputType,
    TFPreTrainedModel,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)

"""TF 2.0 CLIP model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
LARGE_NEGATIVE = ...

def contrastive_loss(logits: tf.Tensor) -> tf.Tensor: ...
def clip_loss(similarity: tf.Tensor) -> tf.Tensor: ...

@dataclass
class TFCLIPOutput(ModelOutput):
    loss: tf.Tensor | None = ...
    logits_per_image: tf.Tensor | None = ...
    logits_per_text: tf.Tensor | None = ...
    text_embeds: tf.Tensor | None = ...
    image_embeds: tf.Tensor | None = ...
    text_model_output: TFBaseModelOutputWithPooling = ...
    vision_model_output: TFBaseModelOutputWithPooling = ...
    def to_tuple(self) -> tuple[Any]: ...

class TFCLIPVisionEmbeddings(keras.layers.Layer):
    def __init__(self, config: CLIPVisionConfig, **kwargs) -> None: ...
    def build(self, input_shape: tf.TensorShape = ...):  # -> None:
        ...
    def call(self, pixel_values: tf.Tensor) -> tf.Tensor: ...

class TFCLIPTextEmbeddings(keras.layers.Layer):
    def __init__(self, config: CLIPTextConfig, **kwargs) -> None: ...
    def build(self, input_shape: tf.TensorShape = ...):  # -> None:
        ...
    def call(
        self,
        input_ids: tf.Tensor | None = ...,
        position_ids: tf.Tensor | None = ...,
        inputs_embeds: tf.Tensor | None = ...,
    ) -> tf.Tensor: ...

class TFCLIPAttention(keras.layers.Layer):
    def __init__(self, config: CLIPConfig, **kwargs) -> None: ...
    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        causal_attention_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = ...,
    ) -> tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFCLIPMLP(keras.layers.Layer):
    def __init__(self, config: CLIPConfig, **kwargs) -> None: ...
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFCLIPEncoderLayer(keras.layers.Layer):
    def __init__(self, config: CLIPConfig, **kwargs) -> None: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        causal_attention_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = ...,
    ) -> tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFCLIPEncoder(keras.layers.Layer):
    def __init__(self, config: CLIPConfig, **kwargs) -> None: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        causal_attention_mask: tf.Tensor,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        training: bool = ...,
    ) -> TFBaseModelOutput | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFCLIPTextTransformer(keras.layers.Layer):
    def __init__(self, config: CLIPTextConfig, **kwargs) -> None: ...
    def call(
        self,
        input_ids: TFModelInputType,
        attention_mask: tf.Tensor,
        position_ids: tf.Tensor,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        training: bool = ...,
    ) -> TFBaseModelOutputWithPooling | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

@keras_serializable
class TFCLIPTextMainLayer(keras.layers.Layer):
    config_class = CLIPTextConfig
    def __init__(self, config: CLIPTextConfig, **kwargs) -> None: ...
    def get_input_embeddings(self) -> keras.layers.Layer: ...
    def set_input_embeddings(self, value: tf.Variable):  # -> None:
        ...
    @unpack_inputs
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> TFBaseModelOutputWithPooling | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFCLIPVisionTransformer(keras.layers.Layer):
    def __init__(self, config: CLIPVisionConfig, **kwargs) -> None: ...
    def call(
        self,
        pixel_values: TFModelInputType,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        training: bool = ...,
    ) -> TFBaseModelOutputWithPooling | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

@keras_serializable
class TFCLIPVisionMainLayer(keras.layers.Layer):
    config_class = CLIPVisionConfig
    def __init__(self, config: CLIPVisionConfig, **kwargs) -> None: ...
    def get_input_embeddings(self) -> keras.layers.Layer: ...
    @unpack_inputs
    def call(
        self,
        pixel_values: TFModelInputType | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> TFBaseModelOutputWithPooling | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

@keras_serializable
class TFCLIPMainLayer(keras.layers.Layer):
    config_class = CLIPConfig
    def __init__(self, config: CLIPConfig, **kwargs) -> None: ...
    def build(self, input_shape: tf.TensorShape = ...):  # -> None:
        ...
    @unpack_inputs
    def get_text_features(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> tf.Tensor: ...
    @unpack_inputs
    def get_image_features(
        self,
        pixel_values: TFModelInputType | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> tf.Tensor: ...
    @unpack_inputs
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        pixel_values: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        return_loss: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> TFCLIPOutput | tuple[tf.Tensor]: ...

class TFCLIPPreTrainedModel(TFPreTrainedModel):
    config_class = CLIPConfig
    base_model_prefix = ...
    _keys_to_ignore_on_load_missing = ...
    _keys_to_ignore_on_load_unexpected = ...

CLIP_START_DOCSTRING = ...
CLIP_TEXT_INPUTS_DOCSTRING = ...
CLIP_VISION_INPUTS_DOCSTRING = ...
CLIP_INPUTS_DOCSTRING = ...

class TFCLIPTextModel(TFCLIPPreTrainedModel):
    config_class = CLIPTextConfig
    def __init__(self, config: CLIPTextConfig, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        CLIP_TEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @replace_return_docstrings(
        output_type=TFBaseModelOutputWithPooling, config_class=CLIPTextConfig
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool | None = ...,
    ) -> TFBaseModelOutputWithPooling | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFCLIPVisionModel(TFCLIPPreTrainedModel):
    config_class = CLIPVisionConfig
    main_input_name = ...
    def __init__(self, config: CLIPVisionConfig, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=TFBaseModelOutputWithPooling, config_class=CLIPVisionConfig
    )
    def call(
        self,
        pixel_values: TFModelInputType | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool | None = ...,
    ) -> TFBaseModelOutputWithPooling | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(CLIP_START_DOCSTRING)
class TFCLIPModel(TFCLIPPreTrainedModel):
    config_class = CLIPConfig
    def __init__(self, config: CLIPConfig, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        CLIP_TEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    def get_text_features(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> tf.Tensor: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    def get_image_features(
        self,
        pixel_values: TFModelInputType | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> tf.Tensor: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        CLIP_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @replace_return_docstrings(output_type=TFCLIPOutput, config_class=CLIPConfig)
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        pixel_values: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        return_loss: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> TFCLIPOutput | tuple[tf.Tensor]: ...
    def serving_output(self, output: TFCLIPOutput) -> TFCLIPOutput: ...
    def build(self, input_shape=...):  # -> None:
        ...

__all__ = [
    "TFCLIPModel",
    "TFCLIPPreTrainedModel",
    "TFCLIPTextModel",
    "TFCLIPVisionModel",
]
