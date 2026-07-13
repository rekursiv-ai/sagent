from dataclasses import dataclass
from typing import Any

import tensorflow as tf

from .configuration_blip import BlipConfig, BlipTextConfig, BlipVisionConfig
from .modeling_tf_blip_text import BLIP_TEXT_INPUTS_DOCSTRING
from ...modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPooling
from ...modeling_tf_utils import (
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

"""TensorFlow BLIP model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...

def contrastive_loss(logits: tf.Tensor) -> tf.Tensor: ...
def blip_loss(similarity: tf.Tensor) -> tf.Tensor: ...

@dataclass
class TFBlipForConditionalGenerationModelOutput(ModelOutput):
    loss: tuple[tf.Tensor] | None = ...
    logits: tuple[tf.Tensor] | None = ...
    image_embeds: tf.Tensor | None = ...
    last_hidden_state: tf.Tensor | None = ...
    hidden_states: tuple[tf.Tensor, ...] | None = ...
    attentions: tuple[tf.Tensor, ...] | None = ...
    @property
    def decoder_logits(self):  # -> tuple[Any] | None:
        ...

@dataclass
class TFBlipTextVisionModelOutput(ModelOutput):
    loss: tf.Tensor | None = ...
    image_embeds: tf.Tensor | None = ...
    last_hidden_state: tf.Tensor | None = ...
    hidden_states: tuple[tf.Tensor, ...] | None = ...
    attentions: tuple[tf.Tensor, ...] | None = ...

@dataclass
class TFBlipImageTextMatchingModelOutput(ModelOutput):
    itm_score: tf.Tensor | None = ...
    loss: tf.Tensor | None = ...
    image_embeds: tf.Tensor | None = ...
    last_hidden_state: tf.Tensor | None = ...
    hidden_states: tuple[tf.Tensor, ...] | None = ...
    vision_pooler_output: tf.Tensor | None = ...
    attentions: tuple[tf.Tensor, ...] | None = ...
    question_embeds: tuple[tf.Tensor] | None = ...

@dataclass
class TFBlipOutput(ModelOutput):
    loss: tf.Tensor | None = ...
    logits_per_image: tf.Tensor | None = ...
    logits_per_text: tf.Tensor | None = ...
    text_embeds: tf.Tensor | None = ...
    image_embeds: tf.Tensor | None = ...
    text_model_output: TFBaseModelOutputWithPooling = ...
    vision_model_output: TFBaseModelOutputWithPooling = ...
    def to_tuple(self) -> tuple[Any]: ...

class TFBlipVisionEmbeddings(keras.layers.Layer):
    def __init__(self, config: BlipVisionConfig, **kwargs) -> None: ...
    def build(self, input_shape=...):  # -> None:
        ...
    def call(self, pixel_values: tf.Tensor) -> tf.Tensor: ...

class TFBlipTextEmbeddings(keras.layers.Layer):
    def __init__(self, config: BlipTextConfig, **kwargs) -> None: ...
    def build(self, input_shape: tf.TensorShape = ...):  # -> None:
        ...
    def call(
        self,
        input_ids: tf.Tensor | None = ...,
        position_ids: tf.Tensor | None = ...,
        inputs_embeds: tf.Tensor | None = ...,
    ) -> tf.Tensor: ...

class TFBlipAttention(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        head_mask: tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        training: bool | None = ...,
    ) -> tuple[tf.Tensor, tf.Tensor | None, tuple[tf.Tensor] | None]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFBlipMLP(keras.layers.Layer):
    def __init__(self, config: BlipConfig, **kwargs) -> None: ...
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFBlipEncoderLayer(keras.layers.Layer):
    def __init__(self, config: BlipConfig, **kwargs) -> None: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        output_attentions: bool | None = ...,
        training: bool | None = ...,
    ) -> tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFBlipPreTrainedModel(TFPreTrainedModel):
    config_class = BlipConfig
    base_model_prefix = ...
    _keys_to_ignore_on_load_missing = ...

BLIP_START_DOCSTRING = ...
BLIP_VISION_INPUTS_DOCSTRING = ...
BLIP_INPUTS_DOCSTRING = ...

@keras_serializable
class TFBlipEncoder(keras.layers.Layer):
    config_class = BlipConfig
    def __init__(self, config: BlipConfig, **kwargs) -> None: ...
    @unpack_inputs
    def call(
        self,
        inputs_embeds,
        attention_mask: tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool | None = ...,
    ) -> tuple | TFBaseModelOutput: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFBlipVisionModel(TFBlipPreTrainedModel):
    main_input_name = ...
    config_class = BlipVisionConfig
    def __init__(self, config: BlipVisionConfig, *args, **kwargs) -> None: ...
    def serving_output(
        self, output: TFBaseModelOutputWithPooling
    ) -> TFBaseModelOutputWithPooling: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(BLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=TFBaseModelOutputWithPooling, config_class=BlipVisionConfig
    )
    def call(
        self,
        pixel_values: tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool | None = ...,
    ) -> tuple | TFBaseModelOutputWithPooling: ...
    def get_input_embeddings(self):  # -> TFBlipVisionEmbeddings:
        ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFBlipMainLayer(keras.layers.Layer):
    config_class = BlipConfig
    def __init__(self, config: BlipConfig, *args, **kwargs) -> None: ...
    def build(self, input_shape=...):  # -> None:
        ...
    @unpack_inputs
    def call(
        self,
        input_ids: tf.Tensor | None = ...,
        pixel_values: tf.Tensor | None = ...,
        attention_mask: tf.Tensor | None = ...,
        position_ids: tf.Tensor | None = ...,
        return_loss: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool | None = ...,
    ) -> tuple | TFBlipOutput: ...

class TFBlipModel(TFBlipPreTrainedModel):
    config_class = BlipConfig
    _keys_to_ignore_on_load_missing = ...
    main_input_name = ...
    def __init__(self, config: BlipConfig, *inputs, **kwargs) -> None: ...
    def serving_output(self, output: TFBlipOutput) -> TFBlipOutput: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(BLIP_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBlipOutput, config_class=BlipConfig)
    def call(
        self,
        input_ids: tf.Tensor | None = ...,
        pixel_values: tf.Tensor | None = ...,
        attention_mask: tf.Tensor | None = ...,
        position_ids: tf.Tensor | None = ...,
        return_loss: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool | None = ...,
    ) -> tuple | TFBlipOutput: ...
    @add_start_docstrings_to_model_forward(BLIP_TEXT_INPUTS_DOCSTRING)
    def get_text_features(
        self,
        input_ids: tf.Tensor | None = ...,
        attention_mask: tf.Tensor | None = ...,
        position_ids: tf.Tensor | None = ...,
        return_dict: bool | None = ...,
    ) -> tf.Tensor: ...
    @add_start_docstrings_to_model_forward(BLIP_VISION_INPUTS_DOCSTRING)
    def get_image_features(
        self, pixel_values: tf.Tensor | None = ..., return_dict: bool | None = ...
    ) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    BLIP_START_DOCSTRING,
)
class TFBlipForConditionalGeneration(TFBlipPreTrainedModel):
    config_class = BlipConfig
    _keys_to_ignore_on_load_missing = ...
    main_input_name = ...
    def __init__(self, config: BlipConfig, *args, **kwargs) -> None: ...
    def get_input_embeddings(self) -> keras.layers.Layer: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(BLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=TFBlipForConditionalGenerationModelOutput, config_class=BlipConfig
    )
    def call(
        self,
        pixel_values: tf.Tensor,
        input_ids: tf.Tensor | None = ...,
        attention_mask: tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        labels: tf.Tensor | None = ...,
        return_dict: bool | None = ...,
        training: bool | None = ...,
    ) -> tuple | TFBlipForConditionalGenerationModelOutput: ...
    def generate(
        self,
        pixel_values: tf.Tensor,
        input_ids: tf.Tensor | None = ...,
        attention_mask: tf.Tensor | None = ...,
        **generate_kwargs,
    ) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    BLIP_START_DOCSTRING,
)
class TFBlipForQuestionAnswering(TFBlipPreTrainedModel):
    config_class = BlipConfig
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config: BlipConfig, *args, **kwargs) -> None: ...
    def get_input_embeddings(self) -> keras.layers.Layer: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(BLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=TFBlipTextVisionModelOutput, config_class=BlipVisionConfig
    )
    def call(
        self,
        input_ids: tf.Tensor,
        pixel_values: tf.Tensor | None = ...,
        decoder_input_ids: tf.Tensor | None = ...,
        decoder_attention_mask: tf.Tensor | None = ...,
        attention_mask: tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        labels: tf.Tensor | None = ...,
        return_dict: bool | None = ...,
        training: bool | None = ...,
    ) -> tuple | TFBlipTextVisionModelOutput: ...
    def generate(
        self,
        input_ids: tf.Tensor,
        pixel_values: tf.Tensor,
        attention_mask: tf.Tensor | None = ...,
        **generate_kwargs,
    ) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    BLIP_START_DOCSTRING,
)
class TFBlipForImageTextRetrieval(TFBlipPreTrainedModel):
    config_class = BlipConfig
    def __init__(self, config: BlipConfig, *args, **kwargs) -> None: ...
    def get_input_embeddings(self) -> keras.layers.Layer: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(BLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=TFBlipImageTextMatchingModelOutput, config_class=BlipVisionConfig
    )
    def call(
        self,
        input_ids: tf.Tensor,
        pixel_values: tf.Tensor | None = ...,
        use_itm_head: bool | None = ...,
        attention_mask: tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool | None = ...,
    ) -> tuple | TFBlipImageTextMatchingModelOutput: ...
    def build(self, input_shape=...):  # -> None:
        ...

__all__ = [
    "TFBlipForConditionalGeneration",
    "TFBlipForImageTextRetrieval",
    "TFBlipForQuestionAnswering",
    "TFBlipModel",
    "TFBlipPreTrainedModel",
    "TFBlipTextModel",
    "TFBlipVisionModel",
]
