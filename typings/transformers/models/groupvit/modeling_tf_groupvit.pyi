from dataclasses import dataclass
from typing import Any

import numpy as np
import tensorflow as tf

from .configuration_groupvit import (
    GroupViTConfig,
    GroupViTTextConfig,
    GroupViTVisionConfig,
)
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
    is_tensorflow_probability_available,
    replace_return_docstrings,
)

"""TF 2.0 GroupViT model."""
logger = ...
if is_tensorflow_probability_available():
    _ = ...
else:
    _ = ...
_CHECKPOINT_FOR_DOC = ...
LARGE_NEGATIVE = ...

def contrastive_loss(logits: tf.Tensor) -> tf.Tensor: ...
def groupvit_loss(similarity: tf.Tensor) -> tf.Tensor: ...
def hard_softmax(logits: tf.Tensor, dim: int) -> tf.Tensor: ...
def gumbel_softmax(
    logits: tf.Tensor, tau: float = ..., hard: bool = ..., dim: int = ...
) -> tf.Tensor: ...
def resize_attention_map(
    attentions: tf.Tensor, height: int, width: int, align_corners: bool = ...
) -> tf.Tensor: ...
def get_grouping_from_attentions(
    attentions: tuple[tf.Tensor], hw_shape: tuple[int]
) -> tf.Tensor: ...

@dataclass
class TFGroupViTModelOutput(ModelOutput):
    loss: tf.Tensor | None = ...
    logits_per_image: tf.Tensor | None = ...
    logits_per_text: tf.Tensor | None = ...
    segmentation_logits: tf.Tensor | None = ...
    text_embeds: tf.Tensor | None = ...
    image_embeds: tf.Tensor | None = ...
    text_model_output: TFBaseModelOutputWithPooling = ...
    vision_model_output: TFBaseModelOutputWithPooling = ...
    def to_tuple(self) -> tuple[Any]: ...

class TFGroupViTCrossAttentionLayer(keras.layers.Layer):
    def __init__(self, config: GroupViTVisionConfig, **kwargs) -> None: ...
    def call(
        self, query: tf.Tensor, key: tf.Tensor, training: bool = ...
    ) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFGroupViTAssignAttention(keras.layers.Layer):
    def __init__(self, config: GroupViTVisionConfig, **kwargs) -> None: ...
    def get_attn(
        self,
        attn: tf.Tensor,
        gumbel: bool = ...,
        hard: bool = ...,
        training: bool = ...,
    ) -> tf.Tensor: ...
    def call(
        self, query: tf.Tensor, key: tf.Tensor, training: bool = ...
    ):  # -> tuple[Any, Any]:
        ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFGroupViTTokenAssign(keras.layers.Layer):
    def __init__(
        self,
        config: GroupViTVisionConfig,
        num_group_token: int,
        num_output_group: int,
        **kwargs,
    ) -> None: ...
    def project_group_token(self, group_tokens: tf.Tensor) -> tf.Tensor: ...
    def call(
        self, image_tokens: tf.Tensor, group_tokens: tf.Tensor, training: bool = ...
    ):  # -> tuple[Any, Any]:
        ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFGroupViTPatchEmbeddings(keras.layers.Layer):
    def __init__(self, config: GroupViTConfig, **kwargs) -> None: ...
    def call(
        self,
        pixel_values: tf.Tensor,
        interpolate_pos_encoding: bool = ...,
        training: bool = ...,
    ) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFGroupViTVisionEmbeddings(keras.layers.Layer):
    def __init__(self, config: GroupViTVisionConfig, **kwargs) -> None: ...
    def build(self, input_shape=...):  # -> None:
        ...
    def interpolate_pos_encoding(self, embeddings, height, width) -> tf.Tensor: ...
    def call(
        self,
        pixel_values: tf.Tensor,
        interpolate_pos_encoding: bool = ...,
        training: bool = ...,
    ) -> tf.Tensor: ...

class TFGroupViTTextEmbeddings(keras.layers.Layer):
    def __init__(self, config: GroupViTTextConfig, **kwargs) -> None: ...
    def build(self, input_shape: tf.TensorShape = ...):  # -> None:
        ...
    def call(
        self,
        input_ids: tf.Tensor | None = ...,
        position_ids: tf.Tensor | None = ...,
        inputs_embeds: tf.Tensor | None = ...,
    ) -> tf.Tensor: ...

class TFGroupViTStage(keras.layers.Layer):
    def __init__(
        self,
        config: GroupViTVisionConfig,
        depth: int,
        num_prev_group_token: int,
        num_group_token: int,
        num_output_group: int,
        **kwargs,
    ) -> None: ...
    def build(self, input_shape=...):  # -> None:
        ...
    @property
    def with_group_token(self):  # -> bool:
        ...
    def split_x(self, x: tf.Tensor) -> tf.Tensor: ...
    def concat_x(
        self, x: tf.Tensor, group_token: tf.Tensor | None = ...
    ) -> tf.Tensor: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        prev_group_token: tf.Tensor | None = ...,
        output_attentions: bool = ...,
        training: bool = ...,
    ) -> tuple[tf.Tensor]: ...

class TFGroupViTMLP(keras.layers.Layer):
    def __init__(
        self,
        config: GroupViTVisionConfig,
        hidden_size: int | None = ...,
        intermediate_size: int | None = ...,
        output_size: int | None = ...,
        **kwargs,
    ) -> None: ...
    def call(self, hidden_states: tf.Tensor, training: bool = ...) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFGroupViTMixerMLP(TFGroupViTMLP):
    def call(self, x, training: bool = ...): ...

class TFGroupViTAttention(keras.layers.Layer):
    def __init__(self, config: GroupViTConfig, **kwargs) -> None: ...
    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor | None = ...,
        causal_attention_mask: tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        encoder_hidden_states: tf.Tensor | None = ...,
        training: bool = ...,
    ) -> tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFGroupViTEncoderLayer(keras.layers.Layer):
    def __init__(self, config: GroupViTConfig, **kwargs) -> None: ...
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

class TFGroupViTTextEncoder(keras.layers.Layer):
    def __init__(self, config: GroupViTTextConfig, **kwargs) -> None: ...
    def call(
        self,
        hidden_states,
        attention_mask: tf.Tensor,
        causal_attention_mask: tf.Tensor,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        training: bool = ...,
    ) -> tuple | TFBaseModelOutput: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFGroupViTVisionEncoder(keras.layers.Layer):
    def __init__(self, config: GroupViTVisionConfig, **kwargs) -> None: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        output_hidden_states: bool,
        output_attentions: bool,
        return_dict: bool,
        training: bool = ...,
    ) -> tuple | TFBaseModelOutput: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFGroupViTTextTransformer(keras.layers.Layer):
    def __init__(self, config: GroupViTTextConfig, **kwargs) -> None: ...
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

class TFGroupViTVisionTransformer(keras.layers.Layer):
    def __init__(self, config: GroupViTVisionConfig, **kwargs) -> None: ...
    def call(
        self,
        pixel_values: TFModelInputType,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        training: bool = ...,
    ) -> tuple | TFBaseModelOutputWithPooling: ...
    def build(self, input_shape=...):  # -> None:
        ...

@keras_serializable
class TFGroupViTTextMainLayer(keras.layers.Layer):
    config_class = GroupViTTextConfig
    def __init__(self, config: GroupViTTextConfig, **kwargs) -> None: ...
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

@keras_serializable
class TFGroupViTVisionMainLayer(keras.layers.Layer):
    config_class = GroupViTVisionConfig
    def __init__(self, config: GroupViTVisionConfig, **kwargs) -> None: ...
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
class TFGroupViTMainLayer(keras.layers.Layer):
    config_class = GroupViTConfig
    def __init__(self, config: GroupViTConfig, **kwargs) -> None: ...
    def build(self, input_shape=...):  # -> None:
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
        output_segmentation: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> TFGroupViTModelOutput | tuple[tf.Tensor]: ...

class TFGroupViTPreTrainedModel(TFPreTrainedModel):
    config_class = GroupViTConfig
    base_model_prefix = ...

GROUPVIT_START_DOCSTRING = ...
GROUPVIT_TEXT_INPUTS_DOCSTRING = ...
GROUPVIT_VISION_INPUTS_DOCSTRING = ...
GROUPVIT_INPUTS_DOCSTRING = ...

class TFGroupViTTextModel(TFGroupViTPreTrainedModel):
    config_class = GroupViTTextConfig
    main_input_name = ...
    def __init__(self, config: GroupViTTextConfig, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        GROUPVIT_TEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @replace_return_docstrings(
        output_type=TFBaseModelOutputWithPooling, config_class=GroupViTTextConfig
    )
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

class TFGroupViTVisionModel(TFGroupViTPreTrainedModel):
    config_class = GroupViTVisionConfig
    main_input_name = ...
    def __init__(self, config: GroupViTVisionConfig, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(GROUPVIT_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=TFBaseModelOutputWithPooling, config_class=GroupViTVisionConfig
    )
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

@add_start_docstrings(GROUPVIT_START_DOCSTRING)
class TFGroupViTModel(TFGroupViTPreTrainedModel):
    config_class = GroupViTConfig
    def __init__(self, config: GroupViTConfig, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        GROUPVIT_TEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
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
    @add_start_docstrings_to_model_forward(GROUPVIT_VISION_INPUTS_DOCSTRING)
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
        GROUPVIT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @replace_return_docstrings(
        output_type=TFGroupViTModelOutput, config_class=GroupViTConfig
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        pixel_values: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        return_loss: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        output_segmentation: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> TFGroupViTModelOutput | tuple[tf.Tensor]: ...
    def serving_output(
        self, output: TFGroupViTModelOutput
    ) -> TFGroupViTModelOutput: ...
    def build(self, input_shape=...):  # -> None:
        ...

__all__ = [
    "TFGroupViTModel",
    "TFGroupViTPreTrainedModel",
    "TFGroupViTTextModel",
    "TFGroupViTVisionModel",
]
