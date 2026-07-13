import numpy as np
import tensorflow as tf

from .configuration_vit import ViTConfig
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFBaseModelOutputWithPooling,
    TFSequenceClassifierOutput,
)
from ...modeling_tf_utils import (
    TFModelInputType,
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)

"""TF 2.0 ViT model."""
logger = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
_EXPECTED_OUTPUT_SHAPE = ...
_IMAGE_CLASS_CHECKPOINT = ...
_IMAGE_CLASS_EXPECTED_OUTPUT = ...

class TFViTEmbeddings(keras.layers.Layer):
    def __init__(self, config: ViTConfig, **kwargs) -> None: ...
    def build(self, input_shape=...):  # -> None:
        ...
    def interpolate_pos_encoding(self, embeddings, height, width) -> tf.Tensor: ...
    def call(
        self,
        pixel_values: tf.Tensor,
        interpolate_pos_encoding: bool = ...,
        training: bool = ...,
    ) -> tf.Tensor: ...

class TFViTPatchEmbeddings(keras.layers.Layer):
    def __init__(self, config: ViTConfig, **kwargs) -> None: ...
    def call(
        self,
        pixel_values: tf.Tensor,
        interpolate_pos_encoding: bool = ...,
        training: bool = ...,
    ) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFViTSelfAttention(keras.layers.Layer):
    def __init__(self, config: ViTConfig, **kwargs) -> None: ...
    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = ...,
    ) -> tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFViTSelfOutput(keras.layers.Layer):
    def __init__(self, config: ViTConfig, **kwargs) -> None: ...
    def call(
        self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = ...
    ) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFViTAttention(keras.layers.Layer):
    def __init__(self, config: ViTConfig, **kwargs) -> None: ...
    def prune_heads(self, heads): ...
    def call(
        self,
        input_tensor: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = ...,
    ) -> tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFViTIntermediate(keras.layers.Layer):
    def __init__(self, config: ViTConfig, **kwargs) -> None: ...
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFViTOutput(keras.layers.Layer):
    def __init__(self, config: ViTConfig, **kwargs) -> None: ...
    def call(
        self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = ...
    ) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFViTLayer(keras.layers.Layer):
    def __init__(self, config: ViTConfig, **kwargs) -> None: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = ...,
    ) -> tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFViTEncoder(keras.layers.Layer):
    def __init__(self, config: ViTConfig, **kwargs) -> None: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        training: bool = ...,
    ) -> TFBaseModelOutput | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

@keras_serializable
class TFViTMainLayer(keras.layers.Layer):
    config_class = ViTConfig
    def __init__(
        self, config: ViTConfig, add_pooling_layer: bool = ..., **kwargs
    ) -> None: ...
    def get_input_embeddings(self) -> keras.layers.Layer: ...
    @unpack_inputs
    def call(
        self,
        pixel_values: TFModelInputType | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        interpolate_pos_encoding: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> TFBaseModelOutputWithPooling | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFViTPreTrainedModel(TFPreTrainedModel):
    config_class = ViTConfig
    base_model_prefix = ...
    main_input_name = ...

VIT_START_DOCSTRING = ...
VIT_INPUTS_DOCSTRING = ...

@add_start_docstrings(
    ...,
    VIT_START_DOCSTRING,
)
class TFViTModel(TFViTPreTrainedModel):
    def __init__(
        self, config: ViTConfig, *inputs, add_pooling_layer=..., **kwargs
    ) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(VIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def call(
        self,
        pixel_values: TFModelInputType | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        interpolate_pos_encoding: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> TFBaseModelOutputWithPooling | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFViTPooler(keras.layers.Layer):
    def __init__(self, config: ViTConfig, **kwargs) -> None: ...
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    VIT_START_DOCSTRING,
)
class TFViTForImageClassification(TFViTPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config: ViTConfig, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(VIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def call(
        self,
        pixel_values: TFModelInputType | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        interpolate_pos_encoding: bool | None = ...,
        return_dict: bool | None = ...,
        labels: np.ndarray | tf.Tensor | None = ...,
        training: bool | None = ...,
    ) -> TFSequenceClassifierOutput | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

__all__ = ["TFViTForImageClassification", "TFViTModel", "TFViTPreTrainedModel"]
