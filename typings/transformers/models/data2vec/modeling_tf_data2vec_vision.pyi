from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from .configuration_data2vec_vision import Data2VecVisionConfig
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFBaseModelOutputWithPooling,
    TFSemanticSegmenterOutput,
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
    replace_return_docstrings,
)

"""TF 2.0 Data2Vec Vision model."""
logger = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
_EXPECTED_OUTPUT_SHAPE = ...
_IMAGE_CLASS_CHECKPOINT = ...
_IMAGE_CLASS_EXPECTED_OUTPUT = ...

@dataclass
class TFData2VecVisionModelOutputWithPooling(TFBaseModelOutputWithPooling):
    last_hidden_state: tf.Tensor | None = ...
    pooler_output: tf.Tensor | None = ...
    hidden_states: tuple[tf.Tensor] | None = ...
    attentions: tuple[tf.Tensor] | None = ...

class TFData2VecVisionDropPath(keras.layers.Layer):
    def __init__(self, drop_path, **kwargs) -> None: ...
    def call(self, x, training=...): ...

class TFData2VecVisionEmbeddings(keras.layers.Layer):
    def __init__(self, config: Data2VecVisionConfig, **kwargs) -> None: ...
    def build(self, input_shape=...):  # -> None:
        ...
    def call(
        self, pixel_values: tf.Tensor, bool_masked_pos: tf.Tensor | None = ...
    ) -> tf.Tensor: ...

class TFData2VecVisionPatchEmbeddings(keras.layers.Layer):
    def __init__(self, config: Data2VecVisionConfig, **kwargs) -> None: ...
    def call(self, pixel_values: tf.Tensor, training: bool = ...) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFData2VecVisionSelfAttention(keras.layers.Layer):
    def __init__(
        self, config: Data2VecVisionConfig, window_size: tuple | None = ..., **kwargs
    ) -> None: ...
    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        relative_position_bias: TFData2VecVisionRelativePositionBias | None = ...,
        training: bool = ...,
    ) -> tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFData2VecVisionSelfOutput(keras.layers.Layer):
    def __init__(self, config: Data2VecVisionConfig, **kwargs) -> None: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        input_tensor: tf.Tensor,
        gamma=...,
        training: bool = ...,
    ) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFData2VecVisionAttention(keras.layers.Layer):
    def __init__(
        self, config: Data2VecVisionConfig, window_size: tuple | None = ..., **kwargs
    ) -> None: ...
    def prune_heads(self, heads): ...
    def call(
        self,
        input_tensor: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        relative_position_bias: TFData2VecVisionRelativePositionBias | None = ...,
        training: bool = ...,
    ) -> tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFData2VecVisionIntermediate(keras.layers.Layer):
    def __init__(self, config: Data2VecVisionConfig, **kwargs) -> None: ...
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFData2VecVisionOutput(keras.layers.Layer):
    def __init__(self, config: Data2VecVisionConfig, **kwargs) -> None: ...
    def call(self, hidden_states: tf.Tensor, training: bool = ...) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFData2VecVisionLayer(keras.layers.Layer):
    def __init__(
        self,
        config: Data2VecVisionConfig,
        window_size: tuple | None = ...,
        drop_path_rate: float = ...,
        **kwargs,
    ) -> None: ...
    def build(self, input_shape: tf.TensorShape = ...):  # -> None:
        ...
    def call(
        self,
        hidden_states: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        relative_position_bias: TFData2VecVisionRelativePositionBias | None = ...,
        training: bool = ...,
    ) -> tuple[tf.Tensor]: ...

class TFData2VecVisionRelativePositionBias(keras.layers.Layer):
    def __init__(
        self, config: Data2VecVisionConfig, window_size: tuple, **kwargs
    ) -> None: ...
    def build(self, input_shape):  # -> None:
        ...
    def get_position_index(self): ...
    def call(self, inputs=...) -> tf.Tensor: ...

class TFData2VecVisionEncoder(keras.layers.Layer):
    def __init__(
        self, config: Data2VecVisionConfig, window_size: tuple | None = ..., **kwargs
    ) -> None: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        head_mask: tf.Tensor | None = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ) -> tuple | TFBaseModelOutput: ...
    def build(self, input_shape=...):  # -> None:
        ...

@keras_serializable
class TFData2VecVisionMainLayer(keras.layers.Layer):
    config_class = Data2VecVisionConfig
    def __init__(
        self, config: Data2VecVisionConfig, add_pooling_layer: bool = ..., **kwargs
    ) -> None: ...
    def get_input_embeddings(self) -> keras.layers.Layer: ...
    @unpack_inputs
    def call(
        self,
        pixel_values: tf.Tensor | None = ...,
        bool_masked_pos: tf.Tensor | None = ...,
        head_mask: tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> tuple | TFData2VecVisionModelOutputWithPooling: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFData2VecVisionPooler(keras.layers.Layer):
    def __init__(self, config: Data2VecVisionConfig, **kwargs) -> None: ...
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFData2VecVisionPreTrainedModel(TFPreTrainedModel):
    config_class = Data2VecVisionConfig
    base_model_prefix = ...
    main_input_name = ...
    _keys_to_ignore_on_load_unexpected = ...

DATA2VEC_VISION_START_DOCSTRING = ...
DATA2VEC_VISION_INPUTS_DOCSTRING = ...

@add_start_docstrings(
    ...,
    DATA2VEC_VISION_START_DOCSTRING,
)
class TFData2VecVisionModel(TFData2VecVisionPreTrainedModel):
    def __init__(
        self,
        config: Data2VecVisionConfig,
        add_pooling_layer: bool = ...,
        *inputs,
        **kwargs,
    ) -> None: ...
    def get_input_embeddings(self): ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(DATA2VEC_VISION_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFData2VecVisionModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def call(
        self,
        pixel_values: TFModelInputType | None = ...,
        bool_masked_pos: tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> tuple | TFData2VecVisionModelOutputWithPooling: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    DATA2VEC_VISION_START_DOCSTRING,
)
class TFData2VecVisionForImageClassification(
    TFData2VecVisionPreTrainedModel, TFSequenceClassificationLoss
):
    def __init__(self, config: Data2VecVisionConfig, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(DATA2VEC_VISION_INPUTS_DOCSTRING)
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
        return_dict: bool | None = ...,
        labels: np.ndarray | tf.Tensor | None = ...,
        training: bool | None = ...,
    ) -> TFSequenceClassifierOutput | tuple: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFData2VecVisionConvModule(keras.layers.Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        padding: str = ...,
        bias: bool = ...,
        dilation: int | tuple[int, int] = ...,
        **kwargs,
    ) -> None: ...
    def call(self, input: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFAdaptiveAvgPool2D(keras.layers.Layer):
    def __init__(
        self, output_dims: tuple[int, int], input_ordering: str = ..., **kwargs
    ) -> None: ...
    def pseudo_1d_pool(self, inputs: tf.Tensor, h_pooling: bool): ...
    def call(self, inputs: tf.Tensor): ...

class TFData2VecVisionPyramidPoolingModule(keras.layers.Layer):
    def __init__(
        self,
        pool_scales: tuple[int, ...],
        in_channels: int,
        out_channels: int,
        **kwargs,
    ) -> None: ...
    def call(self, x: tf.Tensor) -> list[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFData2VecVisionUperHead(keras.layers.Layer):
    def __init__(self, config: Data2VecVisionConfig, **kwargs) -> None: ...
    def psp_forward(self, inputs): ...
    def call(self, encoder_hidden_states: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFData2VecVisionFCNHead(keras.layers.Layer):
    def __init__(
        self,
        config: Data2VecVisionConfig,
        in_index: int = ...,
        kernel_size: int = ...,
        dilation: int | tuple[int, int] = ...,
        **kwargs,
    ) -> None: ...
    def call(self, encoder_hidden_states: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    DATA2VEC_VISION_START_DOCSTRING,
)
class TFData2VecVisionForSemanticSegmentation(TFData2VecVisionPreTrainedModel):
    def __init__(self, config: Data2VecVisionConfig, *inputs, **kwargs) -> None: ...
    def compute_loss(self, logits, auxiliary_logits, labels): ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(DATA2VEC_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=TFSemanticSegmenterOutput, config_class=_CONFIG_FOR_DOC
    )
    def call(
        self,
        pixel_values: tf.Tensor | None = ...,
        head_mask: tf.Tensor | None = ...,
        labels: tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | TFSemanticSegmenterOutput: ...
    def build(self, input_shape=...):  # -> None:
        ...

__all__ = [
    "TFData2VecVisionForImageClassification",
    "TFData2VecVisionForSemanticSegmentation",
    "TFData2VecVisionModel",
    "TFData2VecVisionPreTrainedModel",
]
