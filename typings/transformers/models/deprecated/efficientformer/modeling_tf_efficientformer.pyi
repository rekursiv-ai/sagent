from dataclasses import dataclass

import tensorflow as tf

from .configuration_efficientformer import EfficientFormerConfig
from ....modeling_tf_outputs import (
    TFBaseModelOutput,
    TFBaseModelOutputWithPooling,
    TFImageClassifierOutput,
)
from ....modeling_tf_utils import (
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ....utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)

"""TensorFlow EfficientFormer model."""
logger = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
_EXPECTED_OUTPUT_SHAPE = ...
_IMAGE_CLASS_CHECKPOINT = ...
_IMAGE_CLASS_EXPECTED_OUTPUT = ...

class TFEfficientFormerPatchEmbeddings(keras.layers.Layer):
    def __init__(
        self,
        config: EfficientFormerConfig,
        num_channels: int,
        embed_dim: int,
        apply_norm: bool = ...,
        **kwargs,
    ) -> None: ...
    def call(self, pixel_values: tf.Tensor, training: bool = ...) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFEfficientFormerSelfAttention(keras.layers.Layer):
    def __init__(
        self,
        dim: int,
        key_dim: int,
        num_heads: int,
        attention_ratio: int,
        resolution: int,
        config: EfficientFormerConfig,
        **kwargs,
    ) -> None: ...
    def build(self, input_shape: tf.TensorShape) -> None: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        output_attentions: bool = ...,
        training: bool = ...,
    ) -> tuple[tf.Tensor]: ...

class TFEfficientFormerConvStem(keras.layers.Layer):
    def __init__(
        self, config: EfficientFormerConfig, out_channels: int, **kwargs
    ) -> None: ...
    def call(self, pixel_values: tf.Tensor, training: bool = ...) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFEfficientFormerPooling(keras.layers.Layer):
    def __init__(self, pool_size: int, **kwargs) -> None: ...
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor: ...

class TFEfficientFormerDenseMlp(keras.layers.Layer):
    def __init__(
        self,
        config: EfficientFormerConfig,
        in_features: int,
        hidden_features: int | None = ...,
        out_features: int | None = ...,
        **kwargs,
    ) -> None: ...
    def call(self, hidden_states: tf.Tensor, training: bool = ...) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFEfficientFormerConvMlp(keras.layers.Layer):
    def __init__(
        self,
        config: EfficientFormerConfig,
        in_features: int,
        hidden_features: int | None = ...,
        out_features: int | None = ...,
        drop: float = ...,
        **kwargs,
    ) -> None: ...
    def call(self, hidden_state: tf.Tensor, training: bool = ...) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFEfficientFormerDropPath(keras.layers.Layer):
    def __init__(self, drop_path: float, **kwargs) -> None: ...
    def call(self, x: tf.Tensor, training=...): ...

class TFEfficientFormerFlat(keras.layers.Layer):
    def __init__(self, **kwargs) -> None: ...
    def call(self, hidden_states: tf.Tensor) -> tuple[tf.Tensor]: ...

class TFEfficientFormerMeta3D(keras.layers.Layer):
    def __init__(
        self, config: EfficientFormerConfig, dim: int, drop_path: float = ..., **kwargs
    ) -> None: ...
    def build(self, input_shape=...):  # -> None:
        ...
    def call(
        self,
        hidden_states: tf.Tensor,
        output_attentions: bool = ...,
        training: bool = ...,
    ) -> tuple[tf.Tensor]: ...

class TFEfficientFormerMeta3DLayers(keras.layers.Layer):
    def __init__(self, config: EfficientFormerConfig, **kwargs) -> None: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        output_attentions: bool = ...,
        training: bool = ...,
    ) -> tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFEfficientFormerMeta4D(keras.layers.Layer):
    def __init__(
        self, config: EfficientFormerConfig, dim: int, drop_path: float = ..., **kwargs
    ) -> None: ...
    def build(self, input_shape=...):  # -> None:
        ...
    def call(
        self, hidden_states: tf.Tensor, training: bool = ...
    ) -> tuple[tf.Tensor]: ...

class TFEfficientFormerMeta4DLayers(keras.layers.Layer):
    def __init__(
        self, config: EfficientFormerConfig, stage_idx: int, **kwargs
    ) -> None: ...
    def call(
        self, hidden_states: tf.Tensor, training: bool = ...
    ) -> tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFEfficientFormerIntermediateStage(keras.layers.Layer):
    def __init__(self, config: EfficientFormerConfig, index: int, **kwargs) -> None: ...
    def call(
        self, hidden_states: tf.Tensor, training: bool = ...
    ) -> tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFEfficientFormerLastStage(keras.layers.Layer):
    def __init__(self, config: EfficientFormerConfig, **kwargs) -> None: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        output_attentions: bool = ...,
        training: bool = ...,
    ) -> tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFEfficientFormerEncoder(keras.layers.Layer):
    def __init__(self, config: EfficientFormerConfig, **kwargs) -> None: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        output_hidden_states: bool,
        output_attentions: bool,
        return_dict: bool,
        training: bool = ...,
    ) -> TFBaseModelOutput: ...
    def build(self, input_shape=...):  # -> None:
        ...

@keras_serializable
class TFEfficientFormerMainLayer(keras.layers.Layer):
    config_class = EfficientFormerConfig
    def __init__(self, config: EfficientFormerConfig, **kwargs) -> None: ...
    @unpack_inputs
    def call(
        self,
        pixel_values: tf.Tensor | None = ...,
        output_attentions: tf.Tensor | None = ...,
        output_hidden_states: tf.Tensor | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> TFBaseModelOutput | tuple[tf.Tensor, ...]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFEfficientFormerPreTrainedModel(TFPreTrainedModel):
    config_class = EfficientFormerConfig
    base_model_prefix = ...
    main_input_name = ...

EFFICIENTFORMER_START_DOCSTRING = ...
EFFICIENTFORMER_INPUTS_DOCSTRING = ...

@add_start_docstrings(
    ...,
    EFFICIENTFORMER_START_DOCSTRING,
)
class TFEfficientFormerModel(TFEfficientFormerPreTrainedModel):
    def __init__(self, config: EfficientFormerConfig, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(EFFICIENTFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def call(
        self,
        pixel_values: tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> tuple | TFBaseModelOutput: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    EFFICIENTFORMER_START_DOCSTRING,
)
class TFEfficientFormerForImageClassification(
    TFEfficientFormerPreTrainedModel, TFSequenceClassificationLoss
):
    def __init__(self, config: EfficientFormerConfig) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(EFFICIENTFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=TFImageClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def call(
        self,
        pixel_values: tf.Tensor | None = ...,
        labels: tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> tf.Tensor | TFImageClassifierOutput: ...
    def build(self, input_shape=...):  # -> None:
        ...

@dataclass
class TFEfficientFormerForImageClassificationWithTeacherOutput(ModelOutput):
    logits: tf.Tensor | None = ...
    cls_logits: tf.Tensor | None = ...
    distillation_logits: tf.Tensor | None = ...
    hidden_states: tuple[tf.Tensor] | None = ...
    attentions: tuple[tf.Tensor] | None = ...

@add_start_docstrings(
    ...,
    EFFICIENTFORMER_START_DOCSTRING,
)
class TFEfficientFormerForImageClassificationWithTeacher(
    TFEfficientFormerPreTrainedModel
):
    def __init__(self, config: EfficientFormerConfig) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(EFFICIENTFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=TFEfficientFormerForImageClassificationWithTeacherOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def call(
        self,
        pixel_values: tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> tuple | TFEfficientFormerForImageClassificationWithTeacherOutput: ...
    def build(self, input_shape=...):  # -> None:
        ...

__all__ = [
    "TFEfficientFormerForImageClassification",
    "TFEfficientFormerForImageClassificationWithTeacher",
    "TFEfficientFormerModel",
    "TFEfficientFormerPreTrainedModel",
]
