from dataclasses import dataclass

import tensorflow as tf

from .configuration_deit import DeiTConfig
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFBaseModelOutputWithPooling,
    TFImageClassifierOutput,
    TFMaskedImageModelingOutput,
)
from ...modeling_tf_utils import (
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)

"""TensorFlow DeiT model."""
logger = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
_EXPECTED_OUTPUT_SHAPE = ...
_IMAGE_CLASS_CHECKPOINT = ...
_IMAGE_CLASS_EXPECTED_OUTPUT = ...

@dataclass
class TFDeiTForImageClassificationWithTeacherOutput(ModelOutput):
    logits: tf.Tensor | None = ...
    cls_logits: tf.Tensor | None = ...
    distillation_logits: tf.Tensor | None = ...
    hidden_states: tuple[tf.Tensor] | None = ...
    attentions: tuple[tf.Tensor] | None = ...

class TFDeiTEmbeddings(keras.layers.Layer):
    def __init__(
        self, config: DeiTConfig, use_mask_token: bool = ..., **kwargs
    ) -> None: ...
    def build(self, input_shape=...):  # -> None:
        ...
    def interpolate_pos_encoding(
        self, embeddings: tf.Tensor, height: int, width: int
    ) -> tf.Tensor: ...
    def call(
        self,
        pixel_values: tf.Tensor,
        bool_masked_pos: tf.Tensor | None = ...,
        training: bool = ...,
        interpolate_pos_encoding: bool = ...,
    ) -> tf.Tensor: ...

class TFDeiTPatchEmbeddings(keras.layers.Layer):
    def __init__(self, config: DeiTConfig, **kwargs) -> None: ...
    def call(self, pixel_values: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFDeiTSelfAttention(keras.layers.Layer):
    def __init__(self, config: DeiTConfig, **kwargs) -> None: ...
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

class TFDeiTSelfOutput(keras.layers.Layer):
    def __init__(self, config: DeiTConfig, **kwargs) -> None: ...
    def call(
        self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = ...
    ) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFDeiTAttention(keras.layers.Layer):
    def __init__(self, config: DeiTConfig, **kwargs) -> None: ...
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

class TFDeiTIntermediate(keras.layers.Layer):
    def __init__(self, config: DeiTConfig, **kwargs) -> None: ...
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFDeiTOutput(keras.layers.Layer):
    def __init__(self, config: DeiTConfig, **kwargs) -> None: ...
    def call(
        self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = ...
    ) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFDeiTLayer(keras.layers.Layer):
    def __init__(self, config: DeiTConfig, **kwargs) -> None: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = ...,
    ) -> tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFDeiTEncoder(keras.layers.Layer):
    def __init__(self, config: DeiTConfig, **kwargs) -> None: ...
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
class TFDeiTMainLayer(keras.layers.Layer):
    config_class = DeiTConfig
    def __init__(
        self,
        config: DeiTConfig,
        add_pooling_layer: bool = ...,
        use_mask_token: bool = ...,
        **kwargs,
    ) -> None: ...
    def get_input_embeddings(self) -> TFDeiTPatchEmbeddings: ...
    def get_head_mask(self, head_mask):  # -> list[None]:
        ...
    @unpack_inputs
    def call(
        self,
        pixel_values: tf.Tensor | None = ...,
        bool_masked_pos: tf.Tensor | None = ...,
        head_mask: tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        interpolate_pos_encoding: bool = ...,
        training: bool = ...,
    ) -> TFBaseModelOutputWithPooling | tuple[tf.Tensor, ...]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFDeiTPreTrainedModel(TFPreTrainedModel):
    config_class = DeiTConfig
    base_model_prefix = ...
    main_input_name = ...

DEIT_START_DOCSTRING = ...
DEIT_INPUTS_DOCSTRING = ...

@add_start_docstrings(
    ...,
    DEIT_START_DOCSTRING,
)
class TFDeiTModel(TFDeiTPreTrainedModel):
    def __init__(
        self,
        config: DeiTConfig,
        add_pooling_layer: bool = ...,
        use_mask_token: bool = ...,
        **kwargs,
    ) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(DEIT_INPUTS_DOCSTRING)
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
        bool_masked_pos: tf.Tensor | None = ...,
        head_mask: tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        interpolate_pos_encoding: bool = ...,
        training: bool = ...,
    ) -> tuple | TFBaseModelOutputWithPooling: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFDeiTPooler(keras.layers.Layer):
    def __init__(self, config: DeiTConfig, **kwargs) -> None: ...
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFDeitPixelShuffle(keras.layers.Layer):
    def __init__(self, upscale_factor: int, **kwargs) -> None: ...
    def call(self, x: tf.Tensor) -> tf.Tensor: ...

class TFDeitDecoder(keras.layers.Layer):
    def __init__(self, config: DeiTConfig, **kwargs) -> None: ...
    def call(self, inputs: tf.Tensor, training: bool = ...) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    DEIT_START_DOCSTRING,
)
class TFDeiTForMaskedImageModeling(TFDeiTPreTrainedModel):
    def __init__(self, config: DeiTConfig) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(DEIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=TFMaskedImageModelingOutput, config_class=_CONFIG_FOR_DOC
    )
    def call(
        self,
        pixel_values: tf.Tensor | None = ...,
        bool_masked_pos: tf.Tensor | None = ...,
        head_mask: tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        interpolate_pos_encoding: bool = ...,
        training: bool = ...,
    ) -> tuple | TFMaskedImageModelingOutput: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    DEIT_START_DOCSTRING,
)
class TFDeiTForImageClassification(TFDeiTPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config: DeiTConfig) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(DEIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=TFImageClassifierOutput, config_class=_CONFIG_FOR_DOC
    )
    def call(
        self,
        pixel_values: tf.Tensor | None = ...,
        head_mask: tf.Tensor | None = ...,
        labels: tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        interpolate_pos_encoding: bool = ...,
        training: bool = ...,
    ) -> tf.Tensor | TFImageClassifierOutput: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    DEIT_START_DOCSTRING,
)
class TFDeiTForImageClassificationWithTeacher(TFDeiTPreTrainedModel):
    def __init__(self, config: DeiTConfig) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(DEIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=TFDeiTForImageClassificationWithTeacherOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def call(
        self,
        pixel_values: tf.Tensor | None = ...,
        head_mask: tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        interpolate_pos_encoding: bool = ...,
        training: bool = ...,
    ) -> tuple | TFDeiTForImageClassificationWithTeacherOutput: ...
    def build(self, input_shape=...):  # -> None:
        ...

__all__ = [
    "TFDeiTForImageClassification",
    "TFDeiTForImageClassificationWithTeacher",
    "TFDeiTForMaskedImageModeling",
    "TFDeiTModel",
    "TFDeiTPreTrainedModel",
]
