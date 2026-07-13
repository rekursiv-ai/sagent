import tensorflow as tf

from .configuration_resnet import ResNetConfig
from ...modeling_tf_outputs import (
    TFBaseModelOutputWithNoAttention,
    TFBaseModelOutputWithPoolingAndNoAttention,
    TFImageClassifierOutputWithNoAttention,
)
from ...modeling_tf_utils import (
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

"""TensorFlow ResNet model."""
logger = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
_EXPECTED_OUTPUT_SHAPE = ...
_IMAGE_CLASS_CHECKPOINT = ...
_IMAGE_CLASS_EXPECTED_OUTPUT = ...

class TFResNetConvLayer(keras.layers.Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = ...,
        stride: int = ...,
        activation: str = ...,
        **kwargs,
    ) -> None: ...
    def convolution(self, hidden_state: tf.Tensor) -> tf.Tensor: ...
    def call(self, hidden_state: tf.Tensor, training: bool = ...) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFResNetEmbeddings(keras.layers.Layer):
    def __init__(self, config: ResNetConfig, **kwargs) -> None: ...
    def call(self, pixel_values: tf.Tensor, training: bool = ...) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFResNetShortCut(keras.layers.Layer):
    def __init__(
        self, in_channels: int, out_channels: int, stride: int = ..., **kwargs
    ) -> None: ...
    def call(self, x: tf.Tensor, training: bool = ...) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFResNetBasicLayer(keras.layers.Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = ...,
        activation: str = ...,
        **kwargs,
    ) -> None: ...
    def call(self, hidden_state: tf.Tensor, training: bool = ...) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFResNetBottleNeckLayer(keras.layers.Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = ...,
        activation: str = ...,
        reduction: int = ...,
        **kwargs,
    ) -> None: ...
    def call(self, hidden_state: tf.Tensor, training: bool = ...) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFResNetStage(keras.layers.Layer):
    def __init__(
        self,
        config: ResNetConfig,
        in_channels: int,
        out_channels: int,
        stride: int = ...,
        depth: int = ...,
        **kwargs,
    ) -> None: ...
    def call(self, hidden_state: tf.Tensor, training: bool = ...) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFResNetEncoder(keras.layers.Layer):
    def __init__(self, config: ResNetConfig, **kwargs) -> None: ...
    def call(
        self,
        hidden_state: tf.Tensor,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
        training: bool = ...,
    ) -> TFBaseModelOutputWithNoAttention: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFResNetPreTrainedModel(TFPreTrainedModel):
    config_class = ResNetConfig
    base_model_prefix = ...
    main_input_name = ...
    @property
    def input_signature(self):  # -> dict[str, Any]:
        ...

RESNET_START_DOCSTRING = ...
RESNET_INPUTS_DOCSTRING = ...

@keras_serializable
class TFResNetMainLayer(keras.layers.Layer):
    config_class = ResNetConfig
    def __init__(self, config: ResNetConfig, **kwargs) -> None: ...
    @unpack_inputs
    def call(
        self,
        pixel_values: tf.Tensor,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> tuple[tf.Tensor] | TFBaseModelOutputWithPoolingAndNoAttention: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    RESNET_START_DOCSTRING,
)
class TFResNetModel(TFResNetPreTrainedModel):
    def __init__(self, config: ResNetConfig, **kwargs) -> None: ...
    @add_start_docstrings_to_model_forward(RESNET_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPoolingAndNoAttention,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    @unpack_inputs
    def call(
        self,
        pixel_values: tf.Tensor,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> tuple[tf.Tensor] | TFBaseModelOutputWithPoolingAndNoAttention: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    RESNET_START_DOCSTRING,
)
class TFResNetForImageClassification(
    TFResNetPreTrainedModel, TFSequenceClassificationLoss
):
    def __init__(self, config: ResNetConfig, **kwargs) -> None: ...
    def classifier(self, x: tf.Tensor) -> tf.Tensor: ...
    @add_start_docstrings_to_model_forward(RESNET_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=TFImageClassifierOutputWithNoAttention,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    @unpack_inputs
    def call(
        self,
        pixel_values: tf.Tensor | None = ...,
        labels: tf.Tensor | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> tuple[tf.Tensor] | TFImageClassifierOutputWithNoAttention: ...
    def build(self, input_shape=...):  # -> None:
        ...

__all__ = ["TFResNetForImageClassification", "TFResNetModel", "TFResNetPreTrainedModel"]
