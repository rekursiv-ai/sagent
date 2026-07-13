import tensorflow as tf

from .configuration_regnet import RegNetConfig
from ...file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from ...modeling_tf_outputs import (
    TFBaseModelOutputWithNoAttention,
    TFBaseModelOutputWithPoolingAndNoAttention,
    TFSequenceClassifierOutput,
)
from ...modeling_tf_utils import (
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    keras,
    keras_serializable,
    unpack_inputs,
)

"""TensorFlow RegNet model."""
logger = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
_EXPECTED_OUTPUT_SHAPE = ...
_IMAGE_CLASS_CHECKPOINT = ...
_IMAGE_CLASS_EXPECTED_OUTPUT = ...

class TFRegNetConvLayer(keras.layers.Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = ...,
        stride: int = ...,
        groups: int = ...,
        activation: str | None = ...,
        **kwargs,
    ) -> None: ...
    def call(self, hidden_state): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFRegNetEmbeddings(keras.layers.Layer):
    def __init__(self, config: RegNetConfig, **kwargs) -> None: ...
    def call(self, pixel_values): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFRegNetShortCut(keras.layers.Layer):
    def __init__(
        self, in_channels: int, out_channels: int, stride: int = ..., **kwargs
    ) -> None: ...
    def call(self, inputs: tf.Tensor, training: bool = ...) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFRegNetSELayer(keras.layers.Layer):
    def __init__(self, in_channels: int, reduced_channels: int, **kwargs) -> None: ...
    def call(self, hidden_state): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFRegNetXLayer(keras.layers.Layer):
    def __init__(
        self,
        config: RegNetConfig,
        in_channels: int,
        out_channels: int,
        stride: int = ...,
        **kwargs,
    ) -> None: ...
    def call(self, hidden_state): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFRegNetYLayer(keras.layers.Layer):
    def __init__(
        self,
        config: RegNetConfig,
        in_channels: int,
        out_channels: int,
        stride: int = ...,
        **kwargs,
    ) -> None: ...
    def call(self, hidden_state): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFRegNetStage(keras.layers.Layer):
    def __init__(
        self,
        config: RegNetConfig,
        in_channels: int,
        out_channels: int,
        stride: int = ...,
        depth: int = ...,
        **kwargs,
    ) -> None: ...
    def call(self, hidden_state): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFRegNetEncoder(keras.layers.Layer):
    def __init__(self, config: RegNetConfig, **kwargs) -> None: ...
    def call(
        self,
        hidden_state: tf.Tensor,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ) -> TFBaseModelOutputWithNoAttention: ...
    def build(self, input_shape=...):  # -> None:
        ...

@keras_serializable
class TFRegNetMainLayer(keras.layers.Layer):
    config_class = RegNetConfig
    def __init__(self, config, **kwargs) -> None: ...
    @unpack_inputs
    def call(
        self,
        pixel_values: tf.Tensor,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> TFBaseModelOutputWithPoolingAndNoAttention: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFRegNetPreTrainedModel(TFPreTrainedModel):
    config_class = RegNetConfig
    base_model_prefix = ...
    main_input_name = ...
    @property
    def input_signature(self):  # -> dict[str, Any]:
        ...

REGNET_START_DOCSTRING = ...
REGNET_INPUTS_DOCSTRING = ...

@add_start_docstrings(
    ...,
    REGNET_START_DOCSTRING,
)
class TFRegNetModel(TFRegNetPreTrainedModel):
    def __init__(self, config: RegNetConfig, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(REGNET_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPoolingAndNoAttention,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def call(
        self,
        pixel_values: tf.Tensor,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> TFBaseModelOutputWithPoolingAndNoAttention | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    REGNET_START_DOCSTRING,
)
class TFRegNetForImageClassification(
    TFRegNetPreTrainedModel, TFSequenceClassificationLoss
):
    def __init__(self, config: RegNetConfig, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(REGNET_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def call(
        self,
        pixel_values: tf.Tensor | None = ...,
        labels: tf.Tensor | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> TFSequenceClassifierOutput | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

__all__ = ["TFRegNetForImageClassification", "TFRegNetModel", "TFRegNetPreTrainedModel"]
