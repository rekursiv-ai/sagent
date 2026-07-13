import numpy as np
import tensorflow as tf

from .configuration_convnextv2 import ConvNextV2Config
from ...modeling_tf_outputs import (
    TFBaseModelOutputWithNoAttention,
    TFBaseModelOutputWithPooling,
    TFBaseModelOutputWithPoolingAndNoAttention,
    TFImageClassifierOutputWithNoAttention,
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

"""TF 2.0 ConvNextV2 model."""
logger = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
_EXPECTED_OUTPUT_SHAPE = ...
_IMAGE_CLASS_CHECKPOINT = ...
_IMAGE_CLASS_EXPECTED_OUTPUT = ...

class TFConvNextV2DropPath(keras.layers.Layer):
    def __init__(self, drop_path: float, **kwargs) -> None: ...
    def call(self, x: tf.Tensor, training=...): ...

class TFConvNextV2GRN(keras.layers.Layer):
    def __init__(self, config: ConvNextV2Config, dim: int, **kwargs) -> None: ...
    def build(self, input_shape: tf.TensorShape = ...): ...
    def call(self, hidden_states: tf.Tensor): ...

class TFConvNextV2Embeddings(keras.layers.Layer):
    def __init__(self, config: ConvNextV2Config, **kwargs) -> None: ...
    def call(self, pixel_values): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFConvNextV2Layer(keras.layers.Layer):
    def __init__(
        self, config: ConvNextV2Config, dim: int, drop_path: float = ..., **kwargs
    ) -> None: ...
    def call(self, hidden_states, training=...): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFConvNextV2Stage(keras.layers.Layer):
    def __init__(
        self,
        config: ConvNextV2Config,
        in_channels: int,
        out_channels: int,
        kernel_size: int = ...,
        stride: int = ...,
        depth: int = ...,
        drop_path_rates: list[float] | None = ...,
        **kwargs,
    ) -> None: ...
    def call(self, hidden_states): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFConvNextV2Encoder(keras.layers.Layer):
    def __init__(self, config: ConvNextV2Config, **kwargs) -> None: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | TFBaseModelOutputWithNoAttention: ...
    def build(self, input_shape=...):  # -> None:
        ...

@keras_serializable
class TFConvNextV2MainLayer(keras.layers.Layer):
    config_class = ConvNextV2Config
    def __init__(self, config: ConvNextV2Config, **kwargs) -> None: ...
    @unpack_inputs
    def call(
        self,
        pixel_values: TFModelInputType | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> TFBaseModelOutputWithPooling | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFConvNextV2PreTrainedModel(TFPreTrainedModel):
    config_class = ConvNextV2Config
    base_model_prefix = ...
    main_input_name = ...

CONVNEXTV2_START_DOCSTRING = ...
CONVNEXTV2_INPUTS_DOCSTRING = ...

@add_start_docstrings(
    ...,
    CONVNEXTV2_START_DOCSTRING,
)
class TFConvNextV2Model(TFConvNextV2PreTrainedModel):
    def __init__(self, config: ConvNextV2Config, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(CONVNEXTV2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPoolingAndNoAttention,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def call(
        self,
        pixel_values: TFModelInputType | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> TFBaseModelOutputWithPoolingAndNoAttention | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    CONVNEXTV2_START_DOCSTRING,
)
class TFConvNextV2ForImageClassification(
    TFConvNextV2PreTrainedModel, TFSequenceClassificationLoss
):
    def __init__(self, config: ConvNextV2Config, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(CONVNEXTV2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=TFImageClassifierOutputWithNoAttention,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def call(
        self,
        pixel_values: TFModelInputType | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: np.ndarray | tf.Tensor | None = ...,
        training: bool | None = ...,
    ) -> TFImageClassifierOutputWithNoAttention | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

__all__ = [
    "TFConvNextV2ForImageClassification",
    "TFConvNextV2Model",
    "TFConvNextV2PreTrainedModel",
]
