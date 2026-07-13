import numpy as np
import tensorflow as tf

from .configuration_convnext import ConvNextConfig
from ...modeling_tf_outputs import (
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
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)

"""TF 2.0 ConvNext model."""
logger = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...

class TFConvNextDropPath(keras.layers.Layer):
    def __init__(self, drop_path: float, **kwargs) -> None: ...
    def call(self, x: tf.Tensor, training=...): ...

class TFConvNextEmbeddings(keras.layers.Layer):
    def __init__(self, config: ConvNextConfig, **kwargs) -> None: ...
    def call(self, pixel_values): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFConvNextLayer(keras.layers.Layer):
    def __init__(self, config, dim, drop_path=..., **kwargs) -> None: ...
    def build(self, input_shape: tf.TensorShape = ...):  # -> None:
        ...
    def call(self, hidden_states, training=...): ...

class TFConvNextStage(keras.layers.Layer):
    def __init__(
        self,
        config: ConvNextConfig,
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

class TFConvNextEncoder(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
    def call(
        self, hidden_states, output_hidden_states=..., return_dict=...
    ):  # -> tuple[Any | tuple[Any, ...] | tuple[()], ...] | TFBaseModelOutput:
        ...
    def build(self, input_shape=...):  # -> None:
        ...

@keras_serializable
class TFConvNextMainLayer(keras.layers.Layer):
    config_class = ConvNextConfig
    def __init__(
        self, config: ConvNextConfig, add_pooling_layer: bool = ..., **kwargs
    ) -> None: ...
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

class TFConvNextPreTrainedModel(TFPreTrainedModel):
    config_class = ConvNextConfig
    base_model_prefix = ...
    main_input_name = ...

CONVNEXT_START_DOCSTRING = ...
CONVNEXT_INPUTS_DOCSTRING = ...

@add_start_docstrings(
    ...,
    CONVNEXT_START_DOCSTRING,
)
class TFConvNextModel(TFConvNextPreTrainedModel):
    def __init__(self, config, *inputs, add_pooling_layer=..., **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(CONVNEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=TFBaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC
    )
    def call(
        self,
        pixel_values: TFModelInputType | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> TFBaseModelOutputWithPooling | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    CONVNEXT_START_DOCSTRING,
)
class TFConvNextForImageClassification(
    TFConvNextPreTrainedModel, TFSequenceClassificationLoss
):
    def __init__(self, config: ConvNextConfig, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(CONVNEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=TFSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC
    )
    def call(
        self,
        pixel_values: TFModelInputType | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: np.ndarray | tf.Tensor | None = ...,
        training: bool | None = ...,
    ) -> TFSequenceClassifierOutput | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

__all__ = [
    "TFConvNextForImageClassification",
    "TFConvNextModel",
    "TFConvNextPreTrainedModel",
]
