import tensorflow as tf

from .configuration_mobilevit import MobileViTConfig
from ...file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFBaseModelOutputWithPooling,
    TFImageClassifierOutputWithNoAttention,
    TFSemanticSegmenterOutputWithNoAttention,
)
from ...modeling_tf_utils import (
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    keras,
    keras_serializable,
    unpack_inputs,
)

"""TensorFlow 2.0 MobileViT model."""
logger = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
_EXPECTED_OUTPUT_SHAPE = ...
_IMAGE_CLASS_CHECKPOINT = ...
_IMAGE_CLASS_EXPECTED_OUTPUT = ...

def make_divisible(
    value: int, divisor: int = ..., min_value: int | None = ...
) -> int: ...

class TFMobileViTConvLayer(keras.layers.Layer):
    def __init__(
        self,
        config: MobileViTConfig,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = ...,
        groups: int = ...,
        bias: bool = ...,
        dilation: int = ...,
        use_normalization: bool = ...,
        use_activation: bool | str = ...,
        **kwargs,
    ) -> None: ...
    def call(self, features: tf.Tensor, training: bool = ...) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFMobileViTInvertedResidual(keras.layers.Layer):
    def __init__(
        self,
        config: MobileViTConfig,
        in_channels: int,
        out_channels: int,
        stride: int,
        dilation: int = ...,
        **kwargs,
    ) -> None: ...
    def call(self, features: tf.Tensor, training: bool = ...) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFMobileViTMobileNetLayer(keras.layers.Layer):
    def __init__(
        self,
        config: MobileViTConfig,
        in_channels: int,
        out_channels: int,
        stride: int = ...,
        num_stages: int = ...,
        **kwargs,
    ) -> None: ...
    def call(self, features: tf.Tensor, training: bool = ...) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFMobileViTSelfAttention(keras.layers.Layer):
    def __init__(self, config: MobileViTConfig, hidden_size: int, **kwargs) -> None: ...
    def transpose_for_scores(self, x: tf.Tensor) -> tf.Tensor: ...
    def call(self, hidden_states: tf.Tensor, training: bool = ...) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFMobileViTSelfOutput(keras.layers.Layer):
    def __init__(self, config: MobileViTConfig, hidden_size: int, **kwargs) -> None: ...
    def call(self, hidden_states: tf.Tensor, training: bool = ...) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFMobileViTAttention(keras.layers.Layer):
    def __init__(self, config: MobileViTConfig, hidden_size: int, **kwargs) -> None: ...
    def prune_heads(self, heads): ...
    def call(self, hidden_states: tf.Tensor, training: bool = ...) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFMobileViTIntermediate(keras.layers.Layer):
    def __init__(
        self,
        config: MobileViTConfig,
        hidden_size: int,
        intermediate_size: int,
        **kwargs,
    ) -> None: ...
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFMobileViTOutput(keras.layers.Layer):
    def __init__(
        self,
        config: MobileViTConfig,
        hidden_size: int,
        intermediate_size: int,
        **kwargs,
    ) -> None: ...
    def call(
        self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = ...
    ) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFMobileViTTransformerLayer(keras.layers.Layer):
    def __init__(
        self,
        config: MobileViTConfig,
        hidden_size: int,
        intermediate_size: int,
        **kwargs,
    ) -> None: ...
    def call(self, hidden_states: tf.Tensor, training: bool = ...) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFMobileViTTransformer(keras.layers.Layer):
    def __init__(
        self, config: MobileViTConfig, hidden_size: int, num_stages: int, **kwargs
    ) -> None: ...
    def call(self, hidden_states: tf.Tensor, training: bool = ...) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFMobileViTLayer(keras.layers.Layer):
    def __init__(
        self,
        config: MobileViTConfig,
        in_channels: int,
        out_channels: int,
        stride: int,
        hidden_size: int,
        num_stages: int,
        dilation: int = ...,
        **kwargs,
    ) -> None: ...
    def unfolding(self, features: tf.Tensor) -> tuple[tf.Tensor, dict]: ...
    def folding(self, patches: tf.Tensor, info_dict: dict) -> tf.Tensor: ...
    def call(self, features: tf.Tensor, training: bool = ...) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFMobileViTEncoder(keras.layers.Layer):
    def __init__(self, config: MobileViTConfig, **kwargs) -> None: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
        training: bool = ...,
    ) -> tuple | TFBaseModelOutput: ...
    def build(self, input_shape=...):  # -> None:
        ...

@keras_serializable
class TFMobileViTMainLayer(keras.layers.Layer):
    config_class = MobileViTConfig
    def __init__(
        self, config: MobileViTConfig, expand_output: bool = ..., **kwargs
    ) -> None: ...
    @unpack_inputs
    def call(
        self,
        pixel_values: tf.Tensor | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> tuple[tf.Tensor] | TFBaseModelOutputWithPooling: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFMobileViTPreTrainedModel(TFPreTrainedModel):
    config_class = MobileViTConfig
    base_model_prefix = ...
    main_input_name = ...

MOBILEVIT_START_DOCSTRING = ...
MOBILEVIT_INPUTS_DOCSTRING = ...

@add_start_docstrings(
    ...,
    MOBILEVIT_START_DOCSTRING,
)
class TFMobileViTModel(TFMobileViTPreTrainedModel):
    def __init__(
        self, config: MobileViTConfig, expand_output: bool = ..., *inputs, **kwargs
    ) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(MOBILEVIT_INPUTS_DOCSTRING)
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
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> tuple[tf.Tensor] | TFBaseModelOutputWithPooling: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    MOBILEVIT_START_DOCSTRING,
)
class TFMobileViTForImageClassification(
    TFMobileViTPreTrainedModel, TFSequenceClassificationLoss
):
    def __init__(self, config: MobileViTConfig, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(MOBILEVIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=TFImageClassifierOutputWithNoAttention,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def call(
        self,
        pixel_values: tf.Tensor | None = ...,
        output_hidden_states: bool | None = ...,
        labels: tf.Tensor | None = ...,
        return_dict: bool | None = ...,
        training: bool | None = ...,
    ) -> tuple | TFImageClassifierOutputWithNoAttention: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFMobileViTASPPPooling(keras.layers.Layer):
    def __init__(
        self, config: MobileViTConfig, in_channels: int, out_channels: int, **kwargs
    ) -> None: ...
    def call(self, features: tf.Tensor, training: bool = ...) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFMobileViTASPP(keras.layers.Layer):
    def __init__(self, config: MobileViTConfig, **kwargs) -> None: ...
    def call(self, features: tf.Tensor, training: bool = ...) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFMobileViTDeepLabV3(keras.layers.Layer):
    def __init__(self, config: MobileViTConfig, **kwargs) -> None: ...
    def call(self, hidden_states: tf.Tensor, training: bool = ...) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    MOBILEVIT_START_DOCSTRING,
)
class TFMobileViTForSemanticSegmentation(TFMobileViTPreTrainedModel):
    def __init__(self, config: MobileViTConfig, **kwargs) -> None: ...
    def hf_compute_loss(self, logits, labels): ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(MOBILEVIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=TFSemanticSegmenterOutputWithNoAttention,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        pixel_values: tf.Tensor | None = ...,
        labels: tf.Tensor | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> tuple | TFSemanticSegmenterOutputWithNoAttention: ...
    def build(self, input_shape=...):  # -> None:
        ...

__all__ = [
    "TFMobileViTForImageClassification",
    "TFMobileViTForSemanticSegmentation",
    "TFMobileViTModel",
    "TFMobileViTPreTrainedModel",
]
