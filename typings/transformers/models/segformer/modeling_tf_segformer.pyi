import tensorflow as tf

from .configuration_segformer import SegformerConfig
from ...file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFSemanticSegmenterOutput,
    TFSequenceClassifierOutput,
)
from ...modeling_tf_utils import (
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    keras,
    keras_serializable,
    unpack_inputs,
)

"""TensorFlow SegFormer model."""
logger = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
_EXPECTED_OUTPUT_SHAPE = ...
_IMAGE_CLASS_CHECKPOINT = ...
_IMAGE_CLASS_EXPECTED_OUTPUT = ...

class TFSegformerDropPath(keras.layers.Layer):
    def __init__(self, drop_path: float, **kwargs) -> None: ...
    def call(self, x: tf.Tensor, training=...): ...

class TFSegformerOverlapPatchEmbeddings(keras.layers.Layer):
    def __init__(
        self, patch_size, stride, num_channels, hidden_size, **kwargs
    ) -> None: ...
    def call(self, pixel_values: tf.Tensor) -> tuple[tf.Tensor, int, int]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFSegformerEfficientSelfAttention(keras.layers.Layer):
    def __init__(
        self,
        config: SegformerConfig,
        hidden_size: int,
        num_attention_heads: int,
        sequence_reduction_ratio: int,
        **kwargs,
    ) -> None: ...
    def transpose_for_scores(self, tensor: tf.Tensor) -> tf.Tensor: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        height: int,
        width: int,
        output_attentions: bool = ...,
        training: bool = ...,
    ) -> tf.Tensor | tuple[tf.Tensor, tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFSegformerSelfOutput(keras.layers.Layer):
    def __init__(self, config: SegformerConfig, hidden_size: int, **kwargs) -> None: ...
    def call(self, hidden_states: tf.Tensor, training: bool = ...) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFSegformerAttention(keras.layers.Layer):
    def __init__(
        self,
        config: SegformerConfig,
        hidden_size: int,
        num_attention_heads: int,
        sequence_reduction_ratio: int,
        **kwargs,
    ) -> None: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        height: int,
        width: int,
        output_attentions: bool = ...,
    ) -> tf.Tensor | tuple[tf.Tensor, tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFSegformerDWConv(keras.layers.Layer):
    def __init__(self, dim: int = ..., **kwargs) -> None: ...
    def call(self, hidden_states: tf.Tensor, height: int, width: int) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFSegformerMixFFN(keras.layers.Layer):
    def __init__(
        self,
        config: SegformerConfig,
        in_features: int,
        hidden_features: int | None = ...,
        out_features: int | None = ...,
        **kwargs,
    ) -> None: ...
    def call(
        self, hidden_states: tf.Tensor, height: int, width: int, training: bool = ...
    ) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFSegformerLayer(keras.layers.Layer):
    def __init__(
        self,
        config,
        hidden_size: int,
        num_attention_heads: int,
        drop_path: float,
        sequence_reduction_ratio: int,
        mlp_ratio: int,
        **kwargs,
    ) -> None: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        height: int,
        width: int,
        output_attentions: bool = ...,
        training: bool = ...,
    ) -> tuple: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFSegformerEncoder(keras.layers.Layer):
    def __init__(self, config: SegformerConfig, **kwargs) -> None: ...
    def call(
        self,
        pixel_values: tf.Tensor,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> tuple | TFBaseModelOutput: ...
    def build(self, input_shape=...):  # -> None:
        ...

@keras_serializable
class TFSegformerMainLayer(keras.layers.Layer):
    config_class = SegformerConfig
    def __init__(self, config: SegformerConfig, **kwargs) -> None: ...
    @unpack_inputs
    def call(
        self,
        pixel_values: tf.Tensor,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> tuple | TFBaseModelOutput: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFSegformerPreTrainedModel(TFPreTrainedModel):
    config_class = SegformerConfig
    base_model_prefix = ...
    main_input_name = ...
    @property
    def input_signature(self):  # -> dict[str, Any]:
        ...

SEGFORMER_START_DOCSTRING = ...
SEGFORMER_INPUTS_DOCSTRING = ...

@add_start_docstrings(
    ...,
    SEGFORMER_START_DOCSTRING,
)
class TFSegformerModel(TFSegformerPreTrainedModel):
    def __init__(self, config: SegformerConfig, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        SEGFORMER_INPUTS_DOCSTRING.format("(batch_size, sequence_length)")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def call(
        self,
        pixel_values: tf.Tensor,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> tuple | TFBaseModelOutput: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    SEGFORMER_START_DOCSTRING,
)
class TFSegformerForImageClassification(
    TFSegformerPreTrainedModel, TFSequenceClassificationLoss
):
    def __init__(self, config: SegformerConfig, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        SEGFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
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
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | TFSequenceClassifierOutput: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFSegformerMLP(keras.layers.Layer):
    def __init__(self, input_dim: int, config: SegformerConfig, **kwargs) -> None: ...
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFSegformerDecodeHead(TFSegformerPreTrainedModel):
    def __init__(self, config: SegformerConfig, **kwargs) -> None: ...
    def call(
        self, encoder_hidden_states: tf.Tensor, training: bool = ...
    ) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    SEGFORMER_START_DOCSTRING,
)
class TFSegformerForSemanticSegmentation(TFSegformerPreTrainedModel):
    def __init__(self, config: SegformerConfig, **kwargs) -> None: ...
    def hf_compute_loss(self, logits, labels): ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        SEGFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @replace_return_docstrings(
        output_type=TFSemanticSegmenterOutput, config_class=_CONFIG_FOR_DOC
    )
    def call(
        self,
        pixel_values: tf.Tensor,
        labels: tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | TFSemanticSegmenterOutput: ...
    def build(self, input_shape=...):  # -> None:
        ...

__all__ = [
    "TFSegformerDecodeHead",
    "TFSegformerForImageClassification",
    "TFSegformerForSemanticSegmentation",
    "TFSegformerModel",
    "TFSegformerPreTrainedModel",
]
