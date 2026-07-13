from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import tensorflow as tf

from .configuration_swin import SwinConfig
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

"""TF 2.0 Swin Transformer model."""
logger = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
_EXPECTED_OUTPUT_SHAPE = ...
_IMAGE_CLASS_CHECKPOINT = ...
_IMAGE_CLASS_EXPECTED_OUTPUT = ...

@dataclass
class TFSwinEncoderOutput(ModelOutput):
    last_hidden_state: tf.Tensor | None = ...
    hidden_states: tuple[tf.Tensor, ...] | None = ...
    attentions: tuple[tf.Tensor, ...] | None = ...
    reshaped_hidden_states: tuple[tf.Tensor, ...] | None = ...

@dataclass
class TFSwinModelOutput(ModelOutput):
    last_hidden_state: tf.Tensor | None = ...
    pooler_output: tf.Tensor | None = ...
    hidden_states: tuple[tf.Tensor, ...] | None = ...
    attentions: tuple[tf.Tensor, ...] | None = ...
    reshaped_hidden_states: tuple[tf.Tensor, ...] | None = ...

@dataclass
class TFSwinMaskedImageModelingOutput(ModelOutput):
    loss: tf.Tensor | None = ...
    reconstruction: tf.Tensor | None = ...
    hidden_states: tuple[tf.Tensor, ...] | None = ...
    attentions: tuple[tf.Tensor, ...] | None = ...
    reshaped_hidden_states: tuple[tf.Tensor, ...] | None = ...
    @property
    def logits(self):  # -> None:
        ...

@dataclass
class TFSwinImageClassifierOutput(ModelOutput):
    loss: tf.Tensor | None = ...
    logits: tf.Tensor | None = ...
    hidden_states: tuple[tf.Tensor, ...] | None = ...
    attentions: tuple[tf.Tensor, ...] | None = ...
    reshaped_hidden_states: tuple[tf.Tensor, ...] | None = ...

def window_partition(input_feature: tf.Tensor, window_size: int) -> tf.Tensor: ...
def window_reverse(
    windows: tf.Tensor, window_size: int, height: int, width: int
) -> tf.Tensor: ...
def drop_path(
    input: tf.Tensor,
    drop_prob: float = ...,
    training: bool = ...,
    scale_by_keep: bool = ...,
) -> tf.Tensor: ...

class TFSwinEmbeddings(keras.layers.Layer):
    def __init__(
        self, config: SwinConfig, use_mask_token: bool = ..., **kwargs
    ) -> None: ...
    def build(self, input_shape: tf.TensorShape) -> None: ...
    def call(
        self,
        pixel_values: tf.Tensor,
        bool_masked_pos: bool | None = ...,
        training: bool = ...,
    ) -> tuple[tf.Tensor, tuple[int, int]]: ...

class TFSwinPatchEmbeddings(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
    def maybe_pad(
        self, pixel_values: tf.Tensor, height: int, width: int
    ) -> tf.Tensor: ...
    def call(
        self, pixel_values: tf.Tensor, training: bool = ...
    ) -> tuple[tf.Tensor, tuple[int, int]]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFSwinPatchMerging(keras.layers.Layer):
    def __init__(
        self,
        input_resolution: tuple[int, int],
        dim: int,
        norm_layer: Callable | None = ...,
        **kwargs,
    ) -> None: ...
    def maybe_pad(
        self, input_feature: tf.Tensor, height: int, width: int
    ) -> tf.Tensor: ...
    def call(
        self,
        input_feature: tf.Tensor,
        input_dimensions: tuple[int, int],
        training: bool = ...,
    ) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFSwinDropPath(keras.layers.Layer):
    def __init__(
        self, drop_prob: float | None = ..., scale_by_keep: bool = ..., **kwargs
    ) -> None: ...
    def call(self, input: tf.Tensor, training: bool = ...) -> tf.Tensor: ...

class TFSwinSelfAttention(keras.layers.Layer):
    def __init__(
        self, config: SwinConfig, dim: int, num_heads: int, **kwargs
    ) -> None: ...
    def build(self, input_shape: tf.TensorShape) -> None: ...
    def transpose_for_scores(self, x: tf.Tensor) -> tf.Tensor: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor | None = ...,
        head_mask: tf.Tensor | None = ...,
        output_attentions: bool = ...,
        training: bool = ...,
    ) -> tuple[tf.Tensor, ...]: ...

class TFSwinSelfOutput(keras.layers.Layer):
    def __init__(self, config: SwinConfig, dim: int, **kwargs) -> None: ...
    def call(
        self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = ...
    ) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFSwinAttention(keras.layers.Layer):
    def __init__(
        self, config: SwinConfig, dim: int, num_heads: int, **kwargs
    ) -> None: ...
    def prune_heads(self, heads): ...
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor | None = ...,
        head_mask: tf.Tensor | None = ...,
        output_attentions: bool = ...,
        training: bool = ...,
    ) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFSwinIntermediate(keras.layers.Layer):
    def __init__(self, config: SwinConfig, dim: int, **kwargs) -> None: ...
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFSwinOutput(keras.layers.Layer):
    def __init__(self, config: SwinConfig, dim: int, **kwargs) -> None: ...
    def call(self, hidden_states: tf.Tensor, training: bool = ...) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFSwinLayer(keras.layers.Layer):
    def __init__(
        self,
        config,
        dim,
        input_resolution: tuple[int, int],
        num_heads: int,
        drop_path_rate: float = ...,
        shift_size: int = ...,
        **kwargs,
    ) -> None: ...
    def get_attn_mask(
        self, height: int, width: int, window_size: int, shift_size: int
    ) -> tf.Tensor | None: ...
    def maybe_pad(
        self, hidden_states: tf.Tensor, window_size: int, height: int, width: int
    ) -> tuple[tf.Tensor, tf.Tensor]: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        input_dimensions: tuple[int, int],
        head_mask: tf.Tensor | None = ...,
        output_attentions: bool = ...,
        training: bool = ...,
    ) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFSwinStage(keras.layers.Layer):
    def __init__(
        self,
        config: SwinConfig,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        drop_path: list[float],
        downsample: Callable | None,
        **kwargs,
    ) -> None: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        input_dimensions: tuple[int, int],
        head_mask: tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        training: bool = ...,
    ) -> tuple[tf.Tensor, ...]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFSwinEncoder(keras.layers.Layer):
    def __init__(
        self, config: SwinConfig, grid_size: tuple[int, int], **kwargs
    ) -> None: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        input_dimensions: tuple[int, int],
        head_mask: tf.Tensor | None = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
        training: bool = ...,
    ) -> tuple[tf.Tensor, ...] | TFSwinEncoderOutput: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFSwinPreTrainedModel(TFPreTrainedModel):
    config_class = SwinConfig
    base_model_prefix = ...
    main_input_name = ...

SWIN_START_DOCSTRING = ...
SWIN_INPUTS_DOCSTRING = ...

def normalize_data_format(value: str) -> str: ...

class AdaptiveAveragePooling1D(keras.layers.Layer):
    def __init__(
        self,
        output_size: int | Iterable[int],
        reduce_function: Callable = ...,
        data_format: str | None = ...,
        **kwargs,
    ) -> None: ...
    def call(self, inputs: tf.Tensor, *args) -> None: ...
    def compute_output_shape(self, input_shape: Iterable[int]) -> tf.TensorShape: ...
    def get_config(self) -> dict[str, Any]: ...

@keras_serializable
class TFSwinMainLayer(keras.layers.Layer):
    config_class = SwinConfig
    def __init__(
        self,
        config: SwinConfig,
        add_pooling_layer: bool = ...,
        use_mask_token: bool = ...,
        **kwargs,
    ) -> None: ...
    def get_input_embeddings(self) -> TFSwinPatchEmbeddings: ...
    def get_head_mask(self, head_mask: Any | None) -> list: ...
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
    ) -> TFSwinModelOutput | tuple[tf.Tensor, ...]: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    SWIN_START_DOCSTRING,
)
class TFSwinModel(TFSwinPreTrainedModel):
    def __init__(
        self,
        config: SwinConfig,
        add_pooling_layer: bool = ...,
        use_mask_token: bool = ...,
        **kwargs,
    ) -> None: ...
    @add_start_docstrings_to_model_forward(SWIN_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFSwinModelOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
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
    ) -> TFSwinModelOutput | tuple[tf.Tensor, ...]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFSwinPixelShuffle(keras.layers.Layer):
    def __init__(self, upscale_factor: int, **kwargs) -> None: ...
    def call(self, x: tf.Tensor) -> tf.Tensor: ...

class TFSwinDecoder(keras.layers.Layer):
    def __init__(self, config: SwinConfig, **kwargs) -> None: ...
    def call(self, x: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    SWIN_START_DOCSTRING,
)
class TFSwinForMaskedImageModeling(TFSwinPreTrainedModel):
    def __init__(self, config: SwinConfig) -> None: ...
    @add_start_docstrings_to_model_forward(SWIN_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=TFSwinMaskedImageModelingOutput, config_class=_CONFIG_FOR_DOC
    )
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
    ) -> tuple | TFSwinMaskedImageModelingOutput: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    SWIN_START_DOCSTRING,
)
class TFSwinForImageClassification(TFSwinPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config: SwinConfig) -> None: ...
    @add_start_docstrings_to_model_forward(SWIN_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=TFSwinImageClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    @unpack_inputs
    def call(
        self,
        pixel_values: tf.Tensor | None = ...,
        head_mask: tf.Tensor | None = ...,
        labels: tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> tuple[tf.Tensor, ...] | TFSwinImageClassifierOutput: ...
    def build(self, input_shape=...):  # -> None:
        ...

__all__ = [
    "TFSwinForImageClassification",
    "TFSwinForMaskedImageModeling",
    "TFSwinModel",
    "TFSwinPreTrainedModel",
]
