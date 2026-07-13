from dataclasses import dataclass

import tensorflow as tf

from .configuration_cvt import CvtConfig
from ...modeling_tf_outputs import TFImageClassifierOutputWithNoAttention
from ...modeling_tf_utils import (
    TFModelInputType,
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)

"""TF 2.0 Cvt model."""
logger = ...
_CONFIG_FOR_DOC = ...

@dataclass
class TFBaseModelOutputWithCLSToken(ModelOutput):
    last_hidden_state: tf.Tensor | None = ...
    cls_token_value: tf.Tensor | None = ...
    hidden_states: tuple[tf.Tensor, ...] | None = ...

class TFCvtDropPath(keras.layers.Layer):
    def __init__(self, drop_prob: float, **kwargs) -> None: ...
    def call(self, x: tf.Tensor, training=...): ...

class TFCvtEmbeddings(keras.layers.Layer):
    def __init__(
        self,
        config: CvtConfig,
        patch_size: int,
        num_channels: int,
        embed_dim: int,
        stride: int,
        padding: int,
        dropout_rate: float,
        **kwargs,
    ) -> None: ...
    def call(self, pixel_values: tf.Tensor, training: bool = ...) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFCvtConvEmbeddings(keras.layers.Layer):
    def __init__(
        self,
        config: CvtConfig,
        patch_size: int,
        num_channels: int,
        embed_dim: int,
        stride: int,
        padding: int,
        **kwargs,
    ) -> None: ...
    def call(self, pixel_values: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFCvtSelfAttentionConvProjection(keras.layers.Layer):
    def __init__(
        self,
        config: CvtConfig,
        embed_dim: int,
        kernel_size: int,
        stride: int,
        padding: int,
        **kwargs,
    ) -> None: ...
    def call(self, hidden_state: tf.Tensor, training: bool = ...) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFCvtSelfAttentionLinearProjection(keras.layers.Layer):
    def call(self, hidden_state: tf.Tensor) -> tf.Tensor: ...

class TFCvtSelfAttentionProjection(keras.layers.Layer):
    def __init__(
        self,
        config: CvtConfig,
        embed_dim: int,
        kernel_size: int,
        stride: int,
        padding: int,
        projection_method: str = ...,
        **kwargs,
    ) -> None: ...
    def call(self, hidden_state: tf.Tensor, training: bool = ...) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFCvtSelfAttention(keras.layers.Layer):
    def __init__(
        self,
        config: CvtConfig,
        num_heads: int,
        embed_dim: int,
        kernel_size: int,
        stride_q: int,
        stride_kv: int,
        padding_q: int,
        padding_kv: int,
        qkv_projection_method: str,
        qkv_bias: bool,
        attention_drop_rate: float,
        with_cls_token: bool = ...,
        **kwargs,
    ) -> None: ...
    def rearrange_for_multi_head_attention(
        self, hidden_state: tf.Tensor
    ) -> tf.Tensor: ...
    def call(
        self, hidden_state: tf.Tensor, height: int, width: int, training: bool = ...
    ) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFCvtSelfOutput(keras.layers.Layer):
    def __init__(
        self, config: CvtConfig, embed_dim: int, drop_rate: float, **kwargs
    ) -> None: ...
    def call(self, hidden_state: tf.Tensor, training: bool = ...) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFCvtAttention(keras.layers.Layer):
    def __init__(
        self,
        config: CvtConfig,
        num_heads: int,
        embed_dim: int,
        kernel_size: int,
        stride_q: int,
        stride_kv: int,
        padding_q: int,
        padding_kv: int,
        qkv_projection_method: str,
        qkv_bias: bool,
        attention_drop_rate: float,
        drop_rate: float,
        with_cls_token: bool = ...,
        **kwargs,
    ) -> None: ...
    def prune_heads(self, heads): ...
    def call(
        self, hidden_state: tf.Tensor, height: int, width: int, training: bool = ...
    ): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFCvtIntermediate(keras.layers.Layer):
    def __init__(
        self, config: CvtConfig, embed_dim: int, mlp_ratio: int, **kwargs
    ) -> None: ...
    def call(self, hidden_state: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFCvtOutput(keras.layers.Layer):
    def __init__(
        self,
        config: CvtConfig,
        embed_dim: int,
        mlp_ratio: int,
        drop_rate: int,
        **kwargs,
    ) -> None: ...
    def call(
        self, hidden_state: tf.Tensor, input_tensor: tf.Tensor, training: bool = ...
    ) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFCvtLayer(keras.layers.Layer):
    def __init__(
        self,
        config: CvtConfig,
        num_heads: int,
        embed_dim: int,
        kernel_size: int,
        stride_q: int,
        stride_kv: int,
        padding_q: int,
        padding_kv: int,
        qkv_projection_method: str,
        qkv_bias: bool,
        attention_drop_rate: float,
        drop_rate: float,
        mlp_ratio: float,
        drop_path_rate: float,
        with_cls_token: bool = ...,
        **kwargs,
    ) -> None: ...
    def call(
        self, hidden_state: tf.Tensor, height: int, width: int, training: bool = ...
    ) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFCvtStage(keras.layers.Layer):
    def __init__(self, config: CvtConfig, stage: int, **kwargs) -> None: ...
    def call(
        self, hidden_state: tf.Tensor, training: bool = ...
    ):  # -> tuple[Any, Any | None]:
        ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFCvtEncoder(keras.layers.Layer):
    config_class = CvtConfig
    def __init__(self, config: CvtConfig, **kwargs) -> None: ...
    def call(
        self,
        pixel_values: TFModelInputType,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool | None = ...,
    ) -> TFBaseModelOutputWithCLSToken | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

@keras_serializable
class TFCvtMainLayer(keras.layers.Layer):
    config_class = CvtConfig
    def __init__(self, config: CvtConfig, **kwargs) -> None: ...
    @unpack_inputs
    def call(
        self,
        pixel_values: TFModelInputType | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool | None = ...,
    ) -> TFBaseModelOutputWithCLSToken | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFCvtPreTrainedModel(TFPreTrainedModel):
    config_class = CvtConfig
    base_model_prefix = ...
    main_input_name = ...

TFCVT_START_DOCSTRING = ...
TFCVT_INPUTS_DOCSTRING = ...

@add_start_docstrings(
    ...,
    TFCVT_START_DOCSTRING,
)
class TFCvtModel(TFCvtPreTrainedModel):
    def __init__(self, config: CvtConfig, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(TFCVT_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=TFBaseModelOutputWithCLSToken, config_class=_CONFIG_FOR_DOC
    )
    def call(
        self,
        pixel_values: tf.Tensor | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool | None = ...,
    ) -> TFBaseModelOutputWithCLSToken | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    TFCVT_START_DOCSTRING,
)
class TFCvtForImageClassification(TFCvtPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config: CvtConfig, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(TFCVT_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=TFImageClassifierOutputWithNoAttention, config_class=_CONFIG_FOR_DOC
    )
    def call(
        self,
        pixel_values: tf.Tensor | None = ...,
        labels: tf.Tensor | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool | None = ...,
    ) -> TFImageClassifierOutputWithNoAttention | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

__all__ = ["TFCvtForImageClassification", "TFCvtModel", "TFCvtPreTrainedModel"]
