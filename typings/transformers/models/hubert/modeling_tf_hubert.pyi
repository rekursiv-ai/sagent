from typing import Any

import tensorflow as tf

from .configuration_hubert import HubertConfig
from ...modeling_tf_outputs import TFBaseModelOutput, TFCausalLMOutput
from ...modeling_tf_utils import (
    TFPreTrainedModel,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)

"""TensorFlow Hubert model."""
logger = ...
_CONFIG_FOR_DOC = ...
LARGE_NEGATIVE = ...

class TFHubertGroupNorm(keras.layers.Layer):
    def __init__(
        self,
        groups: int = ...,
        axis: int = ...,
        epsilon: float = ...,
        center: bool = ...,
        scale: bool = ...,
        beta_initializer: keras.initializers.Initializer = ...,
        gamma_initializer: keras.initializers.Initializer = ...,
        beta_regularizer: keras.regularizers.Regularizer = ...,
        gamma_regularizer: keras.regularizers.Regularizer = ...,
        beta_constraint: keras.constraints.Constraint = ...,
        gamma_constraint: keras.constraints.Constraint = ...,
        **kwargs,
    ) -> None: ...
    def build(self, input_shape):  # -> None:
        ...
    def call(self, inputs): ...
    def get_config(self):  # -> dict[Any | str, Any | int | float | bool]:
        ...
    def compute_output_shape(self, input_shape): ...

class TFHubertWeightNormConv1D(keras.layers.Conv1D):
    def __init__(
        self, filters, kernel_size, groups, explicit_padding, **kwargs
    ) -> None: ...
    def build(self, input_shape):  # -> None:
        ...
    def call(self, inputs): ...

class TFHubertNoLayerNormConvLayer(keras.layers.Layer):
    def __init__(
        self, config: HubertConfig, layer_id: int = ..., **kwargs: Any
    ) -> None: ...
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFHubertLayerNormConvLayer(keras.layers.Layer):
    def __init__(
        self, config: HubertConfig, layer_id: int = ..., **kwargs: Any
    ) -> None: ...
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFHubertGroupNormConvLayer(keras.layers.Layer):
    def __init__(
        self, config: HubertConfig, layer_id: int = ..., **kwargs: Any
    ) -> None: ...
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFHubertPositionalConvEmbedding(keras.layers.Layer):
    def __init__(self, config: HubertConfig, **kwargs: Any) -> None: ...
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFHubertSamePadLayer(keras.layers.Layer):
    def __init__(self, num_conv_pos_embeddings, **kwargs) -> None: ...
    def call(self, hidden_states): ...

class TFHubertFeatureEncoder(keras.layers.Layer):
    def __init__(self, config: HubertConfig, **kwargs: Any) -> None: ...
    def call(self, input_values): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFHubertFeatureExtractor(TFHubertFeatureEncoder):
    def __init__(self, config, **kwargs) -> None: ...

class TFHubertFeatureProjection(keras.layers.Layer):
    def __init__(self, config: HubertConfig, **kwargs) -> None: ...
    def call(self, hidden_states: tf.Tensor, training: bool = ...) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFHubertAttention(keras.layers.Layer):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = ...,
        is_decoder: bool = ...,
        bias: bool = ...,
        **kwargs,
    ) -> None: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        key_value_states: tf.Tensor | None = ...,
        past_key_value: tuple[tuple[tf.Tensor]] | None = ...,
        attention_mask: tf.Tensor | None = ...,
        layer_head_mask: tf.Tensor | None = ...,
        training: bool | None = ...,
    ) -> tuple[tf.Tensor, tf.Tensor | None]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFHubertFeedForward(keras.layers.Layer):
    def __init__(self, config: HubertConfig, **kwargs) -> None: ...
    def call(self, hidden_states: tf.Tensor, training: bool = ...) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFHubertEncoderLayer(keras.layers.Layer):
    def __init__(self, config: HubertConfig, **kwargs) -> None: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        training: bool = ...,
    ) -> tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFHubertEncoderLayerStableLayerNorm(keras.layers.Layer):
    def __init__(self, config: HubertConfig, **kwargs) -> None: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        training: bool = ...,
    ) -> tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFHubertEncoder(keras.layers.Layer):
    def __init__(self, config: HubertConfig, **kwargs) -> None: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool | None = ...,
    ) -> TFBaseModelOutput | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFHubertEncoderStableLayerNorm(keras.layers.Layer):
    def __init__(self, config: HubertConfig, **kwargs) -> None: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool | None = ...,
    ) -> TFBaseModelOutput | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

@keras_serializable
class TFHubertMainLayer(keras.layers.Layer):
    config_class = HubertConfig
    def __init__(self, config: HubertConfig, **kwargs) -> None: ...
    def build(self, input_shape=...):  # -> None:
        ...
    @unpack_inputs
    def call(
        self,
        input_values: tf.Tensor,
        attention_mask: tf.Tensor | None = ...,
        token_type_ids: tf.Tensor | None = ...,
        position_ids: tf.Tensor | None = ...,
        head_mask: tf.Tensor | None = ...,
        inputs_embeds: tf.Tensor | None = ...,
        output_attentions: tf.Tensor | None = ...,
        output_hidden_states: tf.Tensor | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
        **kwargs: Any,
    ):  # -> TFBaseModelOutput:
        ...

class TFHubertPreTrainedModel(TFPreTrainedModel):
    config_class = HubertConfig
    base_model_prefix = ...
    main_input_name = ...
    @property
    def input_signature(self):  # -> dict[str, Any]:
        ...
    def __init__(self, config, *inputs, **kwargs) -> None: ...

HUBERT_START_DOCSTRING = ...
HUBERT_INPUTS_DOCSTRING = ...

@add_start_docstrings(
    ...,
    HUBERT_START_DOCSTRING,
)
class TFHubertModel(TFHubertPreTrainedModel):
    def __init__(self, config: HubertConfig, *inputs, **kwargs) -> None: ...
    @add_start_docstrings_to_model_forward(HUBERT_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=TFBaseModelOutput, config_class=_CONFIG_FOR_DOC
    )
    @unpack_inputs
    def call(
        self,
        input_values: tf.Tensor,
        attention_mask: tf.Tensor | None = ...,
        token_type_ids: tf.Tensor | None = ...,
        position_ids: tf.Tensor | None = ...,
        head_mask: tf.Tensor | None = ...,
        inputs_embeds: tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> TFBaseModelOutput | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    HUBERT_START_DOCSTRING,
)
class TFHubertForCTC(TFHubertPreTrainedModel):
    def __init__(self, config: HubertConfig, *inputs, **kwargs) -> None: ...
    def freeze_feature_extractor(self):  # -> None:
        ...
    def freeze_feature_encoder(self):  # -> None:
        ...
    @add_start_docstrings_to_model_forward(HUBERT_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=TFCausalLMOutput, config_class=_CONFIG_FOR_DOC
    )
    @unpack_inputs
    def call(
        self,
        input_values: tf.Tensor,
        attention_mask: tf.Tensor | None = ...,
        token_type_ids: tf.Tensor | None = ...,
        position_ids: tf.Tensor | None = ...,
        head_mask: tf.Tensor | None = ...,
        inputs_embeds: tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        labels: tf.Tensor | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool | None = ...,
    ) -> TFCausalLMOutput | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

__all__ = ["TFHubertForCTC", "TFHubertModel", "TFHubertPreTrainedModel"]
