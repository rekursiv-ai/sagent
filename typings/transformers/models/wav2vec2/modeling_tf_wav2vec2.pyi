from dataclasses import dataclass
from typing import Any

import tensorflow as tf

from .configuration_wav2vec2 import Wav2Vec2Config
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFCausalLMOutput,
    TFSequenceClassifierOutput,
)
from ...modeling_tf_utils import (
    TFPreTrainedModel,
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

"""TensorFlow Wav2Vec2 model."""
logger = ...
_HIDDEN_STATES_START_POSITION = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
LARGE_NEGATIVE = ...

@dataclass
class TFWav2Vec2BaseModelOutput(ModelOutput):
    last_hidden_state: tf.Tensor | None = ...
    extract_features: tf.Tensor | None = ...
    hidden_states: tuple[tf.Tensor] | None = ...
    attentions: tuple[tf.Tensor] | None = ...

class TFWav2Vec2GroupNorm(keras.layers.Layer):
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

class TFWav2Vec2WeightNormConv1D(keras.layers.Conv1D):
    def __init__(
        self, filters, kernel_size, groups, explicit_padding, **kwargs
    ) -> None: ...
    def build(self, input_shape):  # -> None:
        ...
    def call(self, inputs): ...

class TFWav2Vec2NoLayerNormConvLayer(keras.layers.Layer):
    def __init__(
        self, config: Wav2Vec2Config, layer_id: int = ..., **kwargs: Any
    ) -> None: ...
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFWav2Vec2LayerNormConvLayer(keras.layers.Layer):
    def __init__(
        self, config: Wav2Vec2Config, layer_id: int = ..., **kwargs: Any
    ) -> None: ...
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFWav2Vec2GroupNormConvLayer(keras.layers.Layer):
    def __init__(
        self, config: Wav2Vec2Config, layer_id: int = ..., **kwargs: Any
    ) -> None: ...
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFWav2Vec2PositionalConvEmbedding(keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, **kwargs: Any) -> None: ...
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFWav2Vec2SamePadLayer(keras.layers.Layer):
    def __init__(self, num_conv_pos_embeddings, **kwargs) -> None: ...
    def call(self, hidden_states): ...

class TFWav2Vec2FeatureEncoder(keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, **kwargs: Any) -> None: ...
    def call(self, input_values): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFWav2Vec2FeatureExtractor(TFWav2Vec2FeatureEncoder):
    def __init__(self, config, **kwargs) -> None: ...

class TFWav2Vec2FeatureProjection(keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, **kwargs) -> None: ...
    def call(self, hidden_states: tf.Tensor, training: bool = ...) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFWav2Vec2Attention(keras.layers.Layer):
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

class TFWav2Vec2FeedForward(keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, **kwargs) -> None: ...
    def call(self, hidden_states: tf.Tensor, training: bool = ...) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFWav2Vec2EncoderLayer(keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, **kwargs) -> None: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        training: bool = ...,
    ) -> tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFWav2Vec2EncoderLayerStableLayerNorm(keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, **kwargs) -> None: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        training: bool = ...,
    ) -> tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFWav2Vec2Encoder(keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, **kwargs) -> None: ...
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

class TFWav2Vec2EncoderStableLayerNorm(keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, **kwargs) -> None: ...
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
class TFWav2Vec2MainLayer(keras.layers.Layer):
    config_class = Wav2Vec2Config
    def __init__(self, config: Wav2Vec2Config, **kwargs) -> None: ...
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
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
        **kwargs: Any,
    ):  # -> TFWav2Vec2BaseModelOutput:
        ...

class TFWav2Vec2PreTrainedModel(TFPreTrainedModel):
    config_class = Wav2Vec2Config
    base_model_prefix = ...
    main_input_name = ...
    @property
    def input_signature(self):  # -> dict[str, Any]:
        ...
    @property
    def dummy_inputs(self):  # -> dict[str, Any]:
        ...
    def __init__(self, config, *inputs, **kwargs) -> None: ...

WAV2VEC2_START_DOCSTRING = ...
WAV2VEC2_INPUTS_DOCSTRING = ...

@add_start_docstrings(
    ...,
    WAV2VEC2_START_DOCSTRING,
)
class TFWav2Vec2Model(TFWav2Vec2PreTrainedModel):
    def __init__(self, config: Wav2Vec2Config, *inputs, **kwargs) -> None: ...
    @add_start_docstrings_to_model_forward(WAV2VEC2_INPUTS_DOCSTRING)
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
    WAV2VEC2_START_DOCSTRING,
)
class TFWav2Vec2ForCTC(TFWav2Vec2PreTrainedModel):
    def __init__(self, config: Wav2Vec2Config, *inputs, **kwargs) -> None: ...
    def freeze_feature_extractor(self):  # -> None:
        ...
    def freeze_feature_encoder(self):  # -> None:
        ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(WAV2VEC2_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=TFCausalLMOutput, config_class=_CONFIG_FOR_DOC
    )
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

class TFWav2Vec2ForSequenceClassification(TFWav2Vec2PreTrainedModel):
    def __init__(self, config) -> None: ...
    def freeze_feature_extractor(self):  # -> None:
        ...
    def freeze_feature_encoder(self):  # -> None:
        ...
    def freeze_base_model(self):  # -> None:
        ...
    @unpack_inputs
    def call(
        self,
        input_values: tf.Tensor,
        attention_mask: tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: tf.Tensor | None = ...,
        training: bool = ...,
    ) -> TFSequenceClassifierOutput | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

__all__ = [
    "TFWav2Vec2ForCTC",
    "TFWav2Vec2ForSequenceClassification",
    "TFWav2Vec2Model",
    "TFWav2Vec2PreTrainedModel",
]
