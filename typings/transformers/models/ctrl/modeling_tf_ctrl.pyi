import numpy as np
import tensorflow as tf

from .configuration_ctrl import CTRLConfig
from ...modeling_tf_outputs import (
    TFBaseModelOutputWithPast,
    TFCausalLMOutputWithPast,
    TFSequenceClassifierOutput,
)
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
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

"""TF 2.0 CTRL model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...

def angle_defn(pos, i, d_model_size): ...
def positional_encoding(position, d_model_size): ...
def scaled_dot_product_attention(
    q, k, v, mask, attention_mask=..., head_mask=...
):  # -> tuple[Any, Any]:
    ...

class TFMultiHeadAttention(keras.layers.Layer):
    def __init__(
        self, d_model_size, num_heads, output_attentions=..., **kwargs
    ) -> None: ...
    def split_into_heads(self, x, batch_size): ...
    def call(
        self,
        v,
        k,
        q,
        mask,
        layer_past,
        attention_mask,
        head_mask,
        use_cache,
        output_attentions,
        training=...,
    ):  # -> tuple[Any, Any | tuple[None], Any] | tuple[Any, Any | tuple[None]]:
        ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFPointWiseFeedForwardLayer(keras.layers.Layer):
    def __init__(self, d_model_size, dff, **kwargs) -> None: ...
    def call(self, inputs, trainable=...): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFEncoderLayer(keras.layers.Layer):
    def __init__(
        self,
        d_model_size,
        num_heads,
        dff,
        rate=...,
        layer_norm_epsilon=...,
        output_attentions=...,
        **kwargs,
    ) -> None: ...
    def call(
        self,
        x,
        mask,
        layer_past,
        attention_mask,
        head_mask,
        use_cache,
        output_attentions,
        training=...,
    ): ...
    def build(self, input_shape=...):  # -> None:
        ...

@keras_serializable
class TFCTRLMainLayer(keras.layers.Layer):
    config_class = CTRLConfig
    def __init__(self, config, **kwargs) -> None: ...
    def get_input_embeddings(self): ...
    def set_input_embeddings(self, new_embeddings):  # -> None:
        ...
    @unpack_inputs
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        past_key_values: tuple[tuple[np.ndarray | tf.Tensor]] | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool | None = ...,
    ) -> tuple | TFBaseModelOutputWithPast: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFCTRLPreTrainedModel(TFPreTrainedModel):
    config_class = CTRLConfig
    base_model_prefix = ...

CTRL_START_DOCSTRING = ...
CTRL_INPUTS_DOCSTRING = ...

@add_start_docstrings(
    ...,
    CTRL_START_DOCSTRING,
)
class TFCTRLModel(TFCTRLPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(CTRL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        past_key_values: tuple[tuple[np.ndarray | tf.Tensor]] | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool | None = ...,
    ) -> tuple | TFBaseModelOutputWithPast: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFCTRLBiasLayer(keras.layers.Layer):
    def __init__(self, shape, initializer, trainable, name, **kwargs) -> None: ...
    def build(self, input_shape):  # -> None:
        ...
    def call(self, x): ...

@add_start_docstrings(
    ...,
    CTRL_START_DOCSTRING,
)
class TFCTRLLMHeadModel(TFCTRLPreTrainedModel, TFCausalLanguageModelingLoss):
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    def get_output_embeddings(self): ...
    def set_output_embeddings(self, value):  # -> None:
        ...
    def get_bias(self):  # -> dict[str, Any]:
        ...
    def set_bias(self, value):  # -> None:
        ...
    def prepare_inputs_for_generation(
        self, inputs, past_key_values=..., use_cache=..., **kwargs
    ):  # -> dict[str, Any | None]:
        ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(CTRL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFCausalLMOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        past_key_values: tuple[tuple[np.ndarray | tf.Tensor]] | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: np.ndarray | tf.Tensor | None = ...,
        training: bool | None = ...,
    ) -> tuple | TFCausalLMOutputWithPast: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    CTRL_START_DOCSTRING,
)
class TFCTRLForSequenceClassification(
    TFCTRLPreTrainedModel, TFSequenceClassificationLoss
):
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    def get_output_embeddings(self): ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(CTRL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        past_key_values: tuple[tuple[np.ndarray | tf.Tensor]] | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: np.ndarray | tf.Tensor | None = ...,
        training: bool | None = ...,
    ) -> tuple | TFSequenceClassifierOutput: ...
    def build(self, input_shape=...):  # -> None:
        ...

__all__ = [
    "TFCTRLForSequenceClassification",
    "TFCTRLLMHeadModel",
    "TFCTRLModel",
    "TFCTRLPreTrainedModel",
]
