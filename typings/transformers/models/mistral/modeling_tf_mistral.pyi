import tensorflow as tf

from .configuration_mistral import MistralConfig
from ...modeling_tf_outputs import (
    TFBaseModelOutputWithPast,
    TFCausalLMOutputWithPast,
    TFSequenceClassifierOutputWithPast,
)
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward

"""TF 2.0  Mistral model."""
logger = ...
_CONFIG_FOR_DOC = ...

class TFMistralRMSNorm(keras.layers.Layer):
    def __init__(self, hidden_size, eps=..., **kwargs) -> None: ...
    def build(self, input_shape=...):  # -> None:
        ...
    def call(self, hidden_states): ...

class TFMistralRotaryEmbedding(keras.layers.Layer):
    def __init__(
        self, dim, max_position_embeddings=..., base=..., **kwargs
    ) -> None: ...
    def call(self, x, seq_len=...):  # -> tuple[Any, Any]:
        ...

def rotate_half(x): ...
def apply_rotary_pos_emb(
    q, k, cos, sin, position_ids, unsqueeze_dim=...
):  # -> tuple[Any, Any]:
    ...

class TFMistralMLP(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None: ...
    def call(self, x): ...
    def build(self, input_shape=...):  # -> None:
        ...

def repeat_kv(hidden_states: tf.Tensor, n_rep: int) -> tf.Tensor: ...

class TFMistralAttention(keras.layers.Layer):
    def __init__(
        self, config: MistralConfig, layer_idx: int | None = ..., **kwargs
    ) -> None: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor | None = ...,
        position_ids: tf.Tensor | None = ...,
        past_key_value: tuple[tf.Tensor] | None = ...,
        output_attentions: bool | None = ...,
        use_cache: bool | None = ...,
        training=...,
        **kwargs,
    ) -> tuple[tf.Tensor, tf.Tensor | None, tuple[tf.Tensor] | None]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFMistralDecoderLayer(keras.layers.Layer):
    def __init__(self, config: MistralConfig, layer_idx: int, **kwargs) -> None: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor | None = ...,
        position_ids: tf.Tensor | None = ...,
        past_key_value: tuple[tf.Tensor] | None = ...,
        output_attentions: bool | None = ...,
        use_cache: bool | None = ...,
        **kwargs,
    ) -> tuple[tf.Tensor, tuple[tf.Tensor, tf.Tensor] | None]: ...
    def build(self, input_shape=...):  # -> None:
        ...

@keras_serializable
class TFMistralMainLayer(keras.layers.Layer):
    config_class = MistralConfig
    def __init__(self, config: MistralConfig, **kwargs) -> None: ...
    @unpack_inputs
    def call(
        self,
        input_ids: tf.Tensor | None = ...,
        attention_mask: tf.Tensor | None = ...,
        position_ids: tf.Tensor | None = ...,
        past_key_values: list[tf.Tensor] | None = ...,
        inputs_embeds: tf.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | TFBaseModelOutputWithPast: ...
    def build(self, input_shape=...):  # -> None:
        ...

MISTRAL_START_DOCSTRING = ...

@add_start_docstrings(
    ...,
    MISTRAL_START_DOCSTRING,
)
class TFMistralPreTrainedModel(TFPreTrainedModel):
    config_class = MistralConfig
    base_model_prefix = ...

MISTRAL_INPUTS_DOCSTRING = ...

@add_start_docstrings(
    ...,
    MISTRAL_START_DOCSTRING,
)
class TFMistralModel(TFMistralPreTrainedModel):
    def __init__(self, config: MistralConfig, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(MISTRAL_INPUTS_DOCSTRING)
    def call(
        self,
        input_ids: tf.Tensor | None = ...,
        attention_mask: tf.Tensor | None = ...,
        position_ids: tf.Tensor | None = ...,
        past_key_values: list[tf.Tensor] | None = ...,
        inputs_embeds: tf.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | TFBaseModelOutputWithPast: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFMistralForCausalLM(TFMistralPreTrainedModel, TFCausalLanguageModelingLoss):
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        MISTRAL_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    def call(
        self,
        input_ids: tf.Tensor | None = ...,
        attention_mask: tf.Tensor | None = ...,
        position_ids: tf.Tensor | None = ...,
        past_key_values: list[tf.Tensor] | None = ...,
        inputs_embeds: tf.Tensor | None = ...,
        labels: tf.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | TFCausalLMOutputWithPast: ...
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=...,
        attention_mask=...,
        inputs_embeds=...,
        **kwargs,
    ):  # -> dict[str, Any | None]:
        ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    MISTRAL_START_DOCSTRING,
)
class TFMistralForSequenceClassification(
    TFMistralPreTrainedModel, TFSequenceClassificationLoss
):
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        MISTRAL_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    def call(
        self,
        input_ids: tf.Tensor | None = ...,
        attention_mask: tf.Tensor | None = ...,
        position_ids: tf.Tensor | None = ...,
        past_key_values: list[tf.Tensor] | None = ...,
        inputs_embeds: tf.Tensor | None = ...,
        labels: tf.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | TFSequenceClassifierOutputWithPast: ...
    def build(self, input_shape=...):  # -> None:
        ...

__all__ = [
    "TFMistralForCausalLM",
    "TFMistralForSequenceClassification",
    "TFMistralModel",
    "TFMistralPreTrainedModel",
]
