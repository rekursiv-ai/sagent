import numpy as np
import tensorflow as tf

from .configuration_opt import OPTConfig
from ...modeling_tf_outputs import TFBaseModelOutputWithPast, TFCausalLMOutputWithPast
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFModelInputType,
    TFPreTrainedModel,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)

"""TF 2.0 OPT model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
_EXPECTED_OUTPUT_SHAPE = ...
_CAUSAL_LM_EXPECTED_OUTPUT = ...
LARGE_NEGATIVE = ...

class TFOPTLearnedPositionalEmbedding(keras.layers.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, **kwargs) -> None: ...
    def call(self, attention_mask, past_key_values_length: int = ...): ...

class TFOPTAttention(keras.layers.Layer):
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

class TFOPTDecoderLayer(keras.layers.Layer):
    def __init__(self, config: OPTConfig, **kwargs) -> None: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        layer_head_mask: tf.Tensor | None = ...,
        past_key_value: tuple[tuple[np.ndarray | tf.Tensor]] | None = ...,
        training: bool | None = ...,
        output_attentions: bool | None = ...,
        use_cache: bool | None = ...,
    ) -> tuple[tf.Tensor, tf.Tensor, tuple[tuple[tf.Tensor]]]: ...
    def build(self, input_shape=...):  # -> None:
        ...

OPT_START_DOCSTRING = ...

@add_start_docstrings(
    ...,
    OPT_START_DOCSTRING,
)
class TFOPTPreTrainedModel(TFPreTrainedModel):
    config_class = OPTConfig
    base_model_prefix = ...

OPT_INPUTS_DOCSTRING = ...

@keras_serializable
class TFOPTDecoder(keras.layers.Layer):
    config_class = OPTConfig
    def __init__(self, config: OPTConfig, **kwargs) -> None: ...
    def get_embed_tokens(self):  # -> TFSharedEmbeddings:
        ...
    def set_embed_tokens(self, embed_tokens):  # -> None:
        ...
    def set_input_embeddings(self, new_embeddings):  # -> None:
        ...
    @unpack_inputs
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        past_key_values: tuple[tuple[np.ndarray | tf.Tensor]] | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool | None = ...,
    ) -> TFBaseModelOutputWithPast | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

@keras_serializable
class TFOPTMainLayer(keras.layers.Layer):
    config_class = OPTConfig
    def __init__(self, config: OPTConfig, **kwargs) -> None: ...
    def get_input_embeddings(self):  # -> TFSharedEmbeddings:
        ...
    def set_input_embeddings(self, new_embeddings):  # -> None:
        ...
    @unpack_inputs
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        past_key_values: tuple[tuple[np.ndarray | tf.Tensor]] | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool | None = ...,
        **kwargs,
    ) -> TFBaseModelOutputWithPast | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    OPT_START_DOCSTRING,
)
@keras_serializable
class TFOPTModel(TFOPTPreTrainedModel):
    config_class = OPTConfig
    def __init__(self, config: OPTConfig, **kwargs) -> None: ...
    def get_input_embeddings(self):  # -> TFSharedEmbeddings:
        ...
    def set_input_embeddings(self, new_embeddings):  # -> None:
        ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(OPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        past_key_values: tuple[tuple[np.ndarray | tf.Tensor]] | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool | None = ...,
        **kwargs,
    ) -> TFBaseModelOutputWithPast | tuple[tf.Tensor]: ...
    def serving_output(self, output):  # -> TFBaseModelOutputWithPast:
        ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    OPT_START_DOCSTRING,
)
@keras_serializable
class TFOPTForCausalLM(TFOPTPreTrainedModel, TFCausalLanguageModelingLoss):
    config_class = OPTConfig
    def __init__(self, config: OPTConfig, **kwargs) -> None: ...
    def get_output_embeddings(self):  # -> TFSharedEmbeddings:
        ...
    def prepare_inputs_for_generation(
        self, inputs, past_key_values=..., use_cache=..., **kwargs
    ):  # -> dict[str, Any | None]:
        ...
    @unpack_inputs
    @replace_return_docstrings(
        output_type=TFCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFCausalLMOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_CAUSAL_LM_EXPECTED_OUTPUT,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        past_key_values: tuple[tuple[np.ndarray | tf.Tensor]] | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        labels: np.ndarray | tf.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool | None = ...,
        **kwargs,
    ) -> TFCausalLMOutputWithPast | tuple[tf.Tensor]: ...
    def serving_output(self, output):  # -> TFCausalLMOutputWithPast:
        ...
    def build(self, input_shape=...):  # -> None:
        ...

__all__ = ["TFOPTForCausalLM", "TFOPTModel", "TFOPTPreTrainedModel"]
