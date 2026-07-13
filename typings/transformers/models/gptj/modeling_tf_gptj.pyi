import numpy as np
import tensorflow as tf

from .configuration_gptj import GPTJConfig
from ...file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from ...modeling_tf_outputs import (
    TFBaseModelOutputWithPast,
    TFCausalLMOutputWithPast,
    TFQuestionAnsweringModelOutput,
    TFSequenceClassifierOutputWithPast,
)
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFModelInputType,
    TFPreTrainedModel,
    TFQuestionAnsweringLoss,
    TFSequenceClassificationLoss,
    keras,
    keras_serializable,
    unpack_inputs,
)

"""TF 2.0 GPT-J model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...

def create_sinusoidal_positions(num_pos: int, dim: int) -> tf.Tensor: ...
def rotate_every_two(x: tf.Tensor) -> tf.Tensor: ...
def apply_rotary_pos_emb(tensor: tf.Tensor, sincos: tf.Tensor) -> tf.Tensor: ...

class TFGPTJAttention(keras.layers.Layer):
    def __init__(self, config: GPTJConfig, **kwargs) -> None: ...
    def get_causal_mask(self, key_length, query_length) -> tf.Tensor: ...
    @staticmethod
    def get_masked_bias(dtype: tf.DType) -> tf.Tensor: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        layer_past: tuple[tf.Tensor, tf.Tensor] | None = ...,
        attention_mask: tf.Tensor | None = ...,
        position_ids: tf.Tensor | None = ...,
        head_mask: tf.Tensor | None = ...,
        use_cache: bool = ...,
        output_attentions: bool = ...,
    ):  # -> tuple[Any, tuple[Any, Any] | None, Any] | tuple[Any, tuple[Any, Any] | None]:
        ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFGPTJMLP(keras.layers.Layer):
    def __init__(
        self, intermediate_size: int, config: GPTJConfig, **kwargs
    ) -> None: ...
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFGPTJBlock(keras.layers.Layer):
    def __init__(self, config: GPTJConfig, **kwargs) -> None: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        layer_past: tf.Tensor | None = ...,
        attention_mask: tf.Tensor | None = ...,
        position_ids: tf.Tensor | None = ...,
        head_mask: tf.Tensor | None = ...,
        use_cache: bool = ...,
        output_attentions: bool = ...,
    ): ...
    def build(self, input_shape=...):  # -> None:
        ...

@keras_serializable
class TFGPTJMainLayer(keras.layers.Layer):
    config_class = GPTJConfig
    def __init__(self, config: GPTJConfig, *inputs, **kwargs) -> None: ...
    def get_input_embeddings(self):  # -> TFSharedEmbeddings:
        ...
    def set_input_embeddings(self, value: tf.Tensor):  # -> None:
        ...
    @unpack_inputs
    def call(
        self,
        input_ids=...,
        past_key_values=...,
        attention_mask=...,
        token_type_ids=...,
        position_ids=...,
        head_mask=...,
        inputs_embeds=...,
        use_cache=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
        training=...,
    ) -> TFBaseModelOutputWithPast | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFGPTJPreTrainedModel(TFPreTrainedModel):
    config_class = GPTJConfig
    base_model_prefix = ...
    _keys_to_ignore_on_load_unexpected = ...

GPTJ_START_DOCSTRING = ...
GPTJ_INPUTS_DOCSTRING = ...

@add_start_docstrings(
    ...,
    GPTJ_START_DOCSTRING,
)
class TFGPTJModel(TFGPTJPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(GPTJ_INPUTS_DOCSTRING)
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
    ) -> TFBaseModelOutputWithPast | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    GPTJ_START_DOCSTRING,
)
class TFGPTJForCausalLM(TFGPTJPreTrainedModel, TFCausalLanguageModelingLoss):
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    def prepare_inputs_for_generation(
        self, inputs, past_key_values=..., use_cache=..., **kwargs
    ):  # -> dict[str, Any | None]:
        ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        GPTJ_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
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
        labels: np.ndarray | tf.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool | None = ...,
    ) -> TFCausalLMOutputWithPast | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    GPTJ_START_DOCSTRING,
)
class TFGPTJForSequenceClassification(
    TFGPTJPreTrainedModel, TFSequenceClassificationLoss
):
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        GPTJ_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFSequenceClassifierOutputWithPast,
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
        labels: np.ndarray | tf.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool | None = ...,
    ) -> TFSequenceClassifierOutputWithPast | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    GPTJ_START_DOCSTRING,
)
class TFGPTJForQuestionAnswering(TFGPTJPreTrainedModel, TFQuestionAnsweringLoss):
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        GPTJ_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFQuestionAnsweringModelOutput,
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
        start_positions: np.ndarray | tf.Tensor | None = ...,
        end_positions: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool | None = ...,
    ) -> TFQuestionAnsweringModelOutput | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

__all__ = [
    "TFGPTJForCausalLM",
    "TFGPTJForQuestionAnswering",
    "TFGPTJForSequenceClassification",
    "TFGPTJModel",
    "TFGPTJPreTrainedModel",
]
