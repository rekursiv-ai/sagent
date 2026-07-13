from typing import Any

import numpy as np
import tensorflow as tf

from .configuration_xglm import XGLMConfig
from ...file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_tf_outputs import (
    TFBaseModelOutputWithPastAndCrossAttentions,
    TFCausalLMOutputWithCrossAttentions,
)
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFModelInputType,
    TFPreTrainedModel,
    TFSharedEmbeddings,
    keras,
    keras_serializable,
    unpack_inputs,
)

"""TF 2.0 XGLM model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
LARGE_NEGATIVE = ...

def create_sinusoidal_positions(
    num_positions: int, embedding_dim: int, padding_idx: int | None
) -> tf.Tensor: ...

class TFXGLMAttention(keras.layers.Layer):
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

class TFXGLMDecoderLayer(keras.layers.Layer):
    def __init__(self, config: XGLMConfig, **kwargs: Any) -> None: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor | None = ...,
        encoder_hidden_states: tf.Tensor | None = ...,
        encoder_attention_mask: tf.Tensor | None = ...,
        layer_head_mask: tf.Tensor | None = ...,
        cross_attn_layer_head_mask: tf.Tensor | None = ...,
        past_key_value: tuple[tf.Tensor] | None = ...,
        training: bool | None = ...,
    ) -> tuple[tf.Tensor, tf.Tensor, tuple[tuple[tf.Tensor]]]: ...
    def build(self, input_shape=...):  # -> None:
        ...

@keras_serializable
class TFXGLMMainLayer(keras.layers.Layer):
    config_class = XGLMConfig
    def __init__(
        self,
        config: XGLMConfig,
        embed_tokens: TFSharedEmbeddings | None = ...,
        *inputs,
        **kwargs: Any,
    ) -> None: ...
    def get_input_embeddings(self) -> TFSharedEmbeddings: ...
    def set_input_embeddings(self, value: TFSharedEmbeddings) -> None: ...
    def embed_positions(
        self, position_ids: np.ndarray | tf.Tensor | None = ...
    ) -> tf.Tensor: ...
    @unpack_inputs
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        encoder_hidden_states: np.ndarray | tf.Tensor | None = ...,
        encoder_attention_mask: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        cross_attn_head_mask: np.ndarray | tf.Tensor | None = ...,
        past_key_values: tuple[tuple[np.ndarray | tf.Tensor]] | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool | None = ...,
        **kwargs: Any,
    ) -> TFBaseModelOutputWithPastAndCrossAttentions | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFXGLMPreTrainedModel(TFPreTrainedModel):
    config_class = XGLMConfig
    base_model_prefix = ...

XGLM_START_DOCSTRING = ...
XGLM_INPUTS_DOCSTRING = ...

@add_start_docstrings(
    ...,
    XGLM_START_DOCSTRING,
)
class TFXGLMModel(TFXGLMPreTrainedModel):
    def __init__(
        self,
        config: XGLMConfig,
        embed_tokens: TFSharedEmbeddings | None = ...,
        *inputs: Any,
        **kwargs: Any,
    ) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(XGLM_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        encoder_hidden_states: np.ndarray | tf.Tensor | None = ...,
        encoder_attention_mask: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        cross_attn_head_mask: np.ndarray | tf.Tensor | None = ...,
        past_key_values: tuple[tuple[np.ndarray | tf.Tensor]] | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool | None = ...,
        **kwargs: Any,
    ) -> TFBaseModelOutputWithPastAndCrossAttentions | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    ...,
    XGLM_START_DOCSTRING,
)
class TFXGLMForCausalLM(TFXGLMPreTrainedModel, TFCausalLanguageModelingLoss):
    base_model_prefix = ...
    _keys_to_ignore_on_load_missing = ...
    _keys_to_ignore_on_save = ...
    def __init__(
        self,
        config: XGLMConfig,
        embed_tokens: TFSharedEmbeddings | None = ...,
        *inputs: Any,
        **kwargs: Any,
    ) -> None: ...
    def prepare_inputs_for_generation(
        self, inputs, past_key_values=..., use_cache=..., **kwargs
    ):  # -> dict[str, Any | None]:
        ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(XGLM_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=TFCausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFCausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        encoder_hidden_states: np.ndarray | tf.Tensor | None = ...,
        encoder_attention_mask: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        cross_attn_head_mask: np.ndarray | tf.Tensor | None = ...,
        past_key_values: tuple[tuple[np.ndarray | tf.Tensor]] | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        labels: np.ndarray | tf.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool | None = ...,
        **kwargs: Any,
    ) -> TFCausalLMOutputWithCrossAttentions | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...
    def tf_to_pt_weight_rename(
        self, tf_weight
    ):  # -> tuple[Literal['lm_head.weight'], Literal['model.embed_tokens.weight']] | tuple[Any]:
        ...

__all__ = ["TFXGLMForCausalLM", "TFXGLMModel", "TFXGLMPreTrainedModel"]
