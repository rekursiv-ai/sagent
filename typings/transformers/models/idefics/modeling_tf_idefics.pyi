from dataclasses import dataclass

import tensorflow as tf

from .configuration_idefics import IdeficsConfig
from ... import TFPreTrainedModel
from ...modeling_tf_outputs import ModelOutput
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFModelInputType,
    keras_serializable,
    unpack_inputs,
)
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)

"""TF 2.0 Idefics model."""
logger = ...
_CONFIG_FOR_DOC = ...

@dataclass
class TFIdeficsBaseModelOutputWithPast(ModelOutput):
    last_hidden_state: tf.Tensor | None = ...
    past_key_values: tuple[tuple[tf.Tensor]] | None = ...
    hidden_states: tuple[tf.Tensor] | None = ...
    attentions: tuple[tf.Tensor] | None = ...
    image_hidden_states: tuple[tf.Tensor] | None = ...

@dataclass
class TFIdeficsCausalLMOutputWithPast(ModelOutput):
    loss: tf.Tensor | None = ...
    logits: tf.Tensor | None = ...
    past_key_values: list[tf.Tensor] | None = ...
    hidden_states: tuple[tf.Tensor] | None = ...
    attentions: tuple[tf.Tensor] | None = ...
    image_hidden_states: tuple[tf.Tensor] | None = ...

def expand_inputs_for_generation(
    input_ids,
    expand_size=...,
    is_encoder_decoder=...,
    attention_mask=...,
    encoder_outputs=...,
    **model_kwargs,
):  # -> tuple[Any, dict[str, Any]]:
    ...
def update_model_kwargs_for_generation(outputs, model_kwargs): ...
def prepare_inputs_for_generation(
    input_ids, past_key_values=..., **kwargs
):  # -> dict[str, Any | None]:
    ...
def freeze_model(model, module_exceptions=...): ...

class TFIdeficsDecoupledEmbedding(tf.keras.layers.Embedding):
    def __init__(
        self,
        num_embeddings,
        num_additional_embeddings,
        embedding_dim,
        partially_freeze: bool | None = ...,
        dtype=...,
        **kwargs,
    ) -> None: ...
    def call(self, input_ids): ...
    def extra_repr(self) -> str: ...

class TFIdeficsDecoupledLinear(tf.keras.layers.Layer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        out_additional_features: int = ...,
        bias: bool = ...,
        partially_freeze: bool = ...,
        **kwargs,
    ) -> None: ...
    def call(self, inputs: tf.Tensor) -> tf.Tensor: ...
    def get_config(self): ...
    def extra_repr(self) -> str: ...
    @classmethod
    def from_config(cls, config):  # -> Self:
        ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFIdeficsRMSNorm(tf.keras.layers.Layer):
    def __init__(self, hidden_size, eps=..., **kwargs) -> None: ...
    def build(self, input_shape):  # -> None:
        ...
    def call(self, hidden_states): ...

class TFIdeficsEmbedding(tf.keras.layers.Layer):
    def __init__(
        self, dim, max_position_embeddings=..., base=..., **kwargs
    ) -> None: ...
    def call(self, x, seq_len=...):  # -> tuple[Any, Any]:
        ...

def rotate_half(x): ...
def apply_rotary_pos_emb(q, k, cos, sin, position_ids):  # -> tuple[Any, Any]:
    ...

class TFIdeficsMLP(tf.keras.layers.Layer):
    def __init__(
        self, hidden_size: int, intermediate_size: int, hidden_act: str, **kwargs
    ) -> None: ...
    def call(self, x): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFIdeficsAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = ...,
        is_cross_attention: bool = ...,
        config: IdeficsConfig = ...,
        qk_layer_norms: bool = ...,
        **kwargs,
    ) -> None: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        key_value_states: tf.Tensor | None = ...,
        attention_mask: tf.Tensor | None = ...,
        position_ids: tf.Tensor | None = ...,
        past_key_value: tuple[tf.Tensor] | None = ...,
        output_attentions: bool = ...,
        use_cache: bool = ...,
    ) -> tuple[tf.Tensor, tf.Tensor | None, tuple[tf.Tensor] | None]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFIdeficsDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, config: IdeficsConfig, **kwargs) -> None: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor | None = ...,
        position_ids: tf.Tensor | None = ...,
        past_key_value: tuple[tf.Tensor] | None = ...,
        output_attentions: bool | None = ...,
        use_cache: bool | None = ...,
        training=...,
    ) -> tuple[tf.Tensor, tuple[tf.Tensor, tf.Tensor] | None]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFIdeficsGatedCrossAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, config: IdeficsConfig, **kwargs) -> None: ...
    def build(self, input_shape):  # -> None:
        ...
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor | None = ...,
        image_hidden_states: tf.Tensor | None = ...,
        image_attention_mask: tf.Tensor | None = ...,
        cross_attention_gate: tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        use_cache: bool | None = ...,
        past_key_value: tuple[tf.Tensor] | None = ...,
    ) -> tuple[tf.Tensor, tuple[tf.Tensor, tf.Tensor] | None]: ...

LLAMA_START_DOCSTRING = ...

@add_start_docstrings(
    ...,
    LLAMA_START_DOCSTRING,
)
class TFIdeficsPreTrainedModel(TFPreTrainedModel):
    config_class = IdeficsConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...

LLAMA_INPUTS_DOCSTRING = ...

@add_start_docstrings(
    ...,
    LLAMA_START_DOCSTRING,
)
@keras_serializable
class TFIdeficsMainLayer(tf.keras.layers.Layer):
    config_class = IdeficsConfig
    def __init__(
        self, config: IdeficsConfig, add_pooling_year: bool = ..., **kwargs
    ) -> None: ...
    def freeze_relevant_params(self, config=...):  # -> None:
        ...
    def freeze_text_layers(self, module_exceptions=...):  # -> None:
        ...
    def freeze_vision_layers(self, module_exceptions=...):  # -> None:
        ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: tf.Tensor | None = ...,
        position_ids: tf.Tensor | None = ...,
        past_key_values: list[tf.Tensor] | None = ...,
        inputs_embeds: tf.Tensor | None = ...,
        pixel_values: tf.Tensor | None = ...,
        image_encoder_embeddings: tf.Tensor | None = ...,
        perceiver_embeddings: tf.Tensor | None = ...,
        image_attention_mask: tf.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        interpolate_pos_encoding: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool | None = ...,
    ) -> TFIdeficsBaseModelOutputWithPast | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFIdeficsModel(TFIdeficsPreTrainedModel):
    def __init__(self, config: IdeficsConfig, *inputs, **kwargs) -> None: ...
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: tf.Tensor | None = ...,
        position_ids: tf.Tensor | None = ...,
        past_key_values: list[tf.Tensor] | None = ...,
        inputs_embeds: tf.Tensor | None = ...,
        pixel_values: tf.Tensor | None = ...,
        image_encoder_embeddings: tf.Tensor | None = ...,
        perceiver_embeddings: tf.Tensor | None = ...,
        image_attention_mask: tf.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        interpolate_pos_encoding: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool | None = ...,
    ) -> TFIdeficsBaseModelOutputWithPast | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFIdeficsForVisionText2Text(TFPreTrainedModel, TFCausalLanguageModelingLoss):
    _keys_to_ignore_on_load_missing = ...
    _tied_weights_keys = ...
    config_class = IdeficsConfig
    def __init__(self, config, vision_model=..., **kwargs) -> None: ...
    def tie_weights(self):  # -> None:
        ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=TFIdeficsCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: tf.Tensor | None = ...,
        position_ids: tf.Tensor | None = ...,
        past_key_values: list[tf.Tensor] | None = ...,
        inputs_embeds: tf.Tensor | None = ...,
        pixel_values: tf.Tensor | None = ...,
        image_encoder_embeddings: tf.Tensor | None = ...,
        perceiver_embeddings: tf.Tensor | None = ...,
        image_attention_mask: tf.Tensor | None = ...,
        labels: tf.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        interpolate_pos_encoding: bool | None = ...,
        return_dict: bool | None = ...,
        training=...,
    ) -> TFIdeficsCausalLMOutputWithPast | tuple[tf.Tensor]: ...
    def prepare_inputs_for_generation(
        self, input_ids, past=..., **kwargs
    ):  # -> dict[str, Any | None]:
        ...
    def build(self, input_shape=...):  # -> None:
        ...

__all__ = ["TFIdeficsForVisionText2Text", "TFIdeficsModel", "TFIdeficsPreTrainedModel"]
