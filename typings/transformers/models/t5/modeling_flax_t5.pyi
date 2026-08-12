from collections.abc import Callable

from flax.core.frozen_dict import FrozenDict
from jax.random import PRNGKey

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from .configuration_t5 import T5Config
from ...modeling_flax_outputs import (
    FlaxBaseModelOutput,
    FlaxBaseModelOutputWithPastAndCrossAttentions,
    FlaxCausalLMOutputWithCrossAttentions,
)
from ...modeling_flax_utils import FlaxPreTrainedModel
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)

"""Flax T5 model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
remat = ...

def shift_tokens_right(
    input_ids: jnp.ndarray, pad_token_id: int, decoder_start_token_id: int
) -> jnp.ndarray: ...

class FlaxT5LayerNorm(nn.Module):
    hidden_size: int
    dtype: jnp.dtype = ...
    eps: float = ...
    weight_init: Callable[..., np.ndarray] = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states): ...

class FlaxT5DenseActDense(nn.Module):
    config: T5Config
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states, deterministic=...): ...

class FlaxT5DenseGatedActDense(nn.Module):
    config: T5Config
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states, deterministic): ...

class FlaxT5LayerFF(nn.Module):
    config: T5Config
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states, deterministic=...): ...

class FlaxT5Attention(nn.Module):
    config: T5Config
    has_relative_attention_bias: bool = ...
    causal: bool = ...
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def compute_bias(self, query_length, key_length): ...
    def __call__(
        self,
        hidden_states,
        attention_mask=...,
        key_value_states=...,
        position_bias=...,
        use_cache=...,
        output_attentions=...,
        deterministic=...,
        init_cache=...,
    ):  # -> tuple[Any, Array | Any, Any] | tuple[Any, Array | Any]:
        ...

class FlaxT5LayerSelfAttention(nn.Module):
    config: T5Config
    has_relative_attention_bias: bool = ...
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        hidden_states,
        attention_mask=...,
        position_bias=...,
        output_attentions=...,
        deterministic=...,
        init_cache=...,
    ):  # -> tuple[Any, Array | Any, Any] | tuple[Any, Array | Any]:
        ...

class FlaxT5LayerCrossAttention(nn.Module):
    config: T5Config
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        hidden_states,
        key_value_states,
        attention_mask=...,
        position_bias=...,
        output_attentions=...,
        deterministic=...,
    ):  # -> tuple[Any, Array | Any, Any] | tuple[Any, Array | Any]:
        ...

class FlaxT5Block(nn.Module):
    config: T5Config
    has_relative_attention_bias: bool = ...
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        hidden_states,
        attention_mask=...,
        position_bias=...,
        encoder_hidden_states=...,
        encoder_attention_mask=...,
        encoder_decoder_position_bias=...,
        output_attentions=...,
        return_dict=...,
        deterministic=...,
        init_cache=...,
    ):  # -> tuple[Any | tuple[Any, Array | Any, Any] | tuple[Any, Array | Any], *tuple[Array | Any, ...]] | tuple[Any | tuple[Any, Array | Any, Any] | tuple[Any, Array | Any], Array | Any, Any] | tuple[Any | tuple[Any, Array | Any, Any] | tuple[Any, Array | Any], Array | Any]:
        ...

class FlaxT5LayerCollection(nn.Module):
    config: T5Config
    has_relative_attention_bias: bool
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        hidden_states,
        attention_mask=...,
        position_bias=...,
        encoder_hidden_states=...,
        encoder_attention_mask=...,
        encoder_decoder_position_bias=...,
        output_attentions=...,
        deterministic=...,
        init_cache=...,
    ):  # -> tuple[Any | tuple[Any, Array | Any, Any] | tuple[Any, Array | Any], *tuple[Array | Any, ...]] | tuple[Any | tuple[Any, Array | Any, Any] | tuple[Any, Array | Any], Array | Any, Any] | tuple[Any | tuple[Any, Array | Any, Any] | tuple[Any, Array | Any], Array | Any]:
        ...

class FlaxT5BlockCollection(nn.Module):
    config: T5Config
    dtype: jnp.dtype = ...
    gradient_checkpointing: bool = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        hidden_states=...,
        attention_mask=...,
        encoder_hidden_states=...,
        encoder_attention_mask=...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        deterministic: bool = ...,
        init_cache: bool = ...,
    ):  # -> FlaxBaseModelOutputWithPastAndCrossAttentions:
        ...

class FlaxT5Stack(nn.Module):
    config: T5Config
    embed_tokens: nn.Embed
    dtype: jnp.dtype = ...
    gradient_checkpointing: bool = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_ids=...,
        attention_mask=...,
        encoder_hidden_states=...,
        encoder_attention_mask=...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
        deterministic: bool = ...,
        init_cache: bool = ...,
    ):  # -> FlaxBaseModelOutputWithPastAndCrossAttentions:
        ...

T5_ENCODE_INPUTS_DOCSTRING = ...
T5_DECODE_INPUTS_DOCSTRING = ...
T5_INPUTS_DOCSTRING = ...

class FlaxT5PreTrainedModel(FlaxPreTrainedModel):
    config_class = T5Config
    base_model_prefix = ...
    module_class: nn.Module = ...
    def __init__(
        self,
        config: T5Config,
        input_shape: tuple[int] = ...,
        seed: int = ...,
        dtype: jnp.dtype = ...,
        _do_init: bool = ...,
        gradient_checkpointing: bool = ...,
        **kwargs,
    ) -> None: ...
    def enable_gradient_checkpointing(self):  # -> None:
        ...
    def init_weights(
        self, rng: jax.random.PRNGKey, input_shape: tuple, params: FrozenDict = ...
    ) -> FrozenDict: ...
    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: jnp.ndarray | None = ...,
        decoder_input_ids: jnp.ndarray = ...,
        decoder_attention_mask: jnp.ndarray | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        train: bool = ...,
        params: dict | None = ...,
        dropout_rng: PRNGKey = ...,
    ): ...
    def init_cache(self, batch_size, max_length, encoder_outputs): ...
    @add_start_docstrings(T5_ENCODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxBaseModelOutput, config_class=T5Config)
    def encode(
        self,
        input_ids: jnp.ndarray,
        attention_mask: jnp.ndarray | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        train: bool = ...,
        params: dict | None = ...,
        dropout_rng: PRNGKey = ...,
    ): ...
    @add_start_docstrings(T5_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=FlaxBaseModelOutputWithPastAndCrossAttentions, config_class=T5Config
    )
    def decode(
        self,
        decoder_input_ids,
        encoder_outputs,
        encoder_attention_mask: jnp.ndarray | None = ...,
        decoder_attention_mask: jnp.ndarray | None = ...,
        past_key_values: dict | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        train: bool = ...,
        params: dict | None = ...,
        dropout_rng: PRNGKey = ...,
    ): ...

T5_START_DOCSTRING = ...

@add_start_docstrings(
    ...,
    T5_START_DOCSTRING,
)
class FlaxT5Module(nn.Module):
    config: T5Config
    dtype: jnp.dtype = ...
    gradient_checkpointing: bool = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_ids=...,
        attention_mask=...,
        decoder_input_ids=...,
        decoder_attention_mask=...,
        encoder_outputs=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
        deterministic: bool = ...,
    ):  # -> FlaxSeq2SeqModelOutput:
        ...

class FlaxT5Model(FlaxT5PreTrainedModel): ...

FLAX_T5_MODEL_DOCSTRING = ...

@add_start_docstrings(
    ...,
    T5_START_DOCSTRING,
)
class FlaxT5EncoderModule(nn.Module):
    config: T5Config
    dtype: jnp.dtype = ...
    gradient_checkpointing: bool = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_ids=...,
        attention_mask=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict: bool = ...,
        deterministic: bool = ...,
    ):  # -> FlaxBaseModelOutputWithPastAndCrossAttentions:
        ...

class FlaxT5EncoderModel(FlaxT5PreTrainedModel):
    @add_start_docstrings_to_model_forward(T5_ENCODE_INPUTS_DOCSTRING)
    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: jnp.ndarray | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        train: bool = ...,
        params: dict | None = ...,
        dropout_rng: PRNGKey = ...,
    ): ...

@add_start_docstrings(
    """T5 Model with a `language modeling` head on top.""", T5_START_DOCSTRING
)
class FlaxT5ForConditionalGenerationModule(nn.Module):
    config: T5Config
    dtype: jnp.dtype = ...
    gradient_checkpointing: bool = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_ids=...,
        attention_mask=...,
        decoder_input_ids=...,
        decoder_attention_mask=...,
        encoder_outputs=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
        deterministic: bool = ...,
    ):  # -> FlaxSeq2SeqLMOutput:
        ...

class FlaxT5ForConditionalGeneration(FlaxT5PreTrainedModel):
    @add_start_docstrings(T5_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=FlaxCausalLMOutputWithCrossAttentions, config_class=T5Config
    )
    def decode(
        self,
        decoder_input_ids,
        encoder_outputs,
        encoder_attention_mask: jnp.ndarray | None = ...,
        decoder_attention_mask: jnp.ndarray | None = ...,
        past_key_values: dict | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        train: bool = ...,
        params: dict | None = ...,
        dropout_rng: PRNGKey = ...,
    ):  # -> FlaxCausalLMOutputWithCrossAttentions:
        ...
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        max_length,
        attention_mask: jax.Array | None = ...,
        decoder_attention_mask: jax.Array | None = ...,
        encoder_outputs=...,
        **kwargs,
    ):  # -> dict[str, Any | Array | None]:
        ...
    def update_inputs_for_generation(self, model_outputs, model_kwargs): ...

FLAX_T5_CONDITIONAL_GENERATION_DOCSTRING = ...
__all__ = [
    "FlaxT5EncoderModel",
    "FlaxT5ForConditionalGeneration",
    "FlaxT5Model",
    "FlaxT5PreTrainedModel",
]
