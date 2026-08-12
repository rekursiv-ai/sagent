from collections.abc import Callable
from typing import Any

from flax.core.frozen_dict import FrozenDict
from jax.random import PRNGKey

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from .configuration_longt5 import LongT5Config
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

"""Flax LongT5 model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
remat = ...

def shift_tokens_right(
    input_ids: jnp.ndarray, pad_token_id: int, decoder_start_token_id: int
) -> jnp.ndarray: ...

class FlaxLongT5LayerNorm(nn.Module):
    hidden_size: int
    dtype: jnp.dtype = ...
    eps: float = ...
    weight_init: Callable[..., np.ndarray] = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states): ...

class FlaxLongT5DenseActDense(nn.Module):
    config: LongT5Config
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states, deterministic=...): ...

class FlaxLongT5DenseGatedActDense(nn.Module):
    config: LongT5Config
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states, deterministic): ...

class FlaxLongT5LayerFF(nn.Module):
    config: LongT5Config
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states, deterministic=...): ...

class FlaxLongT5Attention(nn.Module):
    config: LongT5Config
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

class FlaxLongT5LocalAttention(nn.Module):
    config: LongT5Config
    has_relative_attention_bias: bool = ...
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def compute_bias(self, block_length: int): ...
    def __call__(
        self,
        hidden_states,
        attention_mask=...,
        key_value_states=...,
        position_bias=...,
        output_attentions=...,
        deterministic=...,
    ):  # -> tuple[Any, NDArray[Any] | ndarray[Any, Any] | Any, Any] | tuple[Any, NDArray[Any] | ndarray[Any, Any] | Any]:
        ...

class FlaxLongT5TransientGlobalAttention(nn.Module):
    config: LongT5Config
    has_relative_attention_bias: bool = ...
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def compute_bias(self, block_length: int): ...
    def compute_side_bias(
        self, attention_mask: np.ndarray, global_segment_ids: np.ndarray
    ) -> np.ndarray: ...
    def __call__(
        self,
        hidden_states,
        attention_mask=...,
        key_value_states=...,
        position_bias=...,
        output_attentions=...,
        deterministic=...,
    ):  # -> tuple[Any, Array | Any, Any] | tuple[Any, Array | Any]:
        ...

class FlaxLongT5LayerLocalSelfAttention(nn.Module):
    config: LongT5Config
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
        **kwargs: Any,
    ):  # -> tuple[Any, NDArray[Any] | ndarray[Any, Any] | Any, Any] | tuple[Any, NDArray[Any] | ndarray[Any, Any] | Any]:
        ...

class FlaxLongT5LayerTransientGlobalSelfAttention(nn.Module):
    config: LongT5Config
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
        **kwargs: Any,
    ):  # -> tuple[Any, Array | Any, Any] | tuple[Any, Array | Any]:
        ...

class FlaxLongT5LayerSelfAttention(nn.Module):
    config: LongT5Config
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

class FlaxLongT5LayerCrossAttention(nn.Module):
    config: LongT5Config
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

class FlaxLongT5Block(nn.Module):
    config: LongT5Config
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
    ):  # -> tuple[Any | tuple[Any, NDArray[Any] | ndarray[Any, Any] | Any, Any] | tuple[Any, NDArray[Any] | ndarray[Any, Any] | Any] | tuple[Any, Array | Any, Any] | tuple[Any, Array | Any], *tuple[NDArray[Any] | ndarray[Any, Any] | Any, ...]] | tuple[Any | tuple[Any, NDArray[Any] | ndarray[Any, Any] | Any, Any] | tuple[Any, NDArray[Any] | ndarray[Any, Any] | Any] | tuple[Any, Array | Any, Any] | tuple[Any, Array | Any], *tuple[Array | Any, ...]] | tuple[Any | tuple[Any, NDArray[Any] | ndarray[Any, Any] | Any, Any] | tuple[Any, NDArray[Any] | ndarray[Any, Any] | Any] | tuple[Any, Array | Any, Any] | tuple[Any, Array | Any], NDArray[Any] | ndarray[Any, Any] | Any, Any] | tuple[Any | tuple[Any, NDArray[Any] | ndarray[Any, Any] | Any, Any] | tuple[Any, NDArray[Any] | ndarray[Any, Any] | Any] | tuple[Any, Array | Any, Any] | tuple[Any, Array | Any], NDArray[Any] | ndarray[Any, Any] | Any] | tuple[Any | tuple[Any, NDArray[Any] | ndarray[Any, Any] | Any, Any] | tuple[Any, NDArray[Any] | ndarray[Any, Any] | Any] | tuple[Any, Array | Any, Any] | tuple[Any, Array | Any], Array | Any, Any] | tuple[Any | tuple[Any, NDArray[Any] | ndarray[Any, Any] | Any, Any] | tuple[Any, NDArray[Any] | ndarray[Any, Any] | Any] | tuple[Any, Array | Any, Any] | tuple[Any, Array | Any], Array | Any]:
        ...

class FlaxLongT5LayerCollection(nn.Module):
    config: LongT5Config
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
    ):  # -> tuple[Any | tuple[Any, NDArray[Any] | ndarray[Any, Any] | Any, Any] | tuple[Any, NDArray[Any] | ndarray[Any, Any] | Any] | tuple[Any, Array | Any, Any] | tuple[Any, Array | Any], *tuple[NDArray[Any] | ndarray[Any, Any] | Any, ...]] | tuple[Any | tuple[Any, NDArray[Any] | ndarray[Any, Any] | Any, Any] | tuple[Any, NDArray[Any] | ndarray[Any, Any] | Any] | tuple[Any, Array | Any, Any] | tuple[Any, Array | Any], *tuple[Array | Any, ...]] | tuple[Any | tuple[Any, NDArray[Any] | ndarray[Any, Any] | Any, Any] | tuple[Any, NDArray[Any] | ndarray[Any, Any] | Any] | tuple[Any, Array | Any, Any] | tuple[Any, Array | Any], NDArray[Any] | ndarray[Any, Any] | Any, Any] | tuple[Any | tuple[Any, NDArray[Any] | ndarray[Any, Any] | Any, Any] | tuple[Any, NDArray[Any] | ndarray[Any, Any] | Any] | tuple[Any, Array | Any, Any] | tuple[Any, Array | Any], NDArray[Any] | ndarray[Any, Any] | Any] | tuple[Any | tuple[Any, NDArray[Any] | ndarray[Any, Any] | Any, Any] | tuple[Any, NDArray[Any] | ndarray[Any, Any] | Any] | tuple[Any, Array | Any, Any] | tuple[Any, Array | Any], Array | Any, Any] | tuple[Any | tuple[Any, NDArray[Any] | ndarray[Any, Any] | Any, Any] | tuple[Any, NDArray[Any] | ndarray[Any, Any] | Any] | tuple[Any, Array | Any, Any] | tuple[Any, Array | Any], Array | Any]:
        ...

class FlaxLongT5BlockCollection(nn.Module):
    config: LongT5Config
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

class FlaxLongT5Stack(nn.Module):
    config: LongT5Config
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

LONGT5_ENCODE_INPUTS_DOCSTRING = ...
LONGT5_DECODE_INPUTS_DOCSTRING = ...
LONGT5_INPUTS_DOCSTRING = ...

class FlaxLongT5PreTrainedModel(FlaxPreTrainedModel):
    config_class = LongT5Config
    base_model_prefix = ...
    module_class: nn.Module = ...
    def __init__(
        self,
        config: LongT5Config,
        input_shape: tuple[int] = ...,
        seed: int = ...,
        dtype: jnp.dtype = ...,
        _do_init: bool = ...,
        **kwargs,
    ) -> None: ...
    def enable_gradient_checkpointing(self):  # -> None:
        ...
    def init_weights(
        self, rng: jax.random.PRNGKey, input_shape: tuple, params: FrozenDict = ...
    ) -> FrozenDict: ...
    @add_start_docstrings_to_model_forward(LONGT5_INPUTS_DOCSTRING)
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
    @add_start_docstrings(LONGT5_ENCODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=FlaxBaseModelOutput, config_class=LongT5Config
    )
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
    @add_start_docstrings(LONGT5_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=FlaxBaseModelOutputWithPastAndCrossAttentions,
        config_class=LongT5Config,
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

LONGT5_START_DOCSTRING = ...

@add_start_docstrings(
    ...,
    LONGT5_START_DOCSTRING,
)
class FlaxLongT5Module(nn.Module):
    config: LongT5Config
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

class FlaxLongT5Model(FlaxLongT5PreTrainedModel): ...

FLAX_LONGT5_MODEL_DOCSTRING = ...

@add_start_docstrings(..., LONGT5_START_DOCSTRING)
class FlaxLongT5ForConditionalGenerationModule(nn.Module):
    config: LongT5Config
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

class FlaxLongT5ForConditionalGeneration(FlaxLongT5PreTrainedModel):
    @add_start_docstrings(LONGT5_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=FlaxCausalLMOutputWithCrossAttentions, config_class=LongT5Config
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

FLAX_LONGT5_CONDITIONAL_GENERATION_DOCSTRING = ...
__all__ = [
    "FlaxLongT5ForConditionalGeneration",
    "FlaxLongT5Model",
    "FlaxLongT5PreTrainedModel",
]
