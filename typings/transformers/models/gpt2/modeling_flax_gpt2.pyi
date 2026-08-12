from typing import Any

from flax.core.frozen_dict import FrozenDict

import flax.linen as nn
import jax
import jax.numpy as jnp

from .configuration_gpt2 import GPT2Config
from ...modeling_flax_utils import FlaxPreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward

logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
GPT2_START_DOCSTRING = ...
GPT2_INPUTS_DOCSTRING = ...

class FlaxConv1D(nn.Module):
    features: int
    use_bias: bool = ...
    dtype: Any = ...
    precision: Any = ...
    @nn.compact
    def __call__(self, inputs):  # -> Array:
        ...

class FlaxGPT2Attention(nn.Module):
    config: GPT2Config
    dtype: jnp.dtype = ...
    causal: bool = ...
    is_cross_attention: bool = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        hidden_states,
        key_value_states: jnp.ndarray | None = ...,
        attention_mask=...,
        deterministic: bool = ...,
        init_cache: bool = ...,
        output_attentions: bool = ...,
    ):  # -> tuple[Any, Any] | tuple[Any]:
        ...

class FlaxGPT2MLP(nn.Module):
    config: GPT2Config
    intermediate_size: int
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states, deterministic: bool = ...): ...

class FlaxGPT2Block(nn.Module):
    config: GPT2Config
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        hidden_states,
        attention_mask=...,
        encoder_hidden_states: jnp.ndarray | None = ...,
        encoder_attention_mask: jnp.ndarray | None = ...,
        deterministic: bool = ...,
        init_cache: bool = ...,
        output_attentions: bool = ...,
    ):  # -> tuple[Any, *tuple[Any, ...]] | tuple[Any] | tuple[Any, Any]:
        ...

class FlaxGPT2PreTrainedModel(FlaxPreTrainedModel):
    config_class = GPT2Config
    base_model_prefix = ...
    module_class: nn.Module = ...
    def __init__(
        self,
        config: GPT2Config,
        input_shape: tuple = ...,
        seed: int = ...,
        dtype: jnp.dtype = ...,
        _do_init: bool = ...,
        **kwargs,
    ) -> None: ...
    def init_weights(
        self, rng: jax.random.PRNGKey, input_shape: tuple, params: FrozenDict = ...
    ) -> FrozenDict: ...
    def init_cache(self, batch_size, max_length): ...
    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    def __call__(
        self,
        input_ids,
        attention_mask=...,
        position_ids=...,
        encoder_hidden_states: jnp.ndarray | None = ...,
        encoder_attention_mask: jnp.ndarray | None = ...,
        params: dict | None = ...,
        past_key_values: dict | None = ...,
        dropout_rng: jax.random.PRNGKey = ...,
        train: bool = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ): ...

class FlaxGPT2BlockCollection(nn.Module):
    config: GPT2Config
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        hidden_states,
        attention_mask=...,
        encoder_hidden_states: jnp.ndarray | None = ...,
        encoder_attention_mask: jnp.ndarray | None = ...,
        deterministic: bool = ...,
        init_cache: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any, tuple[()] | tuple[Any, ...] | None, tuple[()] | tuple[Any, ...] | None, tuple[()] | tuple[Any, ...] | None]:
        ...

class FlaxGPT2Module(nn.Module):
    config: GPT2Config
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        encoder_hidden_states: jnp.ndarray | None = ...,
        encoder_attention_mask: jnp.ndarray | None = ...,
        deterministic=...,
        init_cache: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any | tuple[Any, ...] | tuple[()], ...] | FlaxBaseModelOutputWithPastAndCrossAttentions:
        ...

@add_start_docstrings(
    ...,
    GPT2_START_DOCSTRING,
)
class FlaxGPT2Model(FlaxGPT2PreTrainedModel): ...

class FlaxGPT2LMHeadModule(nn.Module):
    config: GPT2Config
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        encoder_hidden_states: jnp.ndarray | None = ...,
        encoder_attention_mask: jnp.ndarray | None = ...,
        deterministic: bool = ...,
        init_cache: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any, *tuple[Any | tuple[Any, ...] | tuple[()], ...]] | FlaxCausalLMOutputWithCrossAttentions:
        ...

@add_start_docstrings(
    ...,
    GPT2_START_DOCSTRING,
)
class FlaxGPT2LMHeadModel(FlaxGPT2PreTrainedModel):
    def prepare_inputs_for_generation(
        self, input_ids, max_length, attention_mask: jax.Array | None = ...
    ):  # -> dict[str, Any | Array]:
        ...
    def update_inputs_for_generation(self, model_outputs, model_kwargs): ...

__all__ = ["FlaxGPT2LMHeadModel", "FlaxGPT2Model", "FlaxGPT2PreTrainedModel"]
