from flax.core.frozen_dict import FrozenDict

import flax.linen as nn
import jax
import jax.numpy as jnp

from .configuration_bloom import BloomConfig
from ...modeling_flax_utils import FlaxPreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward

"""Flax BLOOM model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
BLOOM_START_DOCSTRING = ...
BLOOM_INPUTS_DOCSTRING = ...

def build_alibi_tensor(
    attention_mask: jnp.ndarray, num_heads: int, dtype: jnp.dtype | None = ...
):  # -> Array:
    ...

class FlaxBloomAttention(nn.Module):
    config: BloomConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        hidden_states,
        residual,
        alibi,
        attention_mask=...,
        deterministic: bool = ...,
        init_cache: bool = ...,
        output_attentions: bool = ...,
    ):  # -> tuple[Any, Any] | tuple[Any]:
        ...

class BloomGELU(nn.Module):
    def setup(self):  # -> None:
        ...
    def __call__(self, x): ...

class FlaxBloomMLP(nn.Module):
    config: BloomConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states, residual, deterministic: bool = ...): ...

class FlaxBloomBlock(nn.Module):
    config: BloomConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        hidden_states,
        alibi,
        attention_mask=...,
        deterministic: bool = ...,
        init_cache: bool = ...,
        output_attentions: bool = ...,
    ):  # -> tuple[Any, Any] | tuple[Any]:
        ...

class FlaxBloomPreTrainedModel(FlaxPreTrainedModel):
    config_class = BloomConfig
    base_model_prefix = ...
    module_class: nn.Module = ...
    def __init__(
        self,
        config: BloomConfig,
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
    @add_start_docstrings_to_model_forward(BLOOM_INPUTS_DOCSTRING)
    def __call__(
        self,
        input_ids,
        attention_mask=...,
        past_key_values: dict | None = ...,
        params: dict | None = ...,
        dropout_rng: jax.random.PRNGKey = ...,
        train: bool = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ): ...

class FlaxBloomBlockCollection(nn.Module):
    config: BloomConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        hidden_states,
        alibi,
        attention_mask=...,
        deterministic: bool = ...,
        init_cache: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
    ):  # -> tuple[Any, tuple[()] | tuple[Any, ...] | None, tuple[()] | tuple[Any, ...] | None]:
        ...

class FlaxBloomModule(nn.Module):
    config: BloomConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_ids=...,
        attention_mask=...,
        deterministic=...,
        init_cache: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any | tuple[()] | tuple[Any, ...], ...] | FlaxBaseModelOutputWithPastAndCrossAttentions:
        ...

@add_start_docstrings(
    ...,
    BLOOM_START_DOCSTRING,
)
class FlaxBloomModel(FlaxBloomPreTrainedModel): ...

class FlaxBloomForCausalLMModule(nn.Module):
    config: BloomConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_ids,
        attention_mask,
        deterministic: bool = ...,
        init_cache: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any, *tuple[Any | tuple[()] | tuple[Any, ...], ...]] | FlaxCausalLMOutput:
        ...

@add_start_docstrings(
    ...,
    BLOOM_START_DOCSTRING,
)
class FlaxBloomForCausalLM(FlaxBloomPreTrainedModel):
    def prepare_inputs_for_generation(
        self, input_ids, max_length, attention_mask: jax.Array | None = ...
    ):  # -> dict[str, Any | Array]:
        ...
    def update_inputs_for_generation(self, model_outputs, model_kwargs): ...

__all__ = ["FlaxBloomForCausalLM", "FlaxBloomModel", "FlaxBloomPreTrainedModel"]
