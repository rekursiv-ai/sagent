from flax.core.frozen_dict import FrozenDict
from jax.random import PRNGKey

import flax.linen as nn
import jax
import jax.numpy as jnp

from .configuration_opt import OPTConfig
from ...modeling_flax_utils import FlaxPreTrainedModel
from ...utils import add_start_docstrings

"""Flax OPT model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
OPT_START_DOCSTRING = ...
OPT_INPUTS_DOCSTRING = ...

class FlaxOPTAttention(nn.Module):
    config: OPTConfig
    embed_dim: int
    num_heads: int
    dropout: float = ...
    causal: bool = ...
    bias: bool = ...
    dtype: jnp.dtype = ...
    def setup(self) -> None: ...
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        key_value_states: jnp.ndarray | None = ...,
        attention_mask: jnp.ndarray | None = ...,
        init_cache: bool = ...,
        deterministic: bool = ...,
    ) -> tuple[jnp.ndarray]: ...

class FlaxOPTDecoderLayer(nn.Module):
    config: OPTConfig
    dtype: jnp.dtype = ...
    def setup(self) -> None: ...
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray,
        init_cache: bool = ...,
        output_attentions: bool = ...,
        deterministic: bool = ...,
    ) -> tuple[jnp.ndarray]: ...

class FlaxOPTDecoderLayerCollection(nn.Module):
    config: OPTConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        hidden_states,
        attention_mask,
        deterministic: bool = ...,
        init_cache: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
    ):  # -> list[Any | ndarray | tuple[()] | tuple[Any | ndarray, ...] | tuple[ndarray, ...] | None]:
        ...

class FlaxOPTLearnedPositionalEmbedding(nn.Embed):
    def setup(self):  # -> None:
        ...
    def __call__(self, positions): ...

class FlaxOPTDecoder(nn.Module):
    config: OPTConfig
    dtype: jnp.dtype = ...
    offset: int = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        init_cache: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
        deterministic: bool = ...,
    ):  # -> tuple[Any | ndarray | tuple[()] | tuple[Any | ndarray, ...] | tuple[ndarray, ...] | tuple[Any | ndarray | tuple[()] | tuple[Any | ndarray, ...] | tuple[ndarray, ...] | None, ...], ...] | FlaxBaseModelOutput:
        ...

class FlaxOPTPreTrainedModel(FlaxPreTrainedModel):
    config_class = OPTConfig
    base_model_prefix: str = ...
    module_class: nn.Module = ...
    def __init__(
        self,
        config: OPTConfig,
        input_shape: tuple[int] = ...,
        seed: int = ...,
        dtype: jnp.dtype = ...,
        _do_init: bool = ...,
        **kwargs,
    ) -> None: ...
    def init_weights(
        self, rng: jax.random.PRNGKey, input_shape: tuple, params: FrozenDict = ...
    ) -> FrozenDict: ...
    def init_cache(self, batch_size, max_length): ...
    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: jnp.ndarray | None = ...,
        position_ids: jnp.ndarray | None = ...,
        params: dict | None = ...,
        past_key_values: dict | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        dropout_rng: PRNGKey = ...,
        deterministic: bool = ...,
    ): ...

class FlaxOPTModule(nn.Module):
    config: OPTConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
        deterministic: bool = ...,
        init_cache=...,
    ):  # -> tuple[Any | ndarray | tuple[()] | tuple[Any | ndarray, ...] | tuple[ndarray, ...] | tuple[Any | ndarray | tuple[()] | tuple[Any | ndarray, ...] | tuple[ndarray, ...] | None, ...], ...] | FlaxBaseModelOutput:
        ...

class FlaxOPTModel(FlaxOPTPreTrainedModel):
    config: OPTConfig
    dtype: jnp.dtype = ...
    module_class = ...

@add_start_docstrings(
    ...,
    OPT_START_DOCSTRING,
)
class FlaxOPTForCausalLMModule(nn.Module):
    config: OPTConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        init_cache: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
        deterministic: bool = ...,
    ):  # -> tuple[Any, *tuple[Any | ndarray | tuple[()] | tuple[Any | ndarray, ...] | tuple[ndarray, ...] | tuple[Any | ndarray | tuple[()] | tuple[Any | ndarray, ...] | tuple[ndarray, ...] | None, ...], ...]] | FlaxMaskedLMOutput:
        ...

@add_start_docstrings(
    ...,
    OPT_START_DOCSTRING,
)
class FlaxOPTForCausalLM(FlaxOPTPreTrainedModel):
    module_class = ...
    def prepare_inputs_for_generation(
        self, input_ids, max_length, attention_mask: jax.Array | None = ...
    ):  # -> dict[str, Any | Array]:
        ...
    def update_inputs_for_generation(self, model_outputs, model_kwargs): ...

__all__ = ["FlaxOPTForCausalLM", "FlaxOPTModel", "FlaxOPTPreTrainedModel"]
