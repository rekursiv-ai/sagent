from flax.core.frozen_dict import FrozenDict
from jax.random import PRNGKey

import flax.linen as nn
import jax
import jax.numpy as jnp

from .configuration_xglm import XGLMConfig
from ...modeling_flax_utils import FlaxPreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward

"""Flax XGLM model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
XGLM_START_DOCSTRING = ...
XGLM_INPUTS_DOCSTRING = ...

def create_sinusoidal_positions(n_pos, dim, padding_idx=...):  # -> Array:
    ...

class FlaxXGLMAttention(nn.Module):
    config: XGLMConfig
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

class FlaxXGLMDecoderLayer(nn.Module):
    config: XGLMConfig
    dtype: jnp.dtype = ...
    def setup(self) -> None: ...
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray,
        encoder_hidden_states: jnp.ndarray | None = ...,
        encoder_attention_mask: jnp.ndarray | None = ...,
        init_cache: bool = ...,
        output_attentions: bool = ...,
        deterministic: bool = ...,
    ) -> tuple[jnp.ndarray]: ...

class FlaxXGLMDecoderLayerCollection(nn.Module):
    config: XGLMConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        hidden_states,
        attention_mask,
        encoder_hidden_states: jnp.ndarray | None = ...,
        encoder_attention_mask: jnp.ndarray | None = ...,
        deterministic: bool = ...,
        init_cache: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any | ndarray | tuple[Any | ndarray | None, ...] | tuple[()] | tuple[ndarray | None, ...], ...] | FlaxBaseModelOutputWithPastAndCrossAttentions:
        ...

class FlaxXGLMModule(nn.Module):
    config: XGLMConfig
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
        init_cache: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
        deterministic: bool = ...,
    ):  # -> tuple[Any | tuple[Any | ndarray | None, ...] | tuple[Any, ...] | tuple[ndarray | None, ...] | ndarray | tuple[()], ...] | FlaxBaseModelOutputWithPastAndCrossAttentions:
        ...

class FlaxXGLMPreTrainedModel(FlaxPreTrainedModel):
    config_class = XGLMConfig
    base_model_prefix: str = ...
    module_class: nn.Module = ...
    def __init__(
        self,
        config: XGLMConfig,
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
    @add_start_docstrings_to_model_forward(XGLM_INPUTS_DOCSTRING)
    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: jnp.ndarray | None = ...,
        position_ids: jnp.ndarray | None = ...,
        encoder_hidden_states: jnp.ndarray | None = ...,
        encoder_attention_mask: jnp.ndarray | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        train: bool = ...,
        params: dict | None = ...,
        past_key_values: dict | None = ...,
        dropout_rng: PRNGKey = ...,
    ): ...

@add_start_docstrings(
    ...,
    XGLM_START_DOCSTRING,
)
class FlaxXGLMModel(FlaxXGLMPreTrainedModel):
    module_class = ...

class FlaxXGLMForCausalLMModule(nn.Module):
    config: XGLMConfig
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
        init_cache: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
        deterministic: bool = ...,
    ):  # -> tuple[Any, *tuple[Any | tuple[Any | ndarray | None, ...] | tuple[Any, ...] | tuple[ndarray | None, ...] | ndarray | tuple[()], ...]] | FlaxCausalLMOutputWithCrossAttentions:
        ...

@add_start_docstrings(
    ...,
    XGLM_START_DOCSTRING,
)
class FlaxXGLMForCausalLM(FlaxXGLMPreTrainedModel):
    module_class = ...
    def prepare_inputs_for_generation(
        self, input_ids, max_length, attention_mask: jax.Array | None = ...
    ):  # -> dict[str, Any | Array]:
        ...
    def update_inputs_for_generation(self, model_outputs, model_kwargs): ...

__all__ = ["FlaxXGLMForCausalLM", "FlaxXGLMModel", "FlaxXGLMPreTrainedModel"]
