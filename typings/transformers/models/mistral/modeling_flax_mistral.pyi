from flax.core.frozen_dict import FrozenDict

import flax.linen as nn
import jax
import jax.numpy as jnp

from .configuration_mistral import MistralConfig
from ...modeling_flax_utils import FlaxPreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward

"""Flax Mistral model."""
logger = ...
_CONFIG_FOR_DOC = ...
_REAL_CHECKPOINT_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
MISTRAL_START_DOCSTRING = ...
MISTRAL_INPUTS_DOCSTRING = ...

class FlaxMistralRMSNorm(nn.Module):
    config: MistralConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states): ...

class FlaxMistralRotaryEmbedding(nn.Module):
    config: MistralConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, key, query, position_ids):  # -> tuple[Array, Array]:
        ...

class FlaxMistralMLP(nn.Module):
    config: MistralConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states): ...

def apply_rotary_pos_emb(tensor, sin_pos, cos_pos): ...
def create_sinusoidal_positions(num_pos, dim):  # -> Array:
    ...
def rotate_half(tensor):  # -> Array:
    ...

class FlaxMistralAttention(nn.Module):
    config: MistralConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray | None = ...,
        position_ids: jnp.ndarray | None = ...,
        deterministic: bool = ...,
        output_attentions: bool = ...,
        init_cache: bool = ...,
    ) -> tuple[jnp.ndarray, jnp.ndarray]: ...

class FlaxMistralDecoderLayer(nn.Module):
    config: MistralConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        hidden_states,
        attention_mask=...,
        position_ids=...,
        deterministic: bool = ...,
        init_cache: bool = ...,
        output_attentions: bool = ...,
    ):  # -> tuple[Any, ndarray]:
        ...

class FlaxMistralPreTrainedModel(FlaxPreTrainedModel):
    config_class = MistralConfig
    base_model_prefix = ...
    module_class: nn.Module = ...
    def __init__(
        self,
        config: MistralConfig,
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
    @add_start_docstrings_to_model_forward(MISTRAL_INPUTS_DOCSTRING)
    def __call__(
        self,
        input_ids,
        attention_mask=...,
        position_ids=...,
        params: dict | None = ...,
        past_key_values: dict | None = ...,
        dropout_rng: jax.random.PRNGKey = ...,
        train: bool = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ): ...

class FlaxMistralLayerCollection(nn.Module):
    config: MistralConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        hidden_states,
        attention_mask=...,
        position_ids=...,
        deterministic: bool = ...,
        init_cache: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any, tuple[()] | tuple[Any, ...] | None, tuple[()] | tuple[ndarray, ...] | None]:
        ...

class FlaxMistralModule(nn.Module):
    config: MistralConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_ids,
        attention_mask=...,
        position_ids=...,
        deterministic=...,
        init_cache: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any | tuple[Any, ...] | tuple[()] | tuple[ndarray, ...], ...] | FlaxBaseModelOutput:
        ...

@add_start_docstrings(
    ...,
    MISTRAL_START_DOCSTRING,
)
class FlaxMistralModel(FlaxMistralPreTrainedModel):
    module_class = ...

class FlaxMistralForCausalLMModule(nn.Module):
    config: MistralConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_ids,
        attention_mask=...,
        position_ids=...,
        deterministic: bool = ...,
        init_cache: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any, *tuple[Any | tuple[Any, ...] | tuple[()] | tuple[ndarray, ...], ...]] | FlaxCausalLMOutput:
        ...

@add_start_docstrings(
    ...,
    MISTRAL_START_DOCSTRING,
)
class FlaxMistralForCausalLM(FlaxMistralPreTrainedModel):
    module_class = ...
    def prepare_inputs_for_generation(
        self, input_ids, max_length, attention_mask: jax.Array | None = ...
    ):  # -> dict[str, Any | Array]:
        ...
    def update_inputs_for_generation(self, model_outputs, model_kwargs): ...

__all__ = ["FlaxMistralForCausalLM", "FlaxMistralModel", "FlaxMistralPreTrainedModel"]
