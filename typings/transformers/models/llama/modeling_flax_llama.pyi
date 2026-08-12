from flax.core.frozen_dict import FrozenDict

import flax.linen as nn
import jax
import jax.numpy as jnp

from .configuration_llama import LlamaConfig
from ...modeling_flax_utils import FlaxPreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward

"""Flax LLaMA model."""
logger = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
_REAL_CHECKPOINT_FOR_DOC = ...
LLAMA_START_DOCSTRING = ...
LLAMA_INPUTS_DOCSTRING = ...

def create_sinusoidal_positions(num_pos, dim):  # -> Array:
    ...
def rotate_half(tensor):  # -> Array:
    ...
def apply_rotary_pos_emb(tensor, sin_pos, cos_pos): ...

class FlaxLlamaRMSNorm(nn.Module):
    config: LlamaConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states): ...

class FlaxLlamaRotaryEmbedding(nn.Module):
    config: LlamaConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, key, query, position_ids):  # -> tuple[Array, Array]:
        ...

class FlaxLlamaAttention(nn.Module):
    config: LlamaConfig
    dtype: jnp.dtype = ...
    causal: bool = ...
    is_cross_attention: bool = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        deterministic: bool = ...,
        init_cache: bool = ...,
        output_attentions: bool = ...,
    ):  # -> tuple[Any, Any] | tuple[Any]:
        ...

class FlaxLlamaMLP(nn.Module):
    config: LlamaConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states): ...

class FlaxLlamaDecoderLayer(nn.Module):
    config: LlamaConfig
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
    ):  # -> tuple[Any, Any] | tuple[Any]:
        ...

class FlaxLlamaPreTrainedModel(FlaxPreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = ...
    module_class: nn.Module = ...
    def __init__(
        self,
        config: LlamaConfig,
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
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
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

class FlaxLlamaLayerCollection(nn.Module):
    config: LlamaConfig
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
    ):  # -> tuple[Any, tuple[()] | tuple[Any, ...] | None, tuple[()] | tuple[Any, ...] | None]:
        ...

class FlaxLlamaModule(nn.Module):
    config: LlamaConfig
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
    ):  # -> tuple[Any | tuple[Any, ...] | tuple[()], ...] | FlaxBaseModelOutput:
        ...

@add_start_docstrings(
    ...,
    LLAMA_START_DOCSTRING,
)
class FlaxLlamaModel(FlaxLlamaPreTrainedModel): ...

class FlaxLlamaForCausalLMModule(nn.Module):
    config: LlamaConfig
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
    ):  # -> tuple[Any, *tuple[Any | tuple[Any, ...] | tuple[()], ...]] | FlaxCausalLMOutput:
        ...

@add_start_docstrings(
    ...,
    LLAMA_START_DOCSTRING,
)
class FlaxLlamaForCausalLM(FlaxLlamaPreTrainedModel):
    def prepare_inputs_for_generation(
        self, input_ids, max_length, attention_mask: jax.Array | None = ...
    ):  # -> dict[str, Any | Array]:
        ...
    def update_inputs_for_generation(self, model_outputs, model_kwargs): ...

__all__ = ["FlaxLlamaForCausalLM", "FlaxLlamaModel", "FlaxLlamaPreTrainedModel"]
