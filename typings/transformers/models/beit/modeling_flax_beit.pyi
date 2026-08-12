from collections.abc import Callable

from flax.core.frozen_dict import FrozenDict

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

from .configuration_beit import BeitConfig
from ...modeling_flax_outputs import FlaxBaseModelOutputWithPooling
from ...modeling_flax_utils import FlaxPreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward

@flax.struct.dataclass
class FlaxBeitModelOutputWithPooling(FlaxBaseModelOutputWithPooling): ...

BEIT_START_DOCSTRING = ...
BEIT_INPUTS_DOCSTRING = ...

def relative_position_index_init(window_size: tuple[int, int]) -> jnp.ndarray: ...
def ones_with_scale(key, shape, scale, dtype=...): ...

class FlaxBeitDropPath(nn.Module):
    rate: float
    @nn.module.compact
    def __call__(self, inputs, deterministic: bool | None = ...): ...

class FlaxBeitPatchEmbeddings(nn.Module):
    config: BeitConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, pixel_values):  # -> Array:
        ...

class FlaxBeitEmbeddings(nn.Module):
    config: BeitConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, pixel_values, bool_masked_pos=..., deterministic=...): ...

class FlaxBeitRelativePositionBias(nn.Module):
    config: BeitConfig
    window_size: tuple[int, int]
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self):  # -> Array:
        ...

class FlaxBeitSelfAttention(nn.Module):
    config: BeitConfig
    window_size: tuple[int, int]
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        hidden_states,
        relative_position_bias=...,
        deterministic: bool = ...,
        output_attentions: bool = ...,
    ):  # -> tuple[Array, Any] | tuple[Array]:
        ...

class FlaxBeitSelfOutput(nn.Module):
    config: BeitConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states, deterministic: bool = ...): ...

class FlaxBeitAttention(nn.Module):
    config: BeitConfig
    window_size: tuple[int, int]
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        hidden_states,
        relative_position_bias=...,
        deterministic=...,
        output_attentions: bool = ...,
    ):  # -> tuple[Any, Any | Array] | tuple[Any]:
        ...

class FlaxBeitIntermediate(nn.Module):
    config: BeitConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states): ...

class FlaxBeitOutput(nn.Module):
    config: BeitConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states, deterministic: bool = ...): ...

class FlaxBeitLayer(nn.Module):
    config: BeitConfig
    window_size: tuple[int, int]
    drop_path_rate: float
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        hidden_states,
        relative_position_bias=...,
        deterministic: bool = ...,
        output_attentions: bool = ...,
    ):  # -> tuple[Any, Any | Array] | tuple[Any]:
        ...

class FlaxBeitLayerCollection(nn.Module):
    config: BeitConfig
    window_size: tuple[int, int]
    drop_path_rates: list[float]
    relative_position_bias: Callable[[], jnp.ndarray]
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        hidden_states,
        deterministic: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any, ...] | FlaxBaseModelOutput:
        ...

class FlaxBeitEncoder(nn.Module):
    config: BeitConfig
    window_size: tuple[int, int]
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        hidden_states,
        deterministic: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any, ...] | FlaxBaseModelOutput:
        ...

class FlaxBeitPreTrainedModel(FlaxPreTrainedModel):
    config_class = BeitConfig
    base_model_prefix = ...
    main_input_name = ...
    module_class: nn.Module = ...
    def __init__(
        self,
        config: BeitConfig,
        input_shape=...,
        seed: int = ...,
        dtype: jnp.dtype = ...,
        _do_init: bool = ...,
        **kwargs,
    ) -> None: ...
    def init_weights(
        self, rng: jax.random.PRNGKey, input_shape: tuple, params: FrozenDict = ...
    ) -> FrozenDict: ...
    @add_start_docstrings_to_model_forward(
        BEIT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    def __call__(
        self,
        pixel_values,
        bool_masked_pos=...,
        params: dict | None = ...,
        dropout_rng: jax.random.PRNGKey = ...,
        train: bool = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ): ...

class FlaxBeitPooler(nn.Module):
    config: BeitConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states): ...

class FlaxBeitModule(nn.Module):
    config: BeitConfig
    dtype: jnp.dtype = ...
    add_pooling_layer: bool = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        pixel_values,
        bool_masked_pos=...,
        deterministic: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any, *tuple[Any, ...]] | tuple[Any, Any, *tuple[Any, ...]] | FlaxBeitModelOutputWithPooling:
        ...

@add_start_docstrings(
    ...,
    BEIT_START_DOCSTRING,
)
class FlaxBeitModel(FlaxBeitPreTrainedModel): ...

FLAX_BEIT_MODEL_DOCSTRING = ...

class FlaxBeitForMaskedImageModelingModule(nn.Module):
    config: BeitConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        pixel_values=...,
        bool_masked_pos=...,
        deterministic: bool = ...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
    ):  # -> tuple[Any, *tuple[Any, ...]] | FlaxMaskedLMOutput:
        ...

@add_start_docstrings(
    ...,
    BEIT_START_DOCSTRING,
)
class FlaxBeitForMaskedImageModeling(FlaxBeitPreTrainedModel): ...

FLAX_BEIT_MLM_DOCSTRING = ...

class FlaxBeitForImageClassificationModule(nn.Module):
    config: BeitConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        pixel_values=...,
        bool_masked_pos=...,
        deterministic: bool = ...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
    ):  # -> tuple[Any, *tuple[Any, ...]] | FlaxSequenceClassifierOutput:
        ...

@add_start_docstrings(
    ...,
    BEIT_START_DOCSTRING,
)
class FlaxBeitForImageClassification(FlaxBeitPreTrainedModel): ...

FLAX_BEIT_CLASSIF_DOCSTRING = ...
__all__ = [
    "FlaxBeitForImageClassification",
    "FlaxBeitForMaskedImageModeling",
    "FlaxBeitModel",
    "FlaxBeitPreTrainedModel",
]
