from flax.core.frozen_dict import FrozenDict

import flax.linen as nn
import jax
import jax.numpy as jnp

from .configuration_dinov2 import Dinov2Config
from ...modeling_flax_utils import FlaxPreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward

"""Flax DINOv2 model."""
DINOV2_START_DOCSTRING = ...
DINOV2_INPUTS_DOCSTRING = ...

class FlaxDinov2PatchEmbeddings(nn.Module):
    config: Dinov2Config
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, pixel_values):  # -> Array:
        ...

class FlaxDinov2Embeddings(nn.Module):
    config: Dinov2Config
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def interpolate_pos_encoding(
        self, config, hidden_states, height, width, position_embeddings
    ):  # -> Array:
        ...
    def __call__(self, pixel_values, deterministic=...): ...

class FlaxDinov2SelfAttention(nn.Module):
    config: Dinov2Config
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self, hidden_states, deterministic: bool = ..., output_attentions: bool = ...
    ):  # -> tuple[Array, Any] | tuple[Array]:
        ...

class FlaxDinov2SelfOutput(nn.Module):
    config: Dinov2Config
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states, input_tensor, deterministic: bool = ...): ...

class FlaxDinov2Attention(nn.Module):
    config: Dinov2Config
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self, hidden_states, deterministic=..., output_attentions: bool = ...
    ):  # -> tuple[Any, Any | Array] | tuple[Any]:
        ...

def ones_with_scale(key, shape, scale, dtype=...): ...

class FlaxDinov2LayerScale(nn.Module):
    config: Dinov2Config
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states): ...

class FlaxDinov2DropPath(nn.Module):
    rate: float
    @nn.module.compact
    def __call__(self, inputs, deterministic: bool | None = ...): ...

class FlaxDinov2MLP(nn.Module):
    config: Dinov2Config
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states): ...

class FlaxDinov2SwiGLUFFN(nn.Module):
    config: Dinov2Config
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states): ...

class FlaxDinov2Layer(nn.Module):
    config: Dinov2Config
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self, hidden_states, deterministic: bool = ..., output_attentions: bool = ...
    ):  # -> tuple[Any, Any | Array] | tuple[Any]:
        ...

class FlaxDinov2LayerCollection(nn.Module):
    config: Dinov2Config
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

class FlaxDinov2Encoder(nn.Module):
    config: Dinov2Config
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

class FlaxDinov2PreTrainedModel(FlaxPreTrainedModel):
    config_class = Dinov2Config
    base_model_prefix = ...
    main_input_name = ...
    module_class: nn.Module = ...
    def __init__(
        self,
        config: Dinov2Config,
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
        DINOV2_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    def __call__(
        self,
        pixel_values,
        params: dict | None = ...,
        dropout_rng: jax.random.PRNGKey = ...,
        train: bool = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ): ...

class FlaxDinov2Module(nn.Module):
    config: Dinov2Config
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        pixel_values,
        deterministic: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any, Any, *tuple[Any, ...]] | FlaxBaseModelOutputWithPooling:
        ...

@add_start_docstrings(
    ...,
    DINOV2_START_DOCSTRING,
)
class FlaxDinov2Model(FlaxDinov2PreTrainedModel): ...

FLAX_VISION_MODEL_DOCSTRING = ...

class FlaxDinov2ForImageClassificationModule(nn.Module):
    config: Dinov2Config
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        pixel_values=...,
        deterministic: bool = ...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
    ):  # -> tuple[Any, *tuple[Any, ...]] | FlaxSequenceClassifierOutput:
        ...

@add_start_docstrings(
    ...,
    DINOV2_START_DOCSTRING,
)
class FlaxDinov2ForImageClassification(FlaxDinov2PreTrainedModel): ...

FLAX_VISION_CLASSIFICATION_DOCSTRING = ...
__all__ = [
    "FlaxDinov2ForImageClassification",
    "FlaxDinov2Model",
    "FlaxDinov2PreTrainedModel",
]
