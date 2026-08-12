from flax.core.frozen_dict import FrozenDict

import flax.linen as nn
import jax
import jax.numpy as jnp

from .configuration_vit import ViTConfig
from ...modeling_flax_utils import FlaxPreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward

VIT_START_DOCSTRING = ...
VIT_INPUTS_DOCSTRING = ...

class FlaxViTPatchEmbeddings(nn.Module):
    config: ViTConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, pixel_values):  # -> Array:
        ...

class FlaxViTEmbeddings(nn.Module):
    config: ViTConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, pixel_values, deterministic=...): ...

class FlaxViTSelfAttention(nn.Module):
    config: ViTConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self, hidden_states, deterministic: bool = ..., output_attentions: bool = ...
    ):  # -> tuple[Array, Any] | tuple[Array]:
        ...

class FlaxViTSelfOutput(nn.Module):
    config: ViTConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states, input_tensor, deterministic: bool = ...): ...

class FlaxViTAttention(nn.Module):
    config: ViTConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self, hidden_states, deterministic=..., output_attentions: bool = ...
    ):  # -> tuple[Any, Any | Array] | tuple[Any]:
        ...

class FlaxViTIntermediate(nn.Module):
    config: ViTConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states): ...

class FlaxViTOutput(nn.Module):
    config: ViTConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states, attention_output, deterministic: bool = ...): ...

class FlaxViTLayer(nn.Module):
    config: ViTConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self, hidden_states, deterministic: bool = ..., output_attentions: bool = ...
    ):  # -> tuple[Any, Any | Array] | tuple[Any]:
        ...

class FlaxViTLayerCollection(nn.Module):
    config: ViTConfig
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

class FlaxViTEncoder(nn.Module):
    config: ViTConfig
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

class FlaxViTPooler(nn.Module):
    config: ViTConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states): ...

class FlaxViTPreTrainedModel(FlaxPreTrainedModel):
    config_class = ViTConfig
    base_model_prefix = ...
    main_input_name = ...
    module_class: nn.Module = ...
    def __init__(
        self,
        config: ViTConfig,
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
        VIT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
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

class FlaxViTModule(nn.Module):
    config: ViTConfig
    dtype: jnp.dtype = ...
    add_pooling_layer: bool = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        pixel_values,
        deterministic: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any, *tuple[Any, ...]] | tuple[Any, Any, *tuple[Any, ...]] | FlaxBaseModelOutputWithPooling:
        ...

@add_start_docstrings(
    ...,
    VIT_START_DOCSTRING,
)
class FlaxViTModel(FlaxViTPreTrainedModel): ...

FLAX_VISION_MODEL_DOCSTRING = ...

class FlaxViTForImageClassificationModule(nn.Module):
    config: ViTConfig
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
    VIT_START_DOCSTRING,
)
class FlaxViTForImageClassification(FlaxViTPreTrainedModel): ...

FLAX_VISION_CLASSIF_DOCSTRING = ...
__all__ = ["FlaxViTForImageClassification", "FlaxViTModel", "FlaxViTPreTrainedModel"]
