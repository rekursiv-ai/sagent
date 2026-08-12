from typing import Any

from flax.core.frozen_dict import FrozenDict

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

from .configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from ...modeling_flax_outputs import FlaxBaseModelOutputWithPooling
from ...modeling_flax_utils import FlaxPreTrainedModel
from ...utils import ModelOutput, add_start_docstrings

logger = ...
CLIP_START_DOCSTRING = ...
CLIP_TEXT_INPUTS_DOCSTRING = ...
CLIP_VISION_INPUTS_DOCSTRING = ...
CLIP_INPUTS_DOCSTRING = ...

@flax.struct.dataclass
class FlaxCLIPTextModelOutput(ModelOutput):
    text_embeds: jnp.ndarray = ...
    last_hidden_state: jnp.ndarray = ...
    hidden_states: tuple[jnp.ndarray, ...] | None = ...
    attentions: tuple[jnp.ndarray, ...] | None = ...

@flax.struct.dataclass
class FlaxCLIPOutput(ModelOutput):
    logits_per_image: jnp.ndarray = ...
    logits_per_text: jnp.ndarray = ...
    text_embeds: jnp.ndarray = ...
    image_embeds: jnp.ndarray = ...
    text_model_output: FlaxBaseModelOutputWithPooling = ...
    vision_model_output: FlaxBaseModelOutputWithPooling = ...
    def to_tuple(self) -> tuple[Any]: ...

class FlaxCLIPVisionEmbeddings(nn.Module):
    config: CLIPVisionConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, pixel_values): ...

class FlaxCLIPTextEmbeddings(nn.Module):
    config: CLIPTextConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, input_ids, position_ids): ...

class FlaxCLIPAttention(nn.Module):
    config: CLIPTextConfig | CLIPVisionConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        hidden_states,
        attention_mask=...,
        deterministic: bool = ...,
        output_attentions: bool = ...,
    ):  # -> tuple[Any, Any] | tuple[Any]:
        ...

class FlaxCLIPMLP(nn.Module):
    config: CLIPTextConfig | CLIPVisionConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states): ...

class FlaxCLIPEncoderLayer(nn.Module):
    config: CLIPTextConfig | CLIPVisionConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        hidden_states,
        attention_mask,
        deterministic: bool = ...,
        output_attentions: bool = ...,
    ):  # -> tuple[Any, Any] | tuple[Any]:
        ...

class FlaxCLIPLayerCollection(nn.Module):
    config: CLIPTextConfig | CLIPVisionConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        hidden_states,
        attention_mask=...,
        deterministic: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any, ...] | FlaxBaseModelOutput:
        ...

class FlaxCLIPEncoder(nn.Module):
    config: CLIPTextConfig | CLIPVisionConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        inputs_embeds,
        attention_mask=...,
        deterministic: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any, ...] | FlaxBaseModelOutput:
        ...

class FlaxCLIPTextTransformer(nn.Module):
    config: CLIPTextConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        deterministic: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any, Any, *tuple[Any, ...]] | FlaxBaseModelOutputWithPooling:
        ...

class FlaxCLIPVisionTransformer(nn.Module):
    config: CLIPVisionConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        pixel_values=...,
        deterministic: bool = ...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict: bool = ...,
    ):  # -> tuple[Any, Any, *tuple[Any, ...]] | FlaxBaseModelOutputWithPooling:
        ...

class FlaxCLIPTextPreTrainedModel(FlaxPreTrainedModel):
    config_class = CLIPTextConfig
    module_class: nn.Module = ...
    def __init__(
        self,
        config: CLIPTextConfig,
        input_shape=...,
        seed: int = ...,
        dtype: jnp.dtype = ...,
        _do_init: bool = ...,
        **kwargs,
    ) -> None: ...
    def init_weights(
        self, rng: jax.random.PRNGKey, input_shape: tuple, params: FrozenDict = ...
    ) -> FrozenDict: ...
    def __call__(
        self,
        input_ids,
        attention_mask=...,
        position_ids=...,
        params: dict | None = ...,
        dropout_rng: jax.random.PRNGKey = ...,
        train: bool = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ): ...

class FlaxCLIPVisionPreTrainedModel(FlaxPreTrainedModel):
    config_class = CLIPVisionConfig
    main_input_name = ...
    module_class: nn.Module = ...
    def __init__(
        self,
        config: CLIPVisionConfig,
        input_shape: tuple | None = ...,
        seed: int = ...,
        dtype: jnp.dtype = ...,
        _do_init: bool = ...,
        **kwargs,
    ) -> None: ...
    def init_weights(
        self, rng: jax.random.PRNGKey, input_shape: tuple, params: FrozenDict = ...
    ) -> FrozenDict: ...
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

class FlaxCLIPPreTrainedModel(FlaxPreTrainedModel):
    config_class = CLIPConfig
    module_class: nn.Module = ...
    def __init__(
        self,
        config: CLIPConfig,
        input_shape: tuple | None = ...,
        seed: int = ...,
        dtype: jnp.dtype = ...,
        _do_init: bool = ...,
        **kwargs,
    ) -> None: ...
    def init_weights(
        self, rng: jax.random.PRNGKey, input_shape: tuple, params: FrozenDict = ...
    ) -> FrozenDict: ...
    def __call__(
        self,
        input_ids,
        pixel_values,
        attention_mask=...,
        position_ids=...,
        params: dict | None = ...,
        dropout_rng: jax.random.PRNGKey = ...,
        train: bool = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ): ...
    def get_text_features(
        self,
        input_ids,
        attention_mask=...,
        position_ids=...,
        params: dict | None = ...,
        dropout_rng: jax.random.PRNGKey = ...,
        train=...,
    ): ...
    def get_image_features(
        self,
        pixel_values,
        params: dict | None = ...,
        dropout_rng: jax.random.PRNGKey = ...,
        train=...,
    ): ...

class FlaxCLIPTextModule(nn.Module):
    config: CLIPTextConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        deterministic: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any, Any, *tuple[Any, ...]] | FlaxBaseModelOutputWithPooling:
        ...

class FlaxCLIPTextModel(FlaxCLIPTextPreTrainedModel): ...

FLAX_CLIP_TEXT_MODEL_DOCSTRING = ...

class FlaxCLIPTextModelWithProjectionModule(nn.Module):
    config: CLIPTextConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        deterministic: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any, Any, *tuple[Any, ...]] | FlaxCLIPTextModelOutput:
        ...

class FlaxCLIPTextModelWithProjection(FlaxCLIPTextPreTrainedModel): ...

FLAX_CLIP_TEXT_MODEL_WITH_PROJECTION_DOCSTRING = ...

class FlaxCLIPVisionModule(nn.Module):
    config: CLIPVisionConfig
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

class FlaxCLIPVisionModel(FlaxCLIPVisionPreTrainedModel): ...

FLAX_CLIP_VISION_MODEL_DOCSTRING = ...

class FlaxCLIPModule(nn.Module):
    config: CLIPConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_ids=...,
        pixel_values=...,
        attention_mask=...,
        position_ids=...,
        deterministic: bool = ...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
    ):  # -> tuple[Array, Array, Any, Any, tuple[Any, Any, *tuple[Any, ...]] | Any | FlaxBaseModelOutputWithPooling, tuple[Any, Any, *tuple[Any, ...]] | Any | FlaxBaseModelOutputWithPooling] | FlaxCLIPOutput:
        ...

@add_start_docstrings(CLIP_START_DOCSTRING)
class FlaxCLIPModel(FlaxCLIPPreTrainedModel): ...

FLAX_CLIP_MODEL_DOCSTRING = ...
__all__ = [
    "FlaxCLIPModel",
    "FlaxCLIPPreTrainedModel",
    "FlaxCLIPTextModel",
    "FlaxCLIPTextModelWithProjection",
    "FlaxCLIPTextPreTrainedModel",
    "FlaxCLIPVisionModel",
    "FlaxCLIPVisionPreTrainedModel",
]
