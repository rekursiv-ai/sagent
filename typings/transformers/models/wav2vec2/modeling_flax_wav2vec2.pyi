from flax.core.frozen_dict import FrozenDict

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

from .configuration_wav2vec2 import Wav2Vec2Config
from ...modeling_flax_utils import FlaxPreTrainedModel
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)

"""Flax Wav2Vec2 model."""
logger = ...

@flax.struct.dataclass
class FlaxWav2Vec2BaseModelOutput(ModelOutput):
    last_hidden_state: jnp.ndarray = ...
    extract_features: jnp.ndarray = ...
    hidden_states: tuple[jnp.ndarray] | None = ...
    attentions: tuple[jnp.ndarray] | None = ...

@flax.struct.dataclass
class FlaxWav2Vec2ForPreTrainingOutput(ModelOutput):
    projected_states: jnp.ndarray = ...
    projected_quantized_states: jnp.ndarray = ...
    codevector_perplexity: jnp.ndarray = ...
    hidden_states: tuple[jnp.ndarray] | None = ...
    attentions: tuple[jnp.ndarray] | None = ...

WAV2VEC2_START_DOCSTRING = ...
WAV2VEC2_INPUTS_DOCSTRING = ...

class FlaxWav2Vec2LayerNormConvLayer(nn.Module):
    config: Wav2Vec2Config
    layer_id: int = ...
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states): ...

class FlaxConvWithWeightNorm(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states): ...

class FlaxWav2Vec2PositionalConvEmbedding(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states): ...

class FlaxConvLayersCollection(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states): ...

class FlaxWav2Vec2FeatureEncoder(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, input_values, freeze_feature_encoder=...): ...

class FlaxWav2Vec2FeatureProjection(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states, deterministic=...):  # -> tuple[Any, Any]:
        ...

class FlaxWav2Vec2Attention(nn.Module):
    config: Wav2Vec2Config
    embed_dim: int
    num_heads: int
    dropout: float = ...
    bias: bool = ...
    dtype: jnp.dtype = ...
    def setup(self) -> None: ...
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        key_value_states: jnp.ndarray | None = ...,
        attention_mask: jnp.ndarray | None = ...,
        deterministic: bool = ...,
    ) -> tuple[jnp.ndarray]: ...

class FlaxWav2Vec2FeedForward(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states, deterministic=...): ...

class FlaxWav2Vec2EncoderLayerStableLayerNorm(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        hidden_states,
        attention_mask=...,
        deterministic=...,
        output_attentions=...,
    ):  # -> tuple[Any, Any] | tuple[Any]:
        ...

class FlaxWav2Vec2EncoderLayerStableLayerNormCollection(nn.Module):
    config: Wav2Vec2Config
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
    ):  # -> tuple[Any | tuple[Any, ...] | tuple[()], ...] | FlaxBaseModelOutput:
        ...

class FlaxWav2Vec2StableLayerNormEncoder(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        hidden_states,
        attention_mask=...,
        deterministic=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
    ):  # -> tuple[Any | tuple[Any, ...] | tuple[()], ...] | FlaxBaseModelOutput:
        ...

class FlaxWav2Vec2GumbelVectorQuantizer(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self, hidden_states, mask_time_indices=..., deterministic=..., temperature=...
    ):  # -> tuple[Any, Array]:
        ...

class FlaxWav2Vec2Adapter(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states, deterministic=...): ...

class FlaxWav2Vec2AdapterLayer(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states): ...

class FlaxWav2Vec2AdapterLayersCollection(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states): ...

class FlaxWav2Vec2PreTrainedModel(FlaxPreTrainedModel):
    config_class = Wav2Vec2Config
    base_model_prefix: str = ...
    main_input_name = ...
    module_class: nn.Module = ...
    def __init__(
        self,
        config: Wav2Vec2Config,
        input_shape: tuple = ...,
        seed: int = ...,
        dtype: jnp.dtype = ...,
        _do_init: bool = ...,
        **kwargs,
    ) -> None: ...
    def init_weights(
        self, rng: jax.random.PRNGKey, input_shape: tuple, params: FrozenDict = ...
    ) -> FrozenDict: ...
    @add_start_docstrings_to_model_forward(WAV2VEC2_INPUTS_DOCSTRING)
    def __call__(
        self,
        input_values,
        attention_mask=...,
        mask_time_indices=...,
        params: dict | None = ...,
        dropout_rng: jax.random.PRNGKey = ...,
        train: bool = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        freeze_feature_encoder: bool = ...,
        return_dict: bool | None = ...,
    ): ...

class FlaxWav2Vec2Module(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_values,
        attention_mask=...,
        mask_time_indices=...,
        deterministic=...,
        output_attentions=...,
        output_hidden_states=...,
        freeze_feature_encoder=...,
        return_dict=...,
    ):  # -> tuple[Any | tuple[Any, ...] | tuple[()], Any, *tuple[Any | tuple[Any, ...] | tuple[()], ...]] | FlaxWav2Vec2BaseModelOutput:
        ...

@add_start_docstrings(
    ...,
    WAV2VEC2_START_DOCSTRING,
)
class FlaxWav2Vec2Model(FlaxWav2Vec2PreTrainedModel): ...

FLAX_WAV2VEC2_MODEL_DOCSTRING = ...

class FlaxWav2Vec2ForCTCModule(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_values,
        attention_mask=...,
        mask_time_indices=...,
        deterministic=...,
        output_attentions=...,
        output_hidden_states=...,
        freeze_feature_encoder=...,
        return_dict=...,
    ):  # -> tuple[Any, *tuple[Any | tuple[Any, ...] | tuple[()], ...]] | FlaxCausalLMOutput:
        ...

@add_start_docstrings(
    ...,
    WAV2VEC2_START_DOCSTRING,
)
class FlaxWav2Vec2ForCTC(FlaxWav2Vec2PreTrainedModel): ...

FLAX_WAV2VEC2_FOR_CTC_DOCSTRING = ...

class FlaxWav2Vec2ForPreTrainingModule(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_values,
        attention_mask=...,
        mask_time_indices=...,
        gumbel_temperature: int = ...,
        deterministic: bool = ...,
        output_attentions=...,
        output_hidden_states=...,
        freeze_feature_encoder=...,
        return_dict=...,
    ):  # -> tuple[Any, Any, Array, *tuple[Any | tuple[Any, ...] | tuple[()], ...]] | FlaxWav2Vec2ForPreTrainingOutput:
        ...

@add_start_docstrings(
    ...,
    WAV2VEC2_START_DOCSTRING,
)
class FlaxWav2Vec2ForPreTraining(FlaxWav2Vec2PreTrainedModel):
    @add_start_docstrings_to_model_forward(WAV2VEC2_INPUTS_DOCSTRING)
    def __call__(
        self,
        input_values,
        attention_mask=...,
        mask_time_indices=...,
        gumbel_temperature: int = ...,
        params: dict | None = ...,
        dropout_rng: jax.random.PRNGKey = ...,
        gumbel_rng: jax.random.PRNGKey = ...,
        train: bool = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        freeze_feature_encoder: bool = ...,
        return_dict: bool | None = ...,
    ): ...

FLAX_WAV2VEC2_FOR_PRETRAINING_DOCSTRING = ...
__all__ = [
    "FlaxWav2Vec2ForCTC",
    "FlaxWav2Vec2ForPreTraining",
    "FlaxWav2Vec2Model",
    "FlaxWav2Vec2PreTrainedModel",
]
