from flax.core.frozen_dict import FrozenDict
from jax.random import PRNGKey

import flax.linen as nn
import jax
import jax.numpy as jnp

from .configuration_whisper import WhisperConfig
from ...modeling_flax_outputs import (
    FlaxBaseModelOutput,
    FlaxBaseModelOutputWithPastAndCrossAttentions,
    FlaxCausalLMOutputWithCrossAttentions,
)
from ...modeling_flax_utils import FlaxPreTrainedModel
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)

"""Flax whisper model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
remat = ...

def sinusoidal_embedding_init(key, shape, dtype=...) -> jax.Array: ...

WHISPER_START_DOCSTRING = ...
WHISPER_INPUTS_DOCSTRING = ...
WHISPER_ENCODE_INPUTS_DOCSTRING = ...
WHISPER_DECODE_INPUTS_DOCSTRING = ...

class FlaxWhisperAttention(nn.Module):
    config: WhisperConfig
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

class FlaxWhisperEncoderLayer(nn.Module):
    config: WhisperConfig
    dtype: jnp.dtype = ...
    def setup(self) -> None: ...
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray,
        output_attentions: bool = ...,
        deterministic: bool = ...,
    ) -> tuple[jnp.ndarray]: ...

class FlaxWhisperEncoderLayerCollection(nn.Module):
    config: WhisperConfig
    dtype: jnp.dtype = ...
    gradient_checkpointing: bool = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        hidden_states,
        attention_mask,
        deterministic: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any | ndarray | tuple[Any | ndarray | None, ...] | tuple[()], ...] | FlaxBaseModelOutput:
        ...

class FlaxWhisperDecoderLayer(nn.Module):
    config: WhisperConfig
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

class FlaxWhisperDecoderLayerCollection(nn.Module):
    config: WhisperConfig
    dtype: jnp.dtype = ...
    gradient_checkpointing: bool = ...
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
    ):  # -> tuple[Any | ndarray | tuple[Any | ndarray | None, ...] | tuple[()], ...] | FlaxBaseModelOutputWithPastAndCrossAttentions:
        ...

class FlaxWhisperEncoder(nn.Module):
    config: WhisperConfig
    dtype: jnp.dtype = ...
    gradient_checkpointing: bool = ...
    def setup(self) -> None: ...
    def __call__(
        self,
        input_features: jnp.ndarray,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
        deterministic: bool = ...,
    ) -> tuple[jnp.ndarray]: ...

class FlaxWhisperDecoder(nn.Module):
    config: WhisperConfig
    dtype: jnp.dtype = ...
    gradient_checkpointing: bool = ...
    def setup(self) -> None: ...
    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: jnp.ndarray,
        position_ids: jnp.ndarray,
        encoder_hidden_states: jnp.ndarray | None = ...,
        init_cache: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
        deterministic: bool = ...,
    ) -> tuple[jnp.ndarray]: ...

class FlaxWhisperModule(nn.Module):
    config: WhisperConfig
    dtype: jnp.dtype = ...
    gradient_checkpointing: bool = ...
    def setup(self) -> None: ...
    def __call__(
        self,
        input_features: jnp.ndarray,
        decoder_input_ids: jnp.ndarray,
        decoder_attention_mask: jnp.ndarray,
        decoder_position_ids: jnp.ndarray,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
        deterministic: bool = ...,
    ):  # -> tuple[ndarray, ndarray] | FlaxSeq2SeqModelOutput:
        ...

class FlaxWhisperPreTrainedModel(FlaxPreTrainedModel):
    config_class = WhisperConfig
    base_model_prefix: str = ...
    main_input_name = ...
    module_class: nn.Module = ...
    def __init__(
        self,
        config: WhisperConfig,
        input_shape: tuple[int] | None = ...,
        seed: int = ...,
        dtype: jnp.dtype = ...,
        _do_init: bool = ...,
        gradient_checkpointing: bool = ...,
        **kwargs,
    ) -> None: ...
    def enable_gradient_checkpointing(self):  # -> None:
        ...
    def init_weights(
        self, rng: jax.random.PRNGKey, input_shape: tuple, params: FrozenDict = ...
    ) -> FrozenDict: ...
    def init_cache(self, batch_size, max_length, encoder_outputs): ...
    @add_start_docstrings(WHISPER_ENCODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=FlaxBaseModelOutput, config_class=WhisperConfig
    )
    def encode(
        self,
        input_features: jnp.ndarray,
        attention_mask: jnp.ndarray | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        train: bool = ...,
        params: dict | None = ...,
        dropout_rng: PRNGKey = ...,
        **kwargs,
    ): ...
    @add_start_docstrings(WHISPER_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=FlaxBaseModelOutputWithPastAndCrossAttentions,
        config_class=WhisperConfig,
    )
    def decode(
        self,
        decoder_input_ids,
        encoder_outputs,
        encoder_attention_mask: jnp.ndarray | None = ...,
        decoder_attention_mask: jnp.ndarray | None = ...,
        decoder_position_ids: jnp.ndarray | None = ...,
        past_key_values: dict | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        train: bool = ...,
        params: dict | None = ...,
        dropout_rng: PRNGKey = ...,
    ): ...
    @add_start_docstrings_to_model_forward(WHISPER_INPUTS_DOCSTRING)
    def __call__(
        self,
        input_features: jnp.ndarray,
        decoder_input_ids: jnp.ndarray,
        attention_mask: jnp.ndarray | None = ...,
        decoder_attention_mask: jnp.ndarray | None = ...,
        position_ids: jnp.ndarray | None = ...,
        decoder_position_ids: jnp.ndarray | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        train: bool = ...,
        params: dict | None = ...,
        dropout_rng: PRNGKey = ...,
    ): ...

@add_start_docstrings(
    ...,
    WHISPER_START_DOCSTRING,
)
class FlaxWhisperModel(FlaxWhisperPreTrainedModel):
    config: WhisperConfig
    dtype: jnp.dtype = ...

class FlaxWhisperForConditionalGenerationModule(nn.Module):
    config: WhisperConfig
    dtype: jnp.dtype = ...
    gradient_checkpointing: bool = ...
    def setup(self) -> None: ...
    def __call__(
        self,
        input_features,
        decoder_input_ids,
        decoder_attention_mask: jnp.ndarray = ...,
        decoder_position_ids: jnp.ndarray = ...,
        position_ids: jnp.ndarray = ...,
        attention_mask: jnp.ndarray = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
        deterministic: bool = ...,
    ):  # -> tuple[Any, ndarray] | FlaxSeq2SeqLMOutput:
        ...

@add_start_docstrings(
    "The Whisper Model with a language modeling head.", WHISPER_START_DOCSTRING
)
class FlaxWhisperForConditionalGeneration(FlaxWhisperPreTrainedModel):
    dtype: jnp.dtype = ...
    @add_start_docstrings(WHISPER_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=FlaxCausalLMOutputWithCrossAttentions, config_class=WhisperConfig
    )
    def decode(
        self,
        decoder_input_ids,
        encoder_outputs,
        encoder_attention_mask: jnp.ndarray | None = ...,
        decoder_attention_mask: jnp.ndarray | None = ...,
        decoder_position_ids: jnp.ndarray | None = ...,
        past_key_values: dict | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        train: bool = ...,
        params: dict | None = ...,
        dropout_rng: PRNGKey = ...,
    ):  # -> FlaxCausalLMOutputWithCrossAttentions:
        ...
    def generate(
        self,
        input_features,
        generation_config=...,
        logits_processor=...,
        return_timestamps=...,
        task=...,
        language=...,
        is_multilingual=...,
        **kwargs,
    ):  # -> FlaxGreedySearchOutput | FlaxSampleOutput | FlaxBeamSearchOutput:
        ...
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        max_length,
        attention_mask: jax.Array | None = ...,
        decoder_attention_mask: jax.Array | None = ...,
        encoder_outputs=...,
        **kwargs,
    ):  # -> dict[str, Any | Array | None]:
        ...
    def update_inputs_for_generation(self, model_outputs, model_kwargs): ...

FLAX_WHISPER_CONDITIONAL_GENERATION_DOCSTRING = ...

class FlaxWhisperForAudioClassificationModule(nn.Module):
    config: WhisperConfig
    dtype: jnp.dtype = ...
    gradient_checkpointing: bool = ...
    def setup(self) -> None: ...
    def __call__(
        self,
        input_features,
        encoder_outputs=...,
        output_attentions=...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any] | FlaxSequenceClassifierOutput:
        ...

@add_start_docstrings(
    ...,
    WHISPER_START_DOCSTRING,
)
class FlaxWhisperForAudioClassification(FlaxWhisperPreTrainedModel):
    dtype: jnp.dtype = ...
    def init_weights(
        self, rng: jax.random.PRNGKey, input_shape: tuple, params: FrozenDict = ...
    ) -> FrozenDict: ...
    @add_start_docstrings_to_model_forward(WHISPER_INPUTS_DOCSTRING)
    def __call__(
        self,
        input_features: jnp.ndarray,
        attention_mask: jnp.ndarray | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        train: bool = ...,
        params: dict | None = ...,
        dropout_rng: PRNGKey = ...,
        **kwargs,
    ): ...

FLAX_WHISPER_AUDIO_CLASSIFICATION_DOCSTRING = ...
__all__ = [
    "FlaxWhisperForAudioClassification",
    "FlaxWhisperForConditionalGeneration",
    "FlaxWhisperModel",
    "FlaxWhisperPreTrainedModel",
]
