from collections.abc import Callable

from flax.core.frozen_dict import FrozenDict
from jax.random import PRNGKey

import flax.linen as nn
import jax
import jax.numpy as jnp

from .configuration_mbart import MBartConfig
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

"""Flax MBart model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
MBART_START_DOCSTRING = ...
MBART_INPUTS_DOCSTRING = ...
MBART_ENCODE_INPUTS_DOCSTRING = ...
MBART_DECODE_INPUTS_DOCSTRING = ...

def shift_tokens_right(input_ids: jnp.ndarray, pad_token_id: int) -> jnp.ndarray: ...

class FlaxMBartAttention(nn.Module):
    config: MBartConfig
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

class FlaxMBartEncoderLayer(nn.Module):
    config: MBartConfig
    dtype: jnp.dtype = ...
    def setup(self) -> None: ...
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray,
        output_attentions: bool = ...,
        deterministic: bool = ...,
    ) -> tuple[jnp.ndarray]: ...

class FlaxMBartEncoderLayerCollection(nn.Module):
    config: MBartConfig
    dtype: jnp.dtype = ...
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
    ):  # -> tuple[Any | ndarray | tuple[Any | ndarray | None, ...] | tuple[()] | tuple[ndarray | None, ...], ...] | FlaxBaseModelOutput:
        ...

class FlaxMBartDecoderLayer(nn.Module):
    config: MBartConfig
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

class FlaxMBartDecoderLayerCollection(nn.Module):
    config: MBartConfig
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

class FlaxMBartClassificationHead(nn.Module):
    config: MBartConfig
    inner_dim: int
    num_classes: int
    pooler_dropout: float
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states: jnp.ndarray, deterministic: bool):  # -> ndarray:
        ...

class FlaxMBartEncoder(nn.Module):
    config: MBartConfig
    embed_tokens: nn.Embed
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
    ):  # -> tuple[Any | tuple[Any | ndarray | None, ...] | tuple[Any, ...] | tuple[ndarray | None, ...] | ndarray | tuple[()], ...] | FlaxBaseModelOutput:
        ...

class FlaxMBartDecoder(nn.Module):
    config: MBartConfig
    embed_tokens: nn.Embed
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

class FlaxMBartModule(nn.Module):
    config: MBartConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_ids,
        attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
        position_ids,
        decoder_position_ids,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
        deterministic: bool = ...,
    ):  # -> tuple[Any | tuple[Any | ndarray | None, ...] | tuple[Any, ...] | tuple[ndarray | None, ...] | ndarray | tuple[()], ...] | FlaxSeq2SeqModelOutput:
        ...

class FlaxMBartPreTrainedModel(FlaxPreTrainedModel):
    config_class = MBartConfig
    base_model_prefix: str = ...
    module_class: nn.Module = ...
    def __init__(
        self,
        config: MBartConfig,
        input_shape: tuple[int] = ...,
        seed: int = ...,
        dtype: jnp.dtype = ...,
        _do_init: bool = ...,
        **kwargs,
    ) -> None: ...
    def init_weights(
        self, rng: jax.random.PRNGKey, input_shape: tuple, params: FrozenDict = ...
    ) -> FrozenDict: ...
    def init_cache(self, batch_size, max_length, encoder_outputs): ...
    @add_start_docstrings(MBART_ENCODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=FlaxBaseModelOutput, config_class=MBartConfig
    )
    def encode(
        self,
        input_ids: jnp.ndarray,
        attention_mask: jnp.ndarray | None = ...,
        position_ids: jnp.ndarray | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        train: bool = ...,
        params: dict | None = ...,
        dropout_rng: PRNGKey = ...,
    ): ...
    @add_start_docstrings(MBART_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=FlaxBaseModelOutputWithPastAndCrossAttentions,
        config_class=MBartConfig,
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
    @add_start_docstrings_to_model_forward(MBART_INPUTS_DOCSTRING)
    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: jnp.ndarray | None = ...,
        decoder_input_ids: jnp.ndarray | None = ...,
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
    MBART_START_DOCSTRING,
)
class FlaxMBartModel(FlaxMBartPreTrainedModel):
    config: MBartConfig
    dtype: jnp.dtype = ...
    module_class = ...

class FlaxMBartForConditionalGenerationModule(nn.Module):
    config: MBartConfig
    dtype: jnp.dtype = ...
    bias_init: Callable[..., jnp.ndarray] = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_ids,
        attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
        position_ids,
        decoder_position_ids,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
        deterministic: bool = ...,
    ):  # -> tuple[Any, *tuple[Any | tuple[Any | ndarray | None, ...] | tuple[Any, ...] | tuple[ndarray | None, ...] | ndarray | tuple[()], ...]] | FlaxSeq2SeqLMOutput:
        ...

@add_start_docstrings(
    ...,
    MBART_START_DOCSTRING,
)
class FlaxMBartForConditionalGeneration(FlaxMBartPreTrainedModel):
    module_class = ...
    dtype: jnp.dtype = ...
    @add_start_docstrings(MBART_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=FlaxCausalLMOutputWithCrossAttentions, config_class=MBartConfig
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

FLAX_MBART_CONDITIONAL_GENERATION_DOCSTRING = ...

class FlaxMBartForSequenceClassificationModule(nn.Module):
    config: MBartConfig
    dtype: jnp.dtype = ...
    num_labels: int | None = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_ids,
        attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
        position_ids,
        decoder_position_ids,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
        deterministic: bool = ...,
    ):  # -> tuple[Any | ndarray, *tuple[Any | tuple[Any | ndarray | None, ...] | tuple[Any, ...] | tuple[ndarray | None, ...] | ndarray | tuple[()], ...]] | FlaxSeq2SeqSequenceClassifierOutput:
        ...

@add_start_docstrings(
    ...,
    MBART_START_DOCSTRING,
)
class FlaxMBartForSequenceClassification(FlaxMBartPreTrainedModel):
    module_class = ...
    dtype = jnp.float32

class FlaxMBartForQuestionAnsweringModule(nn.Module):
    config: MBartConfig
    dtype: jnp.dtype = ...
    num_labels = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_ids,
        attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
        position_ids,
        decoder_position_ids,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
        deterministic: bool = ...,
    ):  # -> tuple[Array, Array, *tuple[Any | tuple[Any | ndarray | None, ...] | tuple[Any, ...] | tuple[ndarray | None, ...] | ndarray | tuple[()], ...]] | FlaxSeq2SeqQuestionAnsweringModelOutput:
        ...

@add_start_docstrings(
    ...,
    MBART_START_DOCSTRING,
)
class FlaxMBartForQuestionAnswering(FlaxMBartPreTrainedModel):
    module_class = ...
    dtype = jnp.float32

__all__ = [
    "FlaxMBartForConditionalGeneration",
    "FlaxMBartForQuestionAnswering",
    "FlaxMBartForSequenceClassification",
    "FlaxMBartModel",
    "FlaxMBartPreTrainedModel",
]
