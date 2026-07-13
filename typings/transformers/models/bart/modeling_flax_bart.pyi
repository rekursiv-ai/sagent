from collections.abc import Callable

from flax.core.frozen_dict import FrozenDict
from jax.random import PRNGKey

import flax.linen as nn
import jax
import jax.numpy as jnp

from .configuration_bart import BartConfig
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

"""Flax Bart model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
BART_START_DOCSTRING = ...
BART_INPUTS_DOCSTRING = ...
BART_ENCODE_INPUTS_DOCSTRING = ...
BART_DECODE_INPUTS_DOCSTRING = ...

def shift_tokens_right(
    input_ids: jnp.ndarray, pad_token_id: int, decoder_start_token_id: int
) -> jnp.ndarray: ...

class FlaxBartAttention(nn.Module):
    config: BartConfig
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

class FlaxBartEncoderLayer(nn.Module):
    config: BartConfig
    dtype: jnp.dtype = ...
    def setup(self) -> None: ...
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray,
        output_attentions: bool = ...,
        deterministic: bool = ...,
    ) -> tuple[jnp.ndarray]: ...

class FlaxBartEncoderLayerCollection(nn.Module):
    config: BartConfig
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

class FlaxBartDecoderLayer(nn.Module):
    config: BartConfig
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

class FlaxBartDecoderLayerCollection(nn.Module):
    config: BartConfig
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

class FlaxBartClassificationHead(nn.Module):
    config: BartConfig
    inner_dim: int
    num_classes: int
    pooler_dropout: float
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states: jnp.ndarray, deterministic: bool):  # -> ndarray:
        ...

class FlaxBartEncoder(nn.Module):
    config: BartConfig
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
    ):  # -> tuple[Any | ndarray | tuple[Any | ndarray | None, ...] | tuple[()] | tuple[ndarray | None, ...], ...] | FlaxBaseModelOutput:
        ...

class FlaxBartDecoder(nn.Module):
    config: BartConfig
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
    ):  # -> tuple[Any | ndarray | tuple[Any | ndarray | None, ...] | tuple[()] | tuple[ndarray | None, ...], ...] | FlaxBaseModelOutputWithPastAndCrossAttentions:
        ...

class FlaxBartModule(nn.Module):
    config: BartConfig
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
    ):  # -> tuple[Any | ndarray | tuple[Any | ndarray | None, ...] | tuple[()] | tuple[ndarray | None, ...], ...] | FlaxSeq2SeqModelOutput:
        ...

class FlaxBartPreTrainedModel(FlaxPreTrainedModel):
    config_class = BartConfig
    base_model_prefix: str = ...
    module_class: nn.Module = ...
    def __init__(
        self,
        config: BartConfig,
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
    @add_start_docstrings(BART_ENCODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxBaseModelOutput, config_class=BartConfig)
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
    @add_start_docstrings(BART_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=FlaxBaseModelOutputWithPastAndCrossAttentions,
        config_class=BartConfig,
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
    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
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
    BART_START_DOCSTRING,
)
class FlaxBartModel(FlaxBartPreTrainedModel):
    config: BartConfig
    dtype: jnp.dtype = ...
    module_class = ...

class FlaxBartForConditionalGenerationModule(nn.Module):
    config: BartConfig
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
    ):  # -> tuple[Any, *tuple[Any | ndarray | tuple[Any | ndarray | None, ...] | tuple[()] | tuple[ndarray | None, ...], ...]] | FlaxSeq2SeqLMOutput:
        ...

@add_start_docstrings(
    ...,
    BART_START_DOCSTRING,
)
class FlaxBartForConditionalGeneration(FlaxBartPreTrainedModel):
    module_class = ...
    dtype: jnp.dtype = ...
    @add_start_docstrings(BART_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=FlaxCausalLMOutputWithCrossAttentions, config_class=BartConfig
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

FLAX_BART_CONDITIONAL_GENERATION_DOCSTRING = ...

class FlaxBartForSequenceClassificationModule(nn.Module):
    config: BartConfig
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
    ):  # -> tuple[Any | ndarray, *tuple[Any | ndarray | tuple[Any | ndarray | None, ...] | tuple[()] | tuple[ndarray | None, ...], ...]] | FlaxSeq2SeqSequenceClassifierOutput:
        ...

@add_start_docstrings(
    ...,
    BART_START_DOCSTRING,
)
class FlaxBartForSequenceClassification(FlaxBartPreTrainedModel):
    module_class = ...
    dtype = jnp.float32

class FlaxBartForQuestionAnsweringModule(nn.Module):
    config: BartConfig
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
    ):  # -> tuple[Array, Array, *tuple[Any | ndarray | tuple[Any | ndarray | None, ...] | tuple[()] | tuple[ndarray | None, ...], ...]] | FlaxSeq2SeqQuestionAnsweringModelOutput:
        ...

@add_start_docstrings(
    ...,
    BART_START_DOCSTRING,
)
class FlaxBartForQuestionAnswering(FlaxBartPreTrainedModel):
    module_class = ...
    dtype = jnp.float32

class FlaxBartDecoderPreTrainedModel(FlaxPreTrainedModel):
    config_class = BartConfig
    base_model_prefix: str = ...
    module_class: nn.Module = ...
    def __init__(
        self,
        config: BartConfig,
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
    @add_start_docstrings_to_model_forward(BART_DECODE_INPUTS_DOCSTRING)
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

class FlaxBartDecoderWrapper(nn.Module):
    config: BartConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self, *args, **kwargs
    ):  # -> tuple[Any | ndarray | tuple[Any | ndarray | None, ...] | tuple[()] | tuple[ndarray | None, ...], ...] | FlaxBaseModelOutputWithPastAndCrossAttentions:
        ...

class FlaxBartForCausalLMModule(nn.Module):
    config: BartConfig
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
    ):  # -> tuple[Any, *tuple[Any | ndarray | tuple[Any | ndarray | None, ...] | tuple[()] | tuple[ndarray | None, ...], ...]] | FlaxCausalLMOutputWithCrossAttentions:
        ...

@add_start_docstrings(
    ...,
    BART_START_DOCSTRING,
)
class FlaxBartForCausalLM(FlaxBartDecoderPreTrainedModel):
    module_class = ...
    def prepare_inputs_for_generation(
        self, input_ids, max_length, attention_mask: jax.Array | None = ...
    ):  # -> dict[str, Any | Array]:
        ...
    def update_inputs_for_generation(self, model_outputs, model_kwargs): ...

__all__ = [
    "FlaxBartDecoderPreTrainedModel",
    "FlaxBartForCausalLM",
    "FlaxBartForConditionalGeneration",
    "FlaxBartForQuestionAnswering",
    "FlaxBartForSequenceClassification",
    "FlaxBartModel",
    "FlaxBartPreTrainedModel",
]
