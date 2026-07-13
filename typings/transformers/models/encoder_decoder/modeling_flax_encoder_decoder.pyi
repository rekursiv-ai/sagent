import os

from flax.core.frozen_dict import FrozenDict
from jax.random import PRNGKey

import flax.linen as nn
import jax
import jax.numpy as jnp

from .configuration_encoder_decoder import EncoderDecoderConfig
from ...modeling_flax_outputs import (
    FlaxBaseModelOutput,
    FlaxCausalLMOutputWithCrossAttentions,
    FlaxSeq2SeqLMOutput,
)
from ...modeling_flax_utils import FlaxPreTrainedModel
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)

"""Classes to support Flax Encoder-Decoder architectures"""
logger = ...
_CONFIG_FOR_DOC = ...
ENCODER_DECODER_START_DOCSTRING = ...
ENCODER_DECODER_INPUTS_DOCSTRING = ...
ENCODER_DECODER_ENCODE_INPUTS_DOCSTRING = ...
ENCODER_DECODER_DECODE_INPUTS_DOCSTRING = ...

class FlaxEncoderDecoderModule(nn.Module):
    config: EncoderDecoderConfig
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
    ):  # -> FlaxSeq2SeqLMOutput:
        ...

@add_start_docstrings(ENCODER_DECODER_START_DOCSTRING)
class FlaxEncoderDecoderModel(FlaxPreTrainedModel):
    config_class = EncoderDecoderConfig
    base_model_prefix = ...
    module_class = FlaxEncoderDecoderModule
    def __init__(
        self,
        config: EncoderDecoderConfig,
        input_shape: tuple | None = ...,
        seed: int = ...,
        dtype: jnp.dtype = ...,
        _do_init: bool = ...,
        **kwargs,
    ) -> None: ...
    def init_weights(
        self, rng: jax.random.PRNGKey, input_shape: tuple, params: FrozenDict = ...
    ) -> FrozenDict: ...
    def init_cache(self, batch_size, max_length, encoder_outputs): ...
    @add_start_docstrings(ENCODER_DECODER_ENCODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=FlaxBaseModelOutput, config_class=_CONFIG_FOR_DOC
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
    ):  # -> FlaxBaseModelOutput:
        ...
    @add_start_docstrings(ENCODER_DECODER_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=FlaxCausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC
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
    @add_start_docstrings_to_model_forward(ENCODER_DECODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=FlaxSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC
    )
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
    @classmethod
    def from_encoder_decoder_pretrained(
        cls,
        encoder_pretrained_model_name_or_path: str | os.PathLike | None = ...,
        decoder_pretrained_model_name_or_path: str | os.PathLike | None = ...,
        *model_args,
        **kwargs,
    ) -> FlaxPreTrainedModel: ...

__all__ = ["FlaxEncoderDecoderModel"]
