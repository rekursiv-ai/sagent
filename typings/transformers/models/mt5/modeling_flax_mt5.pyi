import jax.numpy as jnp

from .configuration_mt5 import MT5Config
from ..t5.modeling_flax_t5 import (
    FlaxT5EncoderModel,
    FlaxT5ForConditionalGeneration,
    FlaxT5Model,
)

"""Flax mT5 model."""
logger = ...
_CONFIG_FOR_DOC = ...

def shift_tokens_right(
    input_ids: jnp.ndarray, pad_token_id: int, decoder_start_token_id: int
) -> jnp.ndarray: ...

class FlaxMT5Model(FlaxT5Model):
    model_type = ...
    config_class = MT5Config

class FlaxMT5EncoderModel(FlaxT5EncoderModel):
    model_type = ...
    config_class = MT5Config

class FlaxMT5ForConditionalGeneration(FlaxT5ForConditionalGeneration):
    model_type = ...
    config_class = MT5Config

__all__ = ["FlaxMT5EncoderModel", "FlaxMT5ForConditionalGeneration", "FlaxMT5Model"]
