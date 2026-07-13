from flax.core.frozen_dict import FrozenDict

import flax.linen as nn
import jax
import jax.numpy as jnp

from .configuration_vision_text_dual_encoder import VisionTextDualEncoderConfig
from ...modeling_flax_utils import FlaxPreTrainedModel
from ...utils import add_start_docstrings

"""Flax VisionTextDualEncoder model."""
logger = ...
_CONFIG_FOR_DOC = ...
VISION_TEXT_DUAL_ENCODER_START_DOCSTRING = ...
VISION_TEXT_DUAL_ENCODER_INPUTS_DOCSTRING = ...

class FlaxVisionTextDualEncoderModule(nn.Module):
    config: VisionTextDualEncoderConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_ids=...,
        pixel_values=...,
        attention_mask=...,
        position_ids=...,
        token_type_ids=...,
        deterministic: bool = ...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
    ):  # -> tuple[Array, Array, Any, Any, Any, Any] | FlaxCLIPOutput:
        ...

@add_start_docstrings(VISION_TEXT_DUAL_ENCODER_START_DOCSTRING)
class FlaxVisionTextDualEncoderModel(FlaxPreTrainedModel):
    config_class = VisionTextDualEncoderConfig
    module_class = FlaxVisionTextDualEncoderModule
    def __init__(
        self,
        config: VisionTextDualEncoderConfig,
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
        token_type_ids=...,
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
        token_type_ids=...,
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
    @classmethod
    def from_vision_text_pretrained(
        cls,
        vision_model_name_or_path: str | None = ...,
        text_model_name_or_path: str | None = ...,
        *model_args,
        **kwargs,
    ) -> FlaxPreTrainedModel: ...

VISION_TEXT_DUAL_ENCODER_MODEL_DOCSTRING = ...
__all__ = ["FlaxVisionTextDualEncoderModel"]
