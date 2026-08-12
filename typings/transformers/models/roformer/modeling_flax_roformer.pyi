from collections.abc import Callable

from flax.core.frozen_dict import FrozenDict

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from .configuration_roformer import RoFormerConfig
from ...modeling_flax_utils import FlaxPreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward

"""Flax RoFormer model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
ROFORMER_START_DOCSTRING = ...
ROFORMER_INPUTS_DOCSTRING = ...

def create_sinusoidal_positions(n_pos, dim):  # -> Array:
    ...

class FlaxRoFormerEmbeddings(nn.Module):
    config: RoFormerConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self, input_ids, token_type_ids, attention_mask, deterministic: bool = ...
    ): ...

class FlaxRoFormerSelfAttention(nn.Module):
    config: RoFormerConfig
    dtype: jnp.dtype = ...
    def setup(self) -> None: ...
    def __call__(
        self,
        hidden_states,
        attention_mask,
        sinusoidal_pos,
        layer_head_mask,
        deterministic=...,
        output_attentions: bool = ...,
    ):  # -> tuple[Array, Array | Any] | tuple[Array]:
        ...
    @staticmethod
    def apply_rotary_position_embeddings(
        sinusoidal_pos, query_layer, key_layer, value_layer=...
    ):  # -> tuple[Array, Array, Array] | tuple[Array, Array]:
        ...

class FlaxRoFormerSelfOutput(nn.Module):
    config: RoFormerConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states, input_tensor, deterministic: bool = ...): ...

class FlaxRoFormerAttention(nn.Module):
    config: RoFormerConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        hidden_states,
        attention_mask,
        sinusoidal_pos,
        layer_head_mask,
        deterministic=...,
        output_attentions: bool = ...,
    ):  # -> tuple[Any, Array | Any] | tuple[Any]:
        ...

class FlaxRoFormerIntermediate(nn.Module):
    config: RoFormerConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states): ...

class FlaxRoFormerOutput(nn.Module):
    config: RoFormerConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states, attention_output, deterministic: bool = ...): ...

class FlaxRoFormerLayer(nn.Module):
    config: RoFormerConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        hidden_states,
        attention_mask,
        sinusiodal_pos,
        layer_head_mask,
        deterministic: bool = ...,
        output_attentions: bool = ...,
    ):  # -> tuple[Any, Array | Any] | tuple[Any]:
        ...

class FlaxRoFormerLayerCollection(nn.Module):
    config: RoFormerConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        hidden_states,
        attention_mask,
        sinusoidal_pos,
        head_mask,
        deterministic: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any, ...] | FlaxBaseModelOutput:
        ...

class FlaxRoFormerEncoder(nn.Module):
    config: RoFormerConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        hidden_states,
        attention_mask,
        head_mask,
        deterministic: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any, ...] | FlaxBaseModelOutput:
        ...

class FlaxRoFormerPredictionHeadTransform(nn.Module):
    config: RoFormerConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states): ...

class FlaxRoFormerLMPredictionHead(nn.Module):
    config: RoFormerConfig
    dtype: jnp.dtype = ...
    bias_init: Callable[..., np.ndarray] = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states, shared_embedding=...): ...

class FlaxRoFormerOnlyMLMHead(nn.Module):
    config: RoFormerConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states, shared_embedding=...): ...

class FlaxRoFormerClassificationHead(nn.Module):
    config: RoFormerConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states, deterministic=...): ...

class FlaxRoFormerPreTrainedModel(FlaxPreTrainedModel):
    config_class = RoFormerConfig
    base_model_prefix = ...
    module_class: nn.Module = ...
    def __init__(
        self,
        config: RoFormerConfig,
        input_shape: tuple = ...,
        seed: int = ...,
        dtype: jnp.dtype = ...,
        _do_init: bool = ...,
        **kwargs,
    ) -> None: ...
    def init_weights(
        self, rng: jax.random.PRNGKey, input_shape: tuple, params: FrozenDict = ...
    ) -> FrozenDict: ...
    @add_start_docstrings_to_model_forward(
        ROFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    def __call__(
        self,
        input_ids,
        attention_mask=...,
        token_type_ids=...,
        head_mask=...,
        params: dict | None = ...,
        dropout_rng: jax.random.PRNGKey = ...,
        train: bool = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ): ...

class FlaxRoFormerModule(nn.Module):
    config: RoFormerConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        head_mask,
        deterministic: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any, *tuple[Any, ...]] | FlaxBaseModelOutput:
        ...

@add_start_docstrings(
    ...,
    ROFORMER_START_DOCSTRING,
)
class FlaxRoFormerModel(FlaxRoFormerPreTrainedModel): ...

class FlaxRoFormerForMaskedLMModule(nn.Module):
    config: RoFormerConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        head_mask,
        deterministic: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any, *tuple[Any, ...]] | FlaxMaskedLMOutput:
        ...

@add_start_docstrings(
    ...,
    ROFORMER_START_DOCSTRING,
)
class FlaxRoFormerForMaskedLM(FlaxRoFormerPreTrainedModel): ...

class FlaxRoFormerForSequenceClassificationModule(nn.Module):
    config: RoFormerConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        head_mask,
        deterministic: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any, *tuple[Any, ...]] | FlaxSequenceClassifierOutput:
        ...

@add_start_docstrings(
    ...,
    ROFORMER_START_DOCSTRING,
)
class FlaxRoFormerForSequenceClassification(FlaxRoFormerPreTrainedModel): ...

class FlaxRoFormerForMultipleChoiceModule(nn.Module):
    config: RoFormerConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        head_mask,
        deterministic: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any, *tuple[Any, ...]] | FlaxMultipleChoiceModelOutput:
        ...

@add_start_docstrings(
    ...,
    ROFORMER_START_DOCSTRING,
)
class FlaxRoFormerForMultipleChoice(FlaxRoFormerPreTrainedModel): ...

class FlaxRoFormerForTokenClassificationModule(nn.Module):
    config: RoFormerConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        head_mask,
        deterministic: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any, *tuple[Any, ...]] | FlaxTokenClassifierOutput:
        ...

@add_start_docstrings(
    ...,
    ROFORMER_START_DOCSTRING,
)
class FlaxRoFormerForTokenClassification(FlaxRoFormerPreTrainedModel): ...

class FlaxRoFormerForQuestionAnsweringModule(nn.Module):
    config: RoFormerConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        head_mask,
        deterministic: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Array, Array, *tuple[Any, ...]] | FlaxQuestionAnsweringModelOutput:
        ...

@add_start_docstrings(
    ...,
    ROFORMER_START_DOCSTRING,
)
class FlaxRoFormerForQuestionAnswering(FlaxRoFormerPreTrainedModel): ...

__all__ = [
    "FlaxRoFormerForMaskedLM",
    "FlaxRoFormerForMultipleChoice",
    "FlaxRoFormerForQuestionAnswering",
    "FlaxRoFormerForSequenceClassification",
    "FlaxRoFormerForTokenClassification",
    "FlaxRoFormerModel",
    "FlaxRoFormerPreTrainedModel",
]
