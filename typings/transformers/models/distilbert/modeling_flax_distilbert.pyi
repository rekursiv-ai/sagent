from collections.abc import Callable

from flax.core.frozen_dict import FrozenDict

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from .configuration_distilbert import DistilBertConfig
from ...modeling_flax_utils import FlaxPreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward

logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
FLAX_DISTILBERT_START_DOCSTRING = ...
DISTILBERT_INPUTS_DOCSTRING = ...

def get_angles(pos, i, d_model): ...
def positional_encoding(position, d_model):  # -> Array:
    ...

class FlaxEmbeddings(nn.Module):
    config: DistilBertConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, input_ids, deterministic: bool = ...): ...

class FlaxMultiHeadSelfAttention(nn.Module):
    config: DistilBertConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        query,
        key,
        value,
        mask,
        deterministic: bool = ...,
        output_attentions: bool = ...,
    ):  # -> tuple[Any, Any] | tuple[Any]:
        ...

class FlaxFFN(nn.Module):
    config: DistilBertConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states, deterministic: bool = ...): ...

class FlaxTransformerBlock(nn.Module):
    config: DistilBertConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        hidden_states,
        attn_mask,
        output_attentions: bool = ...,
        deterministic: bool = ...,
    ):  # -> tuple[Any, Any] | tuple[Any]:
        ...

class FlaxTransformer(nn.Module):
    config: DistilBertConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        hidden_states,
        attention_mask,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        deterministic: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any | tuple[()] | tuple[Any, ...], ...] | FlaxBaseModelOutput:
        ...

class FlaxTransformerEncoder(nn.Module):
    config: DistilBertConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        hidden_states,
        attention_mask,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        deterministic: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any | tuple[()] | tuple[Any, ...], ...] | FlaxBaseModelOutput:
        ...

class FlaxDistilBertLMDecoder(nn.Module):
    config: DistilBertConfig
    dtype: jnp.dtype = ...
    bias_init: Callable[..., np.ndarray] = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, inputs, kernel):  # -> Array:
        ...

class FlaxDistilBertPreTrainedModel(FlaxPreTrainedModel):
    config_class = DistilBertConfig
    base_model_prefix = ...
    module_class: nn.Module = ...
    def __init__(
        self,
        config: DistilBertConfig,
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
        DISTILBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    def __call__(
        self,
        input_ids,
        attention_mask=...,
        head_mask=...,
        params: dict | None = ...,
        dropout_rng: jax.random.PRNGKey = ...,
        train: bool = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ): ...

class FlaxDistilBertModule(nn.Module):
    config: DistilBertConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_ids,
        attention_mask,
        deterministic: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any | tuple[()] | tuple[Any, ...], ...] | FlaxBaseModelOutput:
        ...

@add_start_docstrings(
    ...,
    FLAX_DISTILBERT_START_DOCSTRING,
)
class FlaxDistilBertModel(FlaxDistilBertPreTrainedModel):
    module_class = ...

class FlaxDistilBertForMaskedLMModule(nn.Module):
    config: DistilBertConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_ids,
        attention_mask,
        deterministic: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Array | Any, *tuple[Any | tuple[()] | tuple[Any, ...], ...]] | FlaxMaskedLMOutput:
        ...

@add_start_docstrings(
    ...,
    FLAX_DISTILBERT_START_DOCSTRING,
)
class FlaxDistilBertForMaskedLM(FlaxDistilBertPreTrainedModel):
    module_class = ...

class FlaxDistilBertForSequenceClassificationModule(nn.Module):
    config: DistilBertConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_ids,
        attention_mask,
        deterministic: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any, *tuple[Any | tuple[()] | tuple[Any, ...], ...]] | FlaxSequenceClassifierOutput:
        ...

@add_start_docstrings(
    ...,
    FLAX_DISTILBERT_START_DOCSTRING,
)
class FlaxDistilBertForSequenceClassification(FlaxDistilBertPreTrainedModel):
    module_class = ...

class FlaxDistilBertForMultipleChoiceModule(nn.Module):
    config: DistilBertConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_ids,
        attention_mask,
        deterministic: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any, *tuple[Any | tuple[()] | tuple[Any, ...], ...]] | FlaxMultipleChoiceModelOutput:
        ...

@add_start_docstrings(
    ...,
    FLAX_DISTILBERT_START_DOCSTRING,
)
class FlaxDistilBertForMultipleChoice(FlaxDistilBertPreTrainedModel):
    module_class = ...

class FlaxDistilBertForTokenClassificationModule(nn.Module):
    config: DistilBertConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_ids,
        attention_mask,
        deterministic: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any, *tuple[Any | tuple[()] | tuple[Any, ...], ...]] | FlaxTokenClassifierOutput:
        ...

@add_start_docstrings(
    ...,
    FLAX_DISTILBERT_START_DOCSTRING,
)
class FlaxDistilBertForTokenClassification(FlaxDistilBertPreTrainedModel):
    module_class = ...

class FlaxDistilBertForQuestionAnsweringModule(nn.Module):
    config: DistilBertConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_ids,
        attention_mask,
        deterministic: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Array, Array, *tuple[Any | tuple[()] | tuple[Any, ...], ...]] | FlaxQuestionAnsweringModelOutput:
        ...

@add_start_docstrings(
    ...,
    FLAX_DISTILBERT_START_DOCSTRING,
)
class FlaxDistilBertForQuestionAnswering(FlaxDistilBertPreTrainedModel):
    module_class = ...

__all__ = [
    "FlaxDistilBertForMaskedLM",
    "FlaxDistilBertForMultipleChoice",
    "FlaxDistilBertForQuestionAnswering",
    "FlaxDistilBertForSequenceClassification",
    "FlaxDistilBertForTokenClassification",
    "FlaxDistilBertModel",
    "FlaxDistilBertPreTrainedModel",
]
