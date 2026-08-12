from collections.abc import Callable

from flax.core.frozen_dict import FrozenDict

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from .configuration_albert import AlbertConfig
from ...modeling_flax_utils import FlaxPreTrainedModel
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)

logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...

@flax.struct.dataclass
class FlaxAlbertForPreTrainingOutput(ModelOutput):
    prediction_logits: jnp.ndarray = ...
    sop_logits: jnp.ndarray = ...
    hidden_states: tuple[jnp.ndarray] | None = ...
    attentions: tuple[jnp.ndarray] | None = ...

ALBERT_START_DOCSTRING = ...
ALBERT_INPUTS_DOCSTRING = ...

class FlaxAlbertEmbeddings(nn.Module):
    config: AlbertConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self, input_ids, token_type_ids, position_ids, deterministic: bool = ...
    ): ...

class FlaxAlbertSelfAttention(nn.Module):
    config: AlbertConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        hidden_states,
        attention_mask,
        deterministic=...,
        output_attentions: bool = ...,
    ):  # -> tuple[Any, Any] | tuple[Any]:
        ...

class FlaxAlbertLayer(nn.Module):
    config: AlbertConfig
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

class FlaxAlbertLayerCollection(nn.Module):
    config: AlbertConfig
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
    ):  # -> tuple[Any | tuple[()] | tuple[Any, ...] | tuple[Any], ...] | tuple[Any, ...] | tuple[Any, tuple[()] | tuple[Any, ...] | tuple[Any]] | tuple[Any]:
        ...

class FlaxAlbertLayerCollections(nn.Module):
    config: AlbertConfig
    dtype: jnp.dtype = ...
    layer_index: str | None = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        hidden_states,
        attention_mask,
        deterministic: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
    ):  # -> tuple[Any | tuple[()] | tuple[Any, ...] | tuple[Any], ...] | tuple[Any, ...] | tuple[Any, tuple[()] | tuple[Any, ...] | tuple[Any]] | tuple[Any]:
        ...

class FlaxAlbertLayerGroups(nn.Module):
    config: AlbertConfig
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
    ):  # -> tuple[Any | tuple[()] | tuple[Any, ...] | tuple[Any], ...] | FlaxBaseModelOutput:
        ...

class FlaxAlbertEncoder(nn.Module):
    config: AlbertConfig
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
    ):  # -> tuple[Any | tuple[()] | tuple[Any, ...] | tuple[Any], ...] | FlaxBaseModelOutput:
        ...

class FlaxAlbertOnlyMLMHead(nn.Module):
    config: AlbertConfig
    dtype: jnp.dtype = ...
    bias_init: Callable[..., np.ndarray] = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states, shared_embedding=...): ...

class FlaxAlbertSOPHead(nn.Module):
    config: AlbertConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, pooled_output, deterministic=...): ...

class FlaxAlbertPreTrainedModel(FlaxPreTrainedModel):
    config_class = AlbertConfig
    base_model_prefix = ...
    module_class: nn.Module = ...
    def __init__(
        self,
        config: AlbertConfig,
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
        ALBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    def __call__(
        self,
        input_ids,
        attention_mask=...,
        token_type_ids=...,
        position_ids=...,
        params: dict | None = ...,
        dropout_rng: jax.random.PRNGKey = ...,
        train: bool = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ): ...

class FlaxAlbertModule(nn.Module):
    config: AlbertConfig
    dtype: jnp.dtype = ...
    add_pooling_layer: bool = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids: np.ndarray | None = ...,
        position_ids: np.ndarray | None = ...,
        deterministic: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any | tuple[()] | tuple[Any, ...] | tuple[Any], *tuple[Any | tuple[()] | tuple[Any, ...] | tuple[Any], ...]] | tuple[Any | tuple[()] | tuple[Any, ...] | tuple[Any], Any, *tuple[Any | tuple[()] | tuple[Any, ...] | tuple[Any], ...]] | FlaxBaseModelOutputWithPooling:
        ...

@add_start_docstrings(
    ...,
    ALBERT_START_DOCSTRING,
)
class FlaxAlbertModel(FlaxAlbertPreTrainedModel): ...

class FlaxAlbertForPreTrainingModule(nn.Module):
    config: AlbertConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        deterministic: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any, Any, *tuple[Any | tuple[()] | tuple[Any, ...] | tuple[Any], ...]] | FlaxAlbertForPreTrainingOutput:
        ...

@add_start_docstrings(
    ...,
    ALBERT_START_DOCSTRING,
)
class FlaxAlbertForPreTraining(FlaxAlbertPreTrainedModel): ...

FLAX_ALBERT_FOR_PRETRAINING_DOCSTRING = ...

class FlaxAlbertForMaskedLMModule(nn.Module):
    config: AlbertConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        deterministic: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any, *tuple[Any | tuple[()] | tuple[Any, ...] | tuple[Any], ...]] | tuple[Any, Any, *tuple[Any | tuple[()] | tuple[Any, ...] | tuple[Any], ...]] | FlaxMaskedLMOutput:
        ...

@add_start_docstrings(..., ALBERT_START_DOCSTRING)
class FlaxAlbertForMaskedLM(FlaxAlbertPreTrainedModel): ...

class FlaxAlbertForSequenceClassificationModule(nn.Module):
    config: AlbertConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        deterministic: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any, *tuple[Any | tuple[()] | tuple[Any, ...] | tuple[Any], ...]] | FlaxSequenceClassifierOutput:
        ...

@add_start_docstrings(
    ...,
    ALBERT_START_DOCSTRING,
)
class FlaxAlbertForSequenceClassification(FlaxAlbertPreTrainedModel): ...

class FlaxAlbertForMultipleChoiceModule(nn.Module):
    config: AlbertConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        deterministic: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any, *tuple[Any | tuple[()] | tuple[Any, ...] | tuple[Any], ...]] | FlaxMultipleChoiceModelOutput:
        ...

@add_start_docstrings(
    ...,
    ALBERT_START_DOCSTRING,
)
class FlaxAlbertForMultipleChoice(FlaxAlbertPreTrainedModel): ...

class FlaxAlbertForTokenClassificationModule(nn.Module):
    config: AlbertConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        deterministic: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any, *tuple[Any | tuple[()] | tuple[Any, ...] | tuple[Any], ...]] | tuple[Any, Any, *tuple[Any | tuple[()] | tuple[Any, ...] | tuple[Any], ...]] | FlaxTokenClassifierOutput:
        ...

@add_start_docstrings(
    ...,
    ALBERT_START_DOCSTRING,
)
class FlaxAlbertForTokenClassification(FlaxAlbertPreTrainedModel): ...

class FlaxAlbertForQuestionAnsweringModule(nn.Module):
    config: AlbertConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        deterministic: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Array, Array, *tuple[Any | tuple[()] | tuple[Any, ...] | tuple[Any], ...]] | tuple[Array, Array, Any, *tuple[Any | tuple[()] | tuple[Any, ...] | tuple[Any], ...]] | FlaxQuestionAnsweringModelOutput:
        ...

@add_start_docstrings(
    ...,
    ALBERT_START_DOCSTRING,
)
class FlaxAlbertForQuestionAnswering(FlaxAlbertPreTrainedModel): ...

__all__ = [
    "FlaxAlbertForMaskedLM",
    "FlaxAlbertForMultipleChoice",
    "FlaxAlbertForPreTraining",
    "FlaxAlbertForQuestionAnswering",
    "FlaxAlbertForSequenceClassification",
    "FlaxAlbertForTokenClassification",
    "FlaxAlbertModel",
    "FlaxAlbertPreTrainedModel",
]
