from collections.abc import Callable

from flax.core.frozen_dict import FrozenDict

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from .configuration_roberta_prelayernorm import RobertaPreLayerNormConfig
from ...modeling_flax_utils import FlaxPreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward

"""Flax RoBERTa-PreLayerNorm model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
remat = ...

def create_position_ids_from_input_ids(input_ids, padding_idx): ...

ROBERTA_PRELAYERNORM_START_DOCSTRING = ...
ROBERTA_PRELAYERNORM_INPUTS_DOCSTRING = ...

class FlaxRobertaPreLayerNormEmbeddings(nn.Module):
    config: RobertaPreLayerNormConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_ids,
        token_type_ids,
        position_ids,
        attention_mask,
        deterministic: bool = ...,
    ): ...

class FlaxRobertaPreLayerNormSelfAttention(nn.Module):
    config: RobertaPreLayerNormConfig
    causal: bool = ...
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        hidden_states,
        attention_mask,
        layer_head_mask,
        key_value_states: jnp.ndarray | None = ...,
        init_cache: bool = ...,
        deterministic=...,
        output_attentions: bool = ...,
    ):  # -> tuple[Array, Array | Any] | tuple[Array]:
        ...

class FlaxRobertaPreLayerNormSelfOutput(nn.Module):
    config: RobertaPreLayerNormConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states, input_tensor, deterministic: bool = ...): ...

class FlaxRobertaPreLayerNormAttention(nn.Module):
    config: RobertaPreLayerNormConfig
    causal: bool = ...
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        hidden_states,
        attention_mask,
        layer_head_mask,
        key_value_states=...,
        init_cache=...,
        deterministic=...,
        output_attentions: bool = ...,
    ):  # -> tuple[Any, Array | Any] | tuple[Any]:
        ...

class FlaxRobertaPreLayerNormIntermediate(nn.Module):
    config: RobertaPreLayerNormConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states): ...

class FlaxRobertaPreLayerNormOutput(nn.Module):
    config: RobertaPreLayerNormConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states, attention_output, deterministic: bool = ...): ...

class FlaxRobertaPreLayerNormLayer(nn.Module):
    config: RobertaPreLayerNormConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        hidden_states,
        attention_mask,
        layer_head_mask,
        encoder_hidden_states: jnp.ndarray | None = ...,
        encoder_attention_mask: jnp.ndarray | None = ...,
        init_cache: bool = ...,
        deterministic: bool = ...,
        output_attentions: bool = ...,
    ):  # -> tuple[Any, Array | Any, Array | Any] | tuple[Any, Array | Any] | tuple[Any]:
        ...

class FlaxRobertaPreLayerNormLayerCollection(nn.Module):
    config: RobertaPreLayerNormConfig
    dtype: jnp.dtype = ...
    gradient_checkpointing: bool = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        hidden_states,
        attention_mask,
        head_mask,
        encoder_hidden_states: jnp.ndarray | None = ...,
        encoder_attention_mask: jnp.ndarray | None = ...,
        init_cache: bool = ...,
        deterministic: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any | tuple[Any, ...] | tuple[()] | tuple[Array | Any, ...], ...] | FlaxBaseModelOutputWithPastAndCrossAttentions:
        ...

class FlaxRobertaPreLayerNormEncoder(nn.Module):
    config: RobertaPreLayerNormConfig
    dtype: jnp.dtype = ...
    gradient_checkpointing: bool = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        hidden_states,
        attention_mask,
        head_mask,
        encoder_hidden_states: jnp.ndarray | None = ...,
        encoder_attention_mask: jnp.ndarray | None = ...,
        init_cache: bool = ...,
        deterministic: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any | tuple[Any, ...] | tuple[()] | tuple[Array | Any, ...], ...] | FlaxBaseModelOutputWithPastAndCrossAttentions:
        ...

class FlaxRobertaPreLayerNormPooler(nn.Module):
    config: RobertaPreLayerNormConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states): ...

class FlaxRobertaPreLayerNormLMHead(nn.Module):
    config: RobertaPreLayerNormConfig
    dtype: jnp.dtype = ...
    bias_init: Callable[..., np.ndarray] = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states, shared_embedding=...): ...

class FlaxRobertaPreLayerNormClassificationHead(nn.Module):
    config: RobertaPreLayerNormConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states, deterministic=...): ...

class FlaxRobertaPreLayerNormPreTrainedModel(FlaxPreTrainedModel):
    config_class = RobertaPreLayerNormConfig
    base_model_prefix = ...
    module_class: nn.Module = ...
    def __init__(
        self,
        config: RobertaPreLayerNormConfig,
        input_shape: tuple = ...,
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
    def init_cache(self, batch_size, max_length): ...
    @add_start_docstrings_to_model_forward(
        ROBERTA_PRELAYERNORM_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    def __call__(
        self,
        input_ids,
        attention_mask=...,
        token_type_ids=...,
        position_ids=...,
        head_mask=...,
        encoder_hidden_states=...,
        encoder_attention_mask=...,
        params: dict | None = ...,
        dropout_rng: jax.random.PRNGKey = ...,
        train: bool = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        past_key_values: dict | None = ...,
    ): ...

class FlaxRobertaPreLayerNormModule(nn.Module):
    config: RobertaPreLayerNormConfig
    dtype: jnp.dtype = ...
    add_pooling_layer: bool = ...
    gradient_checkpointing: bool = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids: jnp.ndarray | None = ...,
        position_ids: jnp.ndarray | None = ...,
        head_mask: jnp.ndarray | None = ...,
        encoder_hidden_states: jnp.ndarray | None = ...,
        encoder_attention_mask: jnp.ndarray | None = ...,
        init_cache: bool = ...,
        deterministic: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any, *tuple[Any | tuple[Any, ...] | tuple[()] | tuple[Array | Any, ...], ...]] | tuple[Any, Any, *tuple[Any | tuple[Any, ...] | tuple[()] | tuple[Array | Any, ...], ...]] | FlaxBaseModelOutputWithPoolingAndCrossAttentions:
        ...

@add_start_docstrings(
    ...,
    ROBERTA_PRELAYERNORM_START_DOCSTRING,
)
class FlaxRobertaPreLayerNormModel(FlaxRobertaPreLayerNormPreTrainedModel): ...

class FlaxRobertaPreLayerNormForMaskedLMModule(nn.Module):
    config: RobertaPreLayerNormConfig
    dtype: jnp.dtype = ...
    gradient_checkpointing: bool = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        deterministic: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any, *tuple[Any | tuple[Any, ...] | tuple[()] | tuple[Array | Any, ...], ...]] | tuple[Any, Any, *tuple[Any | tuple[Any, ...] | tuple[()] | tuple[Array | Any, ...], ...]] | FlaxMaskedLMOutput:
        ...

@add_start_docstrings(
    ...,
    ROBERTA_PRELAYERNORM_START_DOCSTRING,
)
class FlaxRobertaPreLayerNormForMaskedLM(FlaxRobertaPreLayerNormPreTrainedModel): ...

class FlaxRobertaPreLayerNormForSequenceClassificationModule(nn.Module):
    config: RobertaPreLayerNormConfig
    dtype: jnp.dtype = ...
    gradient_checkpointing: bool = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        deterministic: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any, *tuple[Any | tuple[Any, ...] | tuple[()] | tuple[Array | Any, ...], ...]] | tuple[Any, Any, *tuple[Any | tuple[Any, ...] | tuple[()] | tuple[Array | Any, ...], ...]] | FlaxSequenceClassifierOutput:
        ...

@add_start_docstrings(
    ...,
    ROBERTA_PRELAYERNORM_START_DOCSTRING,
)
class FlaxRobertaPreLayerNormForSequenceClassification(
    FlaxRobertaPreLayerNormPreTrainedModel
): ...

class FlaxRobertaPreLayerNormForMultipleChoiceModule(nn.Module):
    config: RobertaPreLayerNormConfig
    dtype: jnp.dtype = ...
    gradient_checkpointing: bool = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        deterministic: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any, *tuple[Any | tuple[Any, ...] | tuple[()] | tuple[Array | Any, ...], ...]] | FlaxMultipleChoiceModelOutput:
        ...

@add_start_docstrings(
    ...,
    ROBERTA_PRELAYERNORM_START_DOCSTRING,
)
class FlaxRobertaPreLayerNormForMultipleChoice(
    FlaxRobertaPreLayerNormPreTrainedModel
): ...

class FlaxRobertaPreLayerNormForTokenClassificationModule(nn.Module):
    config: RobertaPreLayerNormConfig
    dtype: jnp.dtype = ...
    gradient_checkpointing: bool = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        deterministic: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any, *tuple[Any | tuple[Any, ...] | tuple[()] | tuple[Array | Any, ...], ...]] | tuple[Any, Any, *tuple[Any | tuple[Any, ...] | tuple[()] | tuple[Array | Any, ...], ...]] | FlaxTokenClassifierOutput:
        ...

@add_start_docstrings(
    ...,
    ROBERTA_PRELAYERNORM_START_DOCSTRING,
)
class FlaxRobertaPreLayerNormForTokenClassification(
    FlaxRobertaPreLayerNormPreTrainedModel
): ...

class FlaxRobertaPreLayerNormForQuestionAnsweringModule(nn.Module):
    config: RobertaPreLayerNormConfig
    dtype: jnp.dtype = ...
    gradient_checkpointing: bool = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        deterministic: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Array, Array, *tuple[Any | tuple[Any, ...] | tuple[()] | tuple[Array | Any, ...], ...]] | tuple[Array, Array, Any, *tuple[Any | tuple[Any, ...] | tuple[()] | tuple[Array | Any, ...], ...]] | FlaxQuestionAnsweringModelOutput:
        ...

@add_start_docstrings(
    ...,
    ROBERTA_PRELAYERNORM_START_DOCSTRING,
)
class FlaxRobertaPreLayerNormForQuestionAnswering(
    FlaxRobertaPreLayerNormPreTrainedModel
): ...

class FlaxRobertaPreLayerNormForCausalLMModule(nn.Module):
    config: RobertaPreLayerNormConfig
    dtype: jnp.dtype = ...
    gradient_checkpointing: bool = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        token_type_ids: jnp.ndarray | None = ...,
        head_mask: jnp.ndarray | None = ...,
        encoder_hidden_states: jnp.ndarray | None = ...,
        encoder_attention_mask: jnp.ndarray | None = ...,
        init_cache: bool = ...,
        deterministic: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any, *tuple[Any | tuple[Any, ...] | tuple[()] | tuple[Array | Any, ...], ...]] | tuple[Any, Any, *tuple[Any | tuple[Any, ...] | tuple[()] | tuple[Array | Any, ...], ...]] | FlaxCausalLMOutputWithCrossAttentions:
        ...

@add_start_docstrings(
    ...,
    ROBERTA_PRELAYERNORM_START_DOCSTRING,
)
class FlaxRobertaPreLayerNormForCausalLM(FlaxRobertaPreLayerNormPreTrainedModel):
    def prepare_inputs_for_generation(
        self, input_ids, max_length, attention_mask: jax.Array | None = ...
    ):  # -> dict[str, Any | Array]:
        ...
    def update_inputs_for_generation(self, model_outputs, model_kwargs): ...

__all__ = [
    "FlaxRobertaPreLayerNormForCausalLM",
    "FlaxRobertaPreLayerNormForMaskedLM",
    "FlaxRobertaPreLayerNormForMultipleChoice",
    "FlaxRobertaPreLayerNormForQuestionAnswering",
    "FlaxRobertaPreLayerNormForSequenceClassification",
    "FlaxRobertaPreLayerNormForTokenClassification",
    "FlaxRobertaPreLayerNormModel",
    "FlaxRobertaPreLayerNormPreTrainedModel",
]
