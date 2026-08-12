from collections.abc import Callable

from flax.core.frozen_dict import FrozenDict

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from .configuration_xlm_roberta import XLMRobertaConfig
from ...modeling_flax_utils import FlaxPreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward

"""Flax XLM-RoBERTa model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
remat = ...

def create_position_ids_from_input_ids(input_ids, padding_idx): ...

XLM_ROBERTA_START_DOCSTRING = ...
XLM_ROBERTA_INPUTS_DOCSTRING = ...

class FlaxXLMRobertaEmbeddings(nn.Module):
    config: XLMRobertaConfig
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

class FlaxXLMRobertaSelfAttention(nn.Module):
    config: XLMRobertaConfig
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

class FlaxXLMRobertaSelfOutput(nn.Module):
    config: XLMRobertaConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states, input_tensor, deterministic: bool = ...): ...

class FlaxXLMRobertaAttention(nn.Module):
    config: XLMRobertaConfig
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

class FlaxXLMRobertaIntermediate(nn.Module):
    config: XLMRobertaConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states): ...

class FlaxXLMRobertaOutput(nn.Module):
    config: XLMRobertaConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states, attention_output, deterministic: bool = ...): ...

class FlaxXLMRobertaLayer(nn.Module):
    config: XLMRobertaConfig
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

class FlaxXLMRobertaLayerCollection(nn.Module):
    config: XLMRobertaConfig
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

class FlaxXLMRobertaEncoder(nn.Module):
    config: XLMRobertaConfig
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

class FlaxXLMRobertaPooler(nn.Module):
    config: XLMRobertaConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states): ...

class FlaxXLMRobertaLMHead(nn.Module):
    config: XLMRobertaConfig
    dtype: jnp.dtype = ...
    bias_init: Callable[..., np.ndarray] = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states, shared_embedding=...): ...

class FlaxXLMRobertaClassificationHead(nn.Module):
    config: XLMRobertaConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states, deterministic=...): ...

class FlaxXLMRobertaPreTrainedModel(FlaxPreTrainedModel):
    config_class = XLMRobertaConfig
    base_model_prefix = ...
    module_class: nn.Module = ...
    def __init__(
        self,
        config: XLMRobertaConfig,
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
        XLM_ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length")
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

class FlaxXLMRobertaModule(nn.Module):
    config: XLMRobertaConfig
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
    ):  # -> tuple[Any | tuple[Any, ...] | tuple[()] | tuple[Array | Any, ...], *tuple[Any | tuple[Any, ...] | tuple[()] | tuple[Array | Any, ...], ...]] | tuple[Any | tuple[Any, ...] | tuple[()] | tuple[Array | Any, ...], Any, *tuple[Any | tuple[Any, ...] | tuple[()] | tuple[Array | Any, ...], ...]] | FlaxBaseModelOutputWithPoolingAndCrossAttentions:
        ...

@add_start_docstrings(
    ...,
    XLM_ROBERTA_START_DOCSTRING,
)
class FlaxXLMRobertaModel(FlaxXLMRobertaPreTrainedModel): ...

class FlaxXLMRobertaForMaskedLMModule(nn.Module):
    config: XLMRobertaConfig
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
    XLM_ROBERTA_START_DOCSTRING,
)
class FlaxXLMRobertaForMaskedLM(FlaxXLMRobertaPreTrainedModel): ...

class FlaxXLMRobertaForSequenceClassificationModule(nn.Module):
    config: XLMRobertaConfig
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
    XLM_ROBERTA_START_DOCSTRING,
)
class FlaxXLMRobertaForSequenceClassification(FlaxXLMRobertaPreTrainedModel): ...

class FlaxXLMRobertaForMultipleChoiceModule(nn.Module):
    config: XLMRobertaConfig
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
    XLM_ROBERTA_START_DOCSTRING,
)
class FlaxXLMRobertaForMultipleChoice(FlaxXLMRobertaPreTrainedModel): ...

class FlaxXLMRobertaForTokenClassificationModule(nn.Module):
    config: XLMRobertaConfig
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
    XLM_ROBERTA_START_DOCSTRING,
)
class FlaxXLMRobertaForTokenClassification(FlaxXLMRobertaPreTrainedModel): ...

class FlaxXLMRobertaForQuestionAnsweringModule(nn.Module):
    config: XLMRobertaConfig
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
    XLM_ROBERTA_START_DOCSTRING,
)
class FlaxXLMRobertaForQuestionAnswering(FlaxXLMRobertaPreTrainedModel): ...

class FlaxXLMRobertaForCausalLMModule(nn.Module):
    config: XLMRobertaConfig
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
    XLM_ROBERTA_START_DOCSTRING,
)
class FlaxXLMRobertaForCausalLM(FlaxXLMRobertaPreTrainedModel):
    def prepare_inputs_for_generation(
        self, input_ids, max_length, attention_mask: jax.Array | None = ...
    ):  # -> dict[str, Any | Array]:
        ...
    def update_inputs_for_generation(self, model_outputs, model_kwargs): ...

__all__ = [
    "FlaxXLMRobertaForCausalLM",
    "FlaxXLMRobertaForMaskedLM",
    "FlaxXLMRobertaForMultipleChoice",
    "FlaxXLMRobertaForQuestionAnswering",
    "FlaxXLMRobertaForSequenceClassification",
    "FlaxXLMRobertaForTokenClassification",
    "FlaxXLMRobertaModel",
    "FlaxXLMRobertaPreTrainedModel",
]
