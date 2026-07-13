from collections.abc import Callable

from flax.core.frozen_dict import FrozenDict

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from .configuration_roberta import RobertaConfig
from ...modeling_flax_utils import FlaxPreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward

logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
remat = ...

def create_position_ids_from_input_ids(input_ids, padding_idx): ...

ROBERTA_START_DOCSTRING = ...
ROBERTA_INPUTS_DOCSTRING = ...

class FlaxRobertaEmbeddings(nn.Module):
    config: RobertaConfig
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

class FlaxRobertaSelfAttention(nn.Module):
    config: RobertaConfig
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

class FlaxRobertaSelfOutput(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states, input_tensor, deterministic: bool = ...): ...

class FlaxRobertaAttention(nn.Module):
    config: RobertaConfig
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

class FlaxRobertaIntermediate(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states): ...

class FlaxRobertaOutput(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states, attention_output, deterministic: bool = ...): ...

class FlaxRobertaLayer(nn.Module):
    config: RobertaConfig
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

class FlaxRobertaLayerCollection(nn.Module):
    config: RobertaConfig
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

class FlaxRobertaEncoder(nn.Module):
    config: RobertaConfig
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

class FlaxRobertaPooler(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states): ...

class FlaxRobertaLMHead(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = ...
    bias_init: Callable[..., np.ndarray] = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states, shared_embedding=...): ...

class FlaxRobertaClassificationHead(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states, deterministic=...): ...

class FlaxRobertaPreTrainedModel(FlaxPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = ...
    module_class: nn.Module = ...
    def __init__(
        self,
        config: RobertaConfig,
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
        ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length")
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

class FlaxRobertaModule(nn.Module):
    config: RobertaConfig
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
    ROBERTA_START_DOCSTRING,
)
class FlaxRobertaModel(FlaxRobertaPreTrainedModel):
    module_class = ...

class FlaxRobertaForMaskedLMModule(nn.Module):
    config: RobertaConfig
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

@add_start_docstrings(..., ROBERTA_START_DOCSTRING)
class FlaxRobertaForMaskedLM(FlaxRobertaPreTrainedModel):
    module_class = ...

class FlaxRobertaForSequenceClassificationModule(nn.Module):
    config: RobertaConfig
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
    ROBERTA_START_DOCSTRING,
)
class FlaxRobertaForSequenceClassification(FlaxRobertaPreTrainedModel):
    module_class = ...

class FlaxRobertaForMultipleChoiceModule(nn.Module):
    config: RobertaConfig
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
    ROBERTA_START_DOCSTRING,
)
class FlaxRobertaForMultipleChoice(FlaxRobertaPreTrainedModel):
    module_class = ...

class FlaxRobertaForTokenClassificationModule(nn.Module):
    config: RobertaConfig
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
    ROBERTA_START_DOCSTRING,
)
class FlaxRobertaForTokenClassification(FlaxRobertaPreTrainedModel):
    module_class = ...

class FlaxRobertaForQuestionAnsweringModule(nn.Module):
    config: RobertaConfig
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
    ROBERTA_START_DOCSTRING,
)
class FlaxRobertaForQuestionAnswering(FlaxRobertaPreTrainedModel):
    module_class = ...

class FlaxRobertaForCausalLMModule(nn.Module):
    config: RobertaConfig
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
    ROBERTA_START_DOCSTRING,
)
class FlaxRobertaForCausalLM(FlaxRobertaPreTrainedModel):
    module_class = ...
    def prepare_inputs_for_generation(
        self, input_ids, max_length, attention_mask: jax.Array | None = ...
    ):  # -> dict[str, Any | Array]:
        ...
    def update_inputs_for_generation(self, model_outputs, model_kwargs): ...

__all__ = [
    "FlaxRobertaForCausalLM",
    "FlaxRobertaForMaskedLM",
    "FlaxRobertaForMultipleChoice",
    "FlaxRobertaForQuestionAnswering",
    "FlaxRobertaForSequenceClassification",
    "FlaxRobertaForTokenClassification",
    "FlaxRobertaModel",
    "FlaxRobertaPreTrainedModel",
]
