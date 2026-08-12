from collections.abc import Callable

from flax.core.frozen_dict import FrozenDict

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

from .configuration_big_bird import BigBirdConfig
from ...modeling_flax_utils import FlaxPreTrainedModel
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)

logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
remat = ...

@flax.struct.dataclass
class FlaxBigBirdForPreTrainingOutput(ModelOutput):
    prediction_logits: jnp.ndarray = ...
    seq_relationship_logits: jnp.ndarray = ...
    hidden_states: tuple[jnp.ndarray] | None = ...
    attentions: tuple[jnp.ndarray] | None = ...

@flax.struct.dataclass
class FlaxBigBirdForQuestionAnsweringModelOutput(ModelOutput):
    start_logits: jnp.ndarray = ...
    end_logits: jnp.ndarray = ...
    pooled_output: jnp.ndarray = ...
    hidden_states: tuple[jnp.ndarray] | None = ...
    attentions: tuple[jnp.ndarray] | None = ...

BIG_BIRD_START_DOCSTRING = ...
BIG_BIRD_INPUTS_DOCSTRING = ...

class FlaxBigBirdEmbeddings(nn.Module):
    config: BigBirdConfig
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

class FlaxBigBirdSelfAttention(nn.Module):
    config: BigBirdConfig
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

class FlaxBigBirdBlockSparseAttention(nn.Module):
    config: BigBirdConfig
    block_sparse_seed: int | None = ...
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    @staticmethod
    def transpose_for_scores(x, n_heads, head_size):  # -> Array:
        ...
    def __call__(
        self, hidden_states, attention_mask, deterministic=..., output_attentions=...
    ):  # -> tuple[Array, None] | tuple[Array]:
        ...
    @staticmethod
    def create_masks_for_block_sparse_attn(
        attention_mask, block_size: int
    ):  # -> tuple[Any, Array, Any, Any]:
        ...
    def bigbird_block_sparse_attention(
        self,
        query_layer,
        key_layer,
        value_layer,
        band_mask,
        from_mask,
        to_mask,
        from_blocked_mask,
        to_blocked_mask,
        n_heads,
        head_size,
        indices_prng_key: jax.random.PRNGKey | None = ...,
        deterministic: bool | None = ...,
        plan_from_length=...,
        plan_num_rand_blocks=...,
        output_attentions=...,
    ):  # -> tuple[Array, None]:
        ...
    @staticmethod
    def jax_gather(params, indices, batch_dims=...): ...

class FlaxBigBirdSelfOutput(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states, input_tensor, deterministic: bool = ...): ...

class FlaxBigBirdAttention(nn.Module):
    config: BigBirdConfig
    layer_id: int | None = ...
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
    ):  # -> tuple[Any, Any | Array | None] | tuple[Any]:
        ...

class FlaxBigBirdIntermediate(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states): ...

class FlaxBigBirdOutput(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states, attention_output, deterministic: bool = ...): ...

class FlaxBigBirdLayer(nn.Module):
    config: BigBirdConfig
    layer_id: int | None = ...
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
    ):  # -> tuple[Any, Any | Array | None, Any | Array | None] | tuple[Any, Any | Array | None] | tuple[Any]:
        ...

class FlaxBigBirdLayerCollection(nn.Module):
    config: BigBirdConfig
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
    ):  # -> tuple[Any | tuple[Any, ...] | tuple[()] | tuple[Any | Array | None, ...], ...] | FlaxBaseModelOutputWithPastAndCrossAttentions:
        ...

class FlaxBigBirdEncoder(nn.Module):
    config: BigBirdConfig
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
    ):  # -> tuple[Any | tuple[Any, ...] | tuple[()] | tuple[Any | Array | None, ...], ...] | FlaxBaseModelOutputWithPastAndCrossAttentions:
        ...

class FlaxBigBirdPredictionHeadTransform(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states): ...

class FlaxBigBirdLMPredictionHead(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = ...
    bias_init: Callable[..., jnp.ndarray] = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states, shared_embedding=...): ...

class FlaxBigBirdOnlyMLMHead(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states, shared_embedding=...): ...

class FlaxBigBirdPreTrainingHeads(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self, hidden_states, pooled_output, shared_embedding=...
    ):  # -> tuple[Any, Any]:
        ...

class FlaxBigBirdPreTrainedModel(FlaxPreTrainedModel):
    config_class = BigBirdConfig
    base_model_prefix = ...
    module_class: nn.Module = ...
    def __init__(
        self,
        config: BigBirdConfig,
        input_shape: tuple | None = ...,
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
        BIG_BIRD_INPUTS_DOCSTRING.format("batch_size, sequence_length")
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
        dropout_rng: jax.random.PRNGKey | None = ...,
        indices_rng: jax.random.PRNGKey | None = ...,
        train: bool = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        past_key_values: dict | None = ...,
    ): ...

class FlaxBigBirdModule(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = ...
    add_pooling_layer: bool = ...
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
        encoder_hidden_states: jnp.ndarray | None = ...,
        encoder_attention_mask: jnp.ndarray | None = ...,
        init_cache: bool = ...,
        deterministic: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any | tuple[Any, ...] | tuple[()] | tuple[Any | Array | None, ...], *tuple[Any | tuple[Any, ...] | tuple[()] | tuple[Any | Array | None, ...], ...]] | tuple[Any | tuple[Any, ...] | tuple[()] | tuple[Any | Array | None, ...], Any, *tuple[Any | tuple[Any, ...] | tuple[()] | tuple[Any | Array | None, ...], ...]] | FlaxBaseModelOutputWithPoolingAndCrossAttentions:
        ...

@add_start_docstrings(
    ...,
    BIG_BIRD_START_DOCSTRING,
)
class FlaxBigBirdModel(FlaxBigBirdPreTrainedModel): ...

class FlaxBigBirdForPreTrainingModule(nn.Module):
    config: BigBirdConfig
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
    ):  # -> tuple[Any, Any, *tuple[Any | tuple[Any, ...] | tuple[()] | tuple[Any | Array | None, ...], ...]] | FlaxBigBirdForPreTrainingOutput:
        ...

@add_start_docstrings(
    ...,
    BIG_BIRD_START_DOCSTRING,
)
class FlaxBigBirdForPreTraining(FlaxBigBirdPreTrainedModel): ...

FLAX_BIG_BIRD_FOR_PRETRAINING_DOCSTRING = ...

class FlaxBigBirdForMaskedLMModule(nn.Module):
    config: BigBirdConfig
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
    ):  # -> tuple[Any, *tuple[Any | tuple[Any, ...] | tuple[()] | tuple[Any | Array | None, ...], ...]] | tuple[Any, Any, *tuple[Any | tuple[Any, ...] | tuple[()] | tuple[Any | Array | None, ...], ...]] | FlaxMaskedLMOutput:
        ...

@add_start_docstrings(
    ...,
    BIG_BIRD_START_DOCSTRING,
)
class FlaxBigBirdForMaskedLM(FlaxBigBirdPreTrainedModel): ...

class FlaxBigBirdClassificationHead(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, features, deterministic=...): ...

class FlaxBigBirdForSequenceClassificationModule(nn.Module):
    config: BigBirdConfig
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
    ):  # -> tuple[Any, *tuple[Any | tuple[Any, ...] | tuple[()] | tuple[Any | Array | None, ...], ...]] | FlaxSequenceClassifierOutput:
        ...

@add_start_docstrings(
    ...,
    BIG_BIRD_START_DOCSTRING,
)
class FlaxBigBirdForSequenceClassification(FlaxBigBirdPreTrainedModel): ...

class FlaxBigBirdForMultipleChoiceModule(nn.Module):
    config: BigBirdConfig
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
    ):  # -> tuple[Any, *tuple[Any | tuple[Any, ...] | tuple[()] | tuple[Any | Array | None, ...], ...]] | FlaxMultipleChoiceModelOutput:
        ...

@add_start_docstrings(
    ...,
    BIG_BIRD_START_DOCSTRING,
)
class FlaxBigBirdForMultipleChoice(FlaxBigBirdPreTrainedModel):
    def __init__(
        self,
        config: BigBirdConfig,
        input_shape: tuple | None = ...,
        seed: int = ...,
        dtype: jnp.dtype = ...,
        _do_init: bool = ...,
        **kwargs,
    ) -> None: ...

class FlaxBigBirdForTokenClassificationModule(nn.Module):
    config: BigBirdConfig
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
    ):  # -> tuple[Any, *tuple[Any | tuple[Any, ...] | tuple[()] | tuple[Any | Array | None, ...], ...]] | tuple[Any, Any, *tuple[Any | tuple[Any, ...] | tuple[()] | tuple[Any | Array | None, ...], ...]] | FlaxTokenClassifierOutput:
        ...

@add_start_docstrings(
    ...,
    BIG_BIRD_START_DOCSTRING,
)
class FlaxBigBirdForTokenClassification(FlaxBigBirdPreTrainedModel): ...

class FlaxBigBirdForQuestionAnsweringHead(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, encoder_output, deterministic=...): ...

class FlaxBigBirdForQuestionAnsweringModule(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = ...
    add_pooling_layer: bool = ...
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
        logits_mask=...,
        deterministic: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Array, Array, *tuple[Any | tuple[Any, ...] | tuple[()] | tuple[Any | Array | None, ...], ...]] | tuple[Array, Array, Any, *tuple[Any | tuple[Any, ...] | tuple[()] | tuple[Any | Array | None, ...], ...]] | FlaxBigBirdForQuestionAnsweringModelOutput:
        ...

@add_start_docstrings(
    ...,
    BIG_BIRD_START_DOCSTRING,
)
class FlaxBigBirdForQuestionAnswering(FlaxBigBirdPreTrainedModel):
    @add_start_docstrings_to_model_forward(
        BIG_BIRD_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    def __call__(
        self,
        input_ids,
        attention_mask=...,
        token_type_ids=...,
        position_ids=...,
        head_mask=...,
        question_lengths=...,
        params: dict | None = ...,
        dropout_rng: jax.random.PRNGKey | None = ...,
        indices_rng: jax.random.PRNGKey | None = ...,
        train: bool = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ): ...
    @staticmethod
    def prepare_question_mask(q_lengths, maxlen: int): ...

class FlaxBigBirdForCausalLMModule(nn.Module):
    config: BigBirdConfig
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
    ):  # -> tuple[Any, *tuple[Any | tuple[Any, ...] | tuple[()] | tuple[Any | Array | None, ...], ...]] | tuple[Any, Any, *tuple[Any | tuple[Any, ...] | tuple[()] | tuple[Any | Array | None, ...], ...]] | FlaxCausalLMOutputWithCrossAttentions:
        ...

@add_start_docstrings(
    ...,
    BIG_BIRD_START_DOCSTRING,
)
class FlaxBigBirdForCausalLM(FlaxBigBirdPreTrainedModel):
    def prepare_inputs_for_generation(
        self, input_ids, max_length, attention_mask: jax.Array | None = ...
    ):  # -> dict[str, Any | Array]:
        ...
    def update_inputs_for_generation(self, model_outputs, model_kwargs): ...

__all__ = [
    "FlaxBigBirdForCausalLM",
    "FlaxBigBirdForMaskedLM",
    "FlaxBigBirdForMultipleChoice",
    "FlaxBigBirdForPreTraining",
    "FlaxBigBirdForQuestionAnswering",
    "FlaxBigBirdForSequenceClassification",
    "FlaxBigBirdForTokenClassification",
    "FlaxBigBirdModel",
    "FlaxBigBirdPreTrainedModel",
]
