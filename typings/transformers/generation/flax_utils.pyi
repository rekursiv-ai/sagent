import flax
import jax.numpy as jnp

from .configuration_utils import GenerationConfig
from .flax_logits_process import FlaxLogitsProcessorList
from ..utils import ModelOutput

logger = ...

@flax.struct.dataclass
class FlaxGreedySearchOutput(ModelOutput):
    sequences: jnp.ndarray | None = ...

@flax.struct.dataclass
class FlaxSampleOutput(ModelOutput):
    sequences: jnp.ndarray | None = ...

@flax.struct.dataclass
class FlaxBeamSearchOutput(ModelOutput):
    sequences: jnp.ndarray | None = ...
    scores: jnp.ndarray | None = ...

@flax.struct.dataclass
class GreedyState:
    cur_len: jnp.ndarray
    sequences: jnp.ndarray
    running_token: jnp.ndarray
    is_sent_finished: jnp.ndarray
    model_kwargs: dict[str, jnp.ndarray]

@flax.struct.dataclass
class SampleState:
    cur_len: jnp.ndarray
    sequences: jnp.ndarray
    running_token: jnp.ndarray
    is_sent_finished: jnp.ndarray
    prng_key: jnp.ndarray
    model_kwargs: dict[str, jnp.ndarray]

@flax.struct.dataclass
class BeamSearchState:
    cur_len: jnp.ndarray
    running_sequences: jnp.ndarray
    running_scores: jnp.ndarray
    sequences: jnp.ndarray
    scores: jnp.ndarray
    is_sent_finished: jnp.ndarray
    model_kwargs: dict[str, jnp.ndarray]

class FlaxGenerationMixin:
    def prepare_inputs_for_generation(self, *args, **kwargs): ...
    def generate(
        self,
        input_ids: jnp.ndarray,
        generation_config: GenerationConfig | None = ...,
        prng_key: jnp.ndarray | None = ...,
        trace: bool = ...,
        params: dict[str, jnp.ndarray] | None = ...,
        logits_processor: FlaxLogitsProcessorList | None = ...,
        **kwargs,
    ):  # -> FlaxGreedySearchOutput | FlaxSampleOutput | FlaxBeamSearchOutput:
        ...
