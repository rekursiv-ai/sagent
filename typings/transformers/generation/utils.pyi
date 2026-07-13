from collections.abc import Callable
from dataclasses import dataclass

import os

import torch

from .configuration_utils import GenerationConfig
from .continuous_batching import ContinuousMixin
from .logits_process import LogitsProcessorList
from .stopping_criteria import StoppingCriteriaList
from .streamers import BaseStreamer
from ..cache_utils import Cache
from ..modeling_utils import PreTrainedModel
from ..tokenization_utils_base import PreTrainedTokenizerBase
from ..utils import ModelOutput

logger = ...
ALL_CACHE_NAMES = ...
GENERATION_MODES_MAPPING = ...

@dataclass
class GenerateDecoderOnlyOutput(ModelOutput):
    sequences: torch.LongTensor
    scores: tuple[torch.FloatTensor] | None = ...
    logits: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[tuple[torch.FloatTensor]] | None = ...
    hidden_states: tuple[tuple[torch.FloatTensor]] | None = ...
    past_key_values: Cache | None = ...

@dataclass
class GenerateEncoderDecoderOutput(ModelOutput):
    sequences: torch.LongTensor
    scores: tuple[torch.FloatTensor] | None = ...
    logits: tuple[torch.FloatTensor] | None = ...
    encoder_attentions: tuple[torch.FloatTensor] | None = ...
    encoder_hidden_states: tuple[torch.FloatTensor] | None = ...
    decoder_attentions: tuple[tuple[torch.FloatTensor]] | None = ...
    cross_attentions: tuple[tuple[torch.FloatTensor]] | None = ...
    decoder_hidden_states: tuple[tuple[torch.FloatTensor]] | None = ...
    past_key_values: Cache | None = ...

@dataclass
class GenerateBeamDecoderOnlyOutput(ModelOutput):
    sequences: torch.LongTensor
    sequences_scores: torch.FloatTensor | None = ...
    scores: tuple[torch.FloatTensor] | None = ...
    logits: tuple[torch.FloatTensor] | None = ...
    beam_indices: torch.LongTensor | None = ...
    attentions: tuple[tuple[torch.FloatTensor]] | None = ...
    hidden_states: tuple[tuple[torch.FloatTensor]] | None = ...
    past_key_values: Cache | None = ...

@dataclass
class GenerateBeamEncoderDecoderOutput(ModelOutput):
    sequences: torch.LongTensor
    sequences_scores: torch.FloatTensor | None = ...
    scores: tuple[torch.FloatTensor] | None = ...
    logits: tuple[torch.FloatTensor] | None = ...
    beam_indices: torch.LongTensor | None = ...
    encoder_attentions: tuple[torch.FloatTensor] | None = ...
    encoder_hidden_states: tuple[torch.FloatTensor] | None = ...
    decoder_attentions: tuple[tuple[torch.FloatTensor]] | None = ...
    cross_attentions: tuple[tuple[torch.FloatTensor]] | None = ...
    decoder_hidden_states: tuple[tuple[torch.FloatTensor]] | None = ...
    past_key_values: Cache | None = ...

GreedySearchDecoderOnlyOutput = GenerateDecoderOnlyOutput
ContrastiveSearchDecoderOnlyOutput = GenerateDecoderOnlyOutput
SampleDecoderOnlyOutput = GenerateDecoderOnlyOutput
ContrastiveSearchEncoderDecoderOutput = GenerateEncoderDecoderOutput
GreedySearchEncoderDecoderOutput = GenerateEncoderDecoderOutput
SampleEncoderDecoderOutput = GenerateEncoderDecoderOutput
BeamSearchDecoderOnlyOutput = GenerateBeamDecoderOnlyOutput
BeamSampleDecoderOnlyOutput = GenerateBeamDecoderOnlyOutput
BeamSearchEncoderDecoderOutput = GenerateBeamEncoderDecoderOutput
BeamSampleEncoderDecoderOutput = GenerateBeamEncoderDecoderOutput
type GreedySearchOutput = (
    GreedySearchEncoderDecoderOutput | GreedySearchDecoderOnlyOutput
)
type SampleOutput = SampleEncoderDecoderOutput | SampleDecoderOnlyOutput
type BeamSearchOutput = BeamSearchEncoderDecoderOutput | BeamSearchDecoderOnlyOutput
type BeamSampleOutput = BeamSampleEncoderDecoderOutput | BeamSampleDecoderOnlyOutput
type ContrastiveSearchOutput = (
    ContrastiveSearchEncoderDecoderOutput | ContrastiveSearchDecoderOnlyOutput
)
type GenerateNonBeamOutput = GenerateDecoderOnlyOutput | GenerateEncoderDecoderOutput
type GenerateBeamOutput = (
    GenerateBeamDecoderOnlyOutput | GenerateBeamEncoderDecoderOutput
)
type GenerateOutput = GenerateNonBeamOutput | GenerateBeamOutput

class GenerationMixin(ContinuousMixin):
    def load_custom_generate(
        self,
        pretrained_model_name_or_path: str | os.PathLike | None = ...,
        trust_remote_code: bool | None = ...,
        **kwargs,
    ) -> Callable: ...
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Cache | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs,
    ):  # -> dict[Any, Any]:
        ...
    def compute_transition_scores(
        self,
        sequences: torch.Tensor,
        scores: tuple[torch.Tensor],
        beam_indices: torch.Tensor | None = ...,
        normalize_logits: bool = ...,
    ) -> torch.Tensor: ...
    @torch.no_grad()
    def generate(
        self,
        inputs: torch.Tensor | None = ...,
        generation_config: GenerationConfig | None = ...,
        logits_processor: LogitsProcessorList | None = ...,
        stopping_criteria: StoppingCriteriaList | None = ...,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], list[int]] | None = ...,
        synced_gpus: bool | None = ...,
        assistant_model: PreTrainedModel | None = ...,
        streamer: BaseStreamer | None = ...,
        negative_prompt_ids: torch.Tensor | None = ...,
        negative_prompt_attention_mask: torch.Tensor | None = ...,
        use_model_defaults: bool | None = ...,
        custom_generate: str | Callable | None = ...,
        **kwargs,
    ) -> GenerateOutput | torch.LongTensor: ...
    def heal_tokens(
        self,
        input_ids: torch.LongTensor,
        tokenizer: PreTrainedTokenizerBase | None = ...,
    ) -> torch.LongTensor: ...
