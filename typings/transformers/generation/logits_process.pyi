from collections.abc import Callable

import torch

from ..generation.configuration_utils import GenerationConfig
from ..utils import add_start_docstrings

logger = ...
LOGITS_PROCESSOR_INPUTS_DOCSTRING = ...

class LogitsProcessor:
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor: ...

class LogitsProcessorList(list):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> torch.FloatTensor: ...

class MinLengthLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        min_length: int,
        eos_token_id: int | list[int] | torch.Tensor,
        device: str = ...,
    ) -> None: ...
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor: ...

class MinNewTokensLengthLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        prompt_length_to_skip: int,
        min_new_tokens: int,
        eos_token_id: int | list[int] | torch.Tensor,
        device: str = ...,
    ) -> None: ...
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor: ...

class TemperatureLogitsWarper(LogitsProcessor):
    def __init__(self, temperature: float) -> None: ...
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor: ...

class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
    def __init__(
        self, penalty: float, prompt_ignore_length: int | None = ...
    ) -> None: ...
    def set_continuous_batching_context(
        self, logits_indices: torch.Tensor, cu_seq_lens_q: torch.Tensor
    ):  # -> None:
        ...
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor: ...

class EncoderRepetitionPenaltyLogitsProcessor(LogitsProcessor):
    def __init__(self, penalty: float, encoder_input_ids: torch.LongTensor) -> None: ...
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor: ...

class TopPLogitsWarper(LogitsProcessor):
    def __init__(
        self, top_p: float, filter_value: float = ..., min_tokens_to_keep: int = ...
    ) -> None: ...
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor: ...

class TopKLogitsWarper(LogitsProcessor):
    def __init__(
        self, top_k: int, filter_value: float = ..., min_tokens_to_keep: int = ...
    ) -> None: ...
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor: ...

class MinPLogitsWarper(LogitsProcessor):
    def __init__(
        self, min_p: float, filter_value: float = ..., min_tokens_to_keep: int = ...
    ) -> None: ...
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor: ...

class TypicalLogitsWarper(LogitsProcessor):
    def __init__(
        self,
        mass: float = ...,
        filter_value: float = ...,
        min_tokens_to_keep: int = ...,
    ) -> None: ...
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor: ...

class EpsilonLogitsWarper(LogitsProcessor):
    def __init__(
        self, epsilon: float, filter_value: float = ..., min_tokens_to_keep: int = ...
    ) -> None: ...
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor: ...

class EtaLogitsWarper(LogitsProcessor):
    def __init__(
        self,
        epsilon: float,
        filter_value: float = ...,
        min_tokens_to_keep: int = ...,
        device: str = ...,
    ) -> None: ...
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor: ...

class NoRepeatNGramLogitsProcessor(LogitsProcessor):
    def __init__(self, ngram_size: int) -> None: ...
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor: ...

class EncoderNoRepeatNGramLogitsProcessor(LogitsProcessor):
    def __init__(
        self, encoder_ngram_size: int, encoder_input_ids: torch.LongTensor
    ) -> None: ...
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor: ...

class SequenceBiasLogitsProcessor(LogitsProcessor):
    def __init__(self, sequence_bias: list[list[list[int] | float]]) -> None: ...
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor: ...

class NoBadWordsLogitsProcessor(SequenceBiasLogitsProcessor):
    def __init__(
        self,
        bad_words_ids: list[list[int]],
        eos_token_id: int | list[int] | torch.Tensor | None = ...,
    ) -> None: ...

class PrefixConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], list[int]],
        num_beams: int,
    ) -> None: ...
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor: ...

class HammingDiversityLogitsProcessor(LogitsProcessor):
    def __init__(
        self, diversity_penalty: float, num_beams: int, num_beam_groups: int
    ) -> None: ...
    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        current_tokens: torch.LongTensor,
        beam_group_idx: int,
    ) -> torch.FloatTensor: ...

class ForcedBOSTokenLogitsProcessor(LogitsProcessor):
    def __init__(self, bos_token_id: int) -> None: ...
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor: ...

class ForcedEOSTokenLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        max_length: int,
        eos_token_id: int | list[int] | torch.Tensor,
        device: str = ...,
    ) -> None: ...
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor: ...

class InfNanRemoveLogitsProcessor(LogitsProcessor):
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor: ...

class ExponentialDecayLengthPenalty(LogitsProcessor):
    def __init__(
        self,
        exponential_decay_length_penalty: tuple[int, float],
        eos_token_id: int | list[int] | torch.Tensor,
        input_ids_seq_length: int,
    ) -> None: ...
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor: ...

class LogitNormalization(LogitsProcessor):
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor: ...

class SuppressTokensAtBeginLogitsProcessor(LogitsProcessor):
    def __init__(
        self, begin_suppress_tokens, begin_index, device: str = ...
    ) -> None: ...
    def set_begin_index(self, begin_index):  # -> None:
        ...
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor: ...

class SuppressTokensLogitsProcessor(LogitsProcessor):
    def __init__(self, suppress_tokens, device: str = ...) -> None: ...
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor: ...

class WhisperTimeStampLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        generate_config: GenerationConfig,
        begin_index: int,
        _detect_timestamp_from_logprob: bool | None = ...,
    ) -> None: ...
    def set_begin_index(self, begin_index):  # -> None:
        ...
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor: ...

class WhisperNoSpeechDetection(LogitsProcessor):
    def __init__(
        self, no_speech_token: int, begin_index: int, scores_is_logprobs: bool = ...
    ) -> None: ...
    def set_model(self, model):  # -> None:
        ...
    def set_inputs(self, inputs):  # -> None:
        ...
    @property
    def no_speech_prob(self):  # -> list[float] | Tensor:
        ...
    def set_begin_index(self, begin_index):  # -> None:
        ...
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor: ...

class ClassifierFreeGuidanceLogitsProcessor(LogitsProcessor):
    def __init__(self, guidance_scale) -> None: ...
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor: ...

class AlternatingCodebooksLogitsProcessor(LogitsProcessor):
    def __init__(
        self, input_start_len: int, semantic_vocab_size: int, codebook_size: int
    ) -> None: ...
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor: ...

class UnbatchedClassifierFreeGuidanceLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        guidance_scale: float,
        model,
        unconditional_ids: torch.LongTensor | None = ...,
        unconditional_attention_mask: torch.LongTensor | None = ...,
        use_cache: bool = ...,
    ) -> None: ...
    def get_unconditional_logits(self, input_ids): ...
    def __call__(self, input_ids, scores):  # -> Tensor:
        ...

class BarkEosPrioritizerLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        eos_token_id: int | list[int] | torch.Tensor,
        min_eos_p: float,
        device: str = ...,
    ) -> None: ...
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor: ...

class WatermarkLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        vocab_size,
        device,
        greenlist_ratio: float = ...,
        bias: float = ...,
        hashing_key: int = ...,
        seeding_scheme: str = ...,
        context_width: int = ...,
    ) -> None: ...
    def set_seed(self, input_seq: torch.LongTensor):  # -> None:
        ...
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor: ...

class SynthIDTextWatermarkState:
    def __init__(
        self,
        batch_size: int,
        ngram_len: int,
        context_history_size: int,
        device: torch.device,
    ) -> None: ...

class SynthIDTextWatermarkLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        ngram_len: int,
        keys: list[int],
        sampling_table_size: int,
        sampling_table_seed: int,
        context_history_size: int,
        device: torch.device,
        skip_first_ngram_calls: bool = ...,
        debug_mode: bool = ...,
    ) -> None: ...
    def update_scores(
        self, scores: torch.FloatTensor, g_values: torch.FloatTensor
    ) -> torch.FloatTensor: ...
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor: ...
    def accumulate_hash(
        self,
        current_hash: torch.LongTensor,
        data: torch.LongTensor,
        multiplier: int = ...,
        increment: int = ...,
    ) -> torch.LongTensor: ...
    def compute_ngram_keys(self, ngrams: torch.LongTensor) -> torch.LongTensor: ...
    def sample_g_values(self, ngram_keys: torch.LongTensor) -> torch.LongTensor: ...
    def compute_g_values(self, input_ids: torch.LongTensor) -> torch.LongTensor: ...
    def compute_context_repetition_mask(
        self, input_ids: torch.LongTensor
    ) -> torch.LongTensor: ...
    def compute_eos_token_mask(
        self, input_ids: torch.LongTensor, eos_token_id: int
    ) -> torch.LongTensor: ...
    def expected_mean_g_value(
        self, vocab_size: int, coinflip_prob: float = ...
    ) -> float: ...

class DiaClassifierFreeGuidanceLogitsProcessor(LogitsProcessor):
    def __init__(
        self, guidance_scale: float, guidance_top_k: int | None = ...
    ) -> None: ...
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor: ...

class DiaEOSChannelFilterLogitsProcessor(LogitsProcessor):
    def __init__(self, num_channels: int, eos_token_id: int) -> None: ...
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor: ...

class DiaEOSDelayPatternLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        delay_pattern: list[int],
        eos_token_id: int,
        max_generation_len: int,
        device: str = ...,
    ) -> None: ...
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor: ...
