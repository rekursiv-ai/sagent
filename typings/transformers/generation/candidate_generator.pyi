from typing import Any

from torch import nn

import torch

from .configuration_utils import GenerationConfig
from .logits_process import LogitsProcessorList
from ..modeling_utils import PreTrainedModel
from ..tokenization_utils_base import PreTrainedTokenizerBase

class CandidateGenerator:
    def get_candidates(
        self, input_ids: torch.LongTensor
    ) -> tuple[torch.LongTensor, torch.FloatTensor | None]: ...
    def update_candidate_strategy(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, num_matches: int
    ): ...

class AssistedCandidateGenerator(CandidateGenerator):
    def __init__(
        self,
        input_ids: torch.LongTensor,
        assistant_model: PreTrainedModel,
        generation_config: GenerationConfig,
        model_kwargs: dict,
        inputs_tensor: torch.Tensor | None = ...,
        logits_processor: LogitsProcessorList | None = ...,
    ) -> None: ...
    def get_candidates(
        self, input_ids: torch.LongTensor
    ) -> tuple[torch.LongTensor, torch.FloatTensor | None]: ...
    def update_candidate_strategy(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, num_matches: int
    ):  # -> None:
        ...

class AssistedCandidateGeneratorDifferentTokenizers(AssistedCandidateGenerator):
    def __init__(
        self,
        input_ids: torch.LongTensor,
        assistant_model: PreTrainedModel,
        target_tokenizer: PreTrainedTokenizerBase,
        assistant_tokenizer: PreTrainedTokenizerBase,
        generation_config: GenerationConfig,
        model_kwargs: dict,
        inputs_tensor: torch.Tensor | None = ...,
        logits_processor: LogitsProcessorList | None = ...,
    ) -> None: ...
    def convert_source_tokens_to_target_tokens(
        self, input_ids, source_tokenizer, destination_tokenizer
    ): ...
    def get_candidates(
        self, input_ids: torch.LongTensor
    ) -> tuple[torch.LongTensor, torch.FloatTensor | None]: ...

class _PruneReindexingLMHead(nn.Module):
    def __init__(self, original_lm_head, assistant_overlap_token_ids) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class _MapInputEmbedding(nn.Module):
    def __init__(
        self, original_embedding: nn.Embedding, assistant_overlap_token_ids
    ) -> None: ...
    def forward(self, input_ids: torch.LongTensor) -> torch.FloatTensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.FloatTensor: ...

class AssistantToTargetTranslator:
    FILTER_VALUE: float = ...
    SUPPRESS_TOKEN_ID: int = ...
    def __init__(
        self,
        target_tokenizer: PreTrainedTokenizerBase,
        assistant_tokenizer: PreTrainedTokenizerBase,
        target_vocab_size: int,
        assistant_model: PreTrainedModel | None = ...,
        assistant_prune_lm_head: bool = ...,
    ) -> None: ...
    def unmap_input_ids(self):  # -> None:
        ...
    def get_target_ids(
        self,
        assistant_input_ids,
        target_input_ids,
        assistant_candidate_ids: torch.LongTensor,
    ) -> torch.LongTensor: ...
    def get_target_logits(
        self, assistant_logits: torch.FloatTensor
    ) -> torch.FloatTensor: ...

class AssistantVocabTranslatorCache:
    _cache = ...
    @classmethod
    def get_translator(
        cls,
        target_tokenizer: PreTrainedTokenizerBase,
        assistant_tokenizer: PreTrainedTokenizerBase,
        target_vocab_size: int,
        assistant_model: PreTrainedModel | None = ...,
        assistant_prune_lm_head: bool = ...,
    ) -> AssistantToTargetTranslator: ...
    @classmethod
    def cleanup(cls):  # -> None:
        ...

class UniversalSpeculativeDecodingGenerator(
    AssistedCandidateGeneratorDifferentTokenizers
):
    def __init__(
        self,
        input_ids: torch.LongTensor,
        assistant_model: PreTrainedModel,
        target_tokenizer: PreTrainedTokenizerBase,
        assistant_tokenizer: PreTrainedTokenizerBase,
        generation_config: GenerationConfig,
        model_kwargs: dict,
        atm_translator: AssistantToTargetTranslator,
        inputs_tensor: torch.Tensor | None = ...,
        logits_processor: LogitsProcessorList | None = ...,
    ) -> None: ...
    def get_candidates(
        self, input_ids: torch.LongTensor
    ) -> tuple[torch.LongTensor, torch.FloatTensor | None]: ...

class PromptLookupCandidateGenerator(CandidateGenerator):
    def __init__(
        self,
        eos_token_id: torch.Tensor | None = ...,
        num_output_tokens: int = ...,
        max_matching_ngram_size: int = ...,
        max_length: int = ...,
        logits_processor: LogitsProcessorList | None = ...,
        vocab_size: int | None = ...,
    ) -> None: ...
    def get_candidates(
        self, input_ids: torch.LongTensor
    ) -> tuple[torch.LongTensor, torch.FloatTensor | None]: ...
    def update_candidate_strategy(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, num_matches: int
    ):  # -> None:
        ...

class EarlyExitCandidateGenerator(AssistedCandidateGenerator):
    def __init__(
        self,
        input_ids: torch.LongTensor,
        assistant_model: PreTrainedModel,
        generation_config: GenerationConfig,
        model_kwargs: dict,
        inputs_tensor: torch.Tensor | None = ...,
        logits_processor: LogitsProcessorList | None = ...,
    ) -> None: ...
    def get_candidates(
        self, input_ids: torch.LongTensor
    ) -> tuple[torch.LongTensor, torch.FloatTensor | None]: ...
