from abc import ABC

import torch

from ..tokenization_utils_base import PreTrainedTokenizerBase
from ..utils import add_start_docstrings

logger = ...
STOP_STRING_EMBEDDING_CACHE = ...
STOPPING_CRITERIA_INPUTS_DOCSTRING = ...

class StoppingCriteria(ABC):
    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> torch.BoolTensor: ...

class MaxLengthCriteria(StoppingCriteria):
    def __init__(
        self, max_length: int, max_position_embeddings: int | None = ...
    ) -> None: ...
    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> torch.BoolTensor: ...

class MaxTimeCriteria(StoppingCriteria):
    def __init__(
        self, max_time: float, initial_timestamp: float | None = ...
    ) -> None: ...
    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> torch.BoolTensor: ...

class StopStringCriteria(StoppingCriteria):
    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, stop_strings: str | list[str]
    ) -> None: ...
    def clean_and_embed_tokens_with_cache(
        self, token_list, token_indices, tokenizer
    ):  # -> tuple[Any | str, Any | str, Any | str]:
        ...
    @staticmethod
    def clean_tokenizer_vocab(
        tokenizer, static_prefix=...
    ):  # -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        ...
    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> torch.Tensor: ...

class EosTokenCriteria(StoppingCriteria):
    def __init__(self, eos_token_id: int | list[int] | torch.Tensor) -> None: ...
    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> torch.BoolTensor: ...

class ConfidenceCriteria(StoppingCriteria):
    def __init__(self, assistant_confidence_threshold) -> None: ...
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> torch.BoolTensor: ...

class StoppingCriteriaList(list):
    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> torch.BoolTensor: ...
    @property
    def max_length(self) -> int | None: ...

def validate_stopping_criteria(
    stopping_criteria: StoppingCriteriaList, max_length: int
) -> StoppingCriteriaList: ...
