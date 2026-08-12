from functools import lru_cache

import pandas as pd

from ....file_utils import (
    ExplicitEnum,
    PaddingStrategy,
    TensorType,
    add_end_docstrings,
)
from ....tokenization_utils import PreTrainedTokenizer
from ....tokenization_utils_base import (
    ENCODE_KWARGS_DOCSTRING,
    BatchEncoding,
    TextInput,
    TruncationStrategy,
)

"""Tokenization classes for TAPEX."""
logger = ...
VOCAB_FILES_NAMES = ...

class TapexTruncationStrategy(ExplicitEnum):
    DROP_ROWS_TO_FIT = ...

TAPEX_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING = ...

@lru_cache
def bytes_to_unicode():  # -> dict[int, str]:
    ...
def get_pairs(word):  # -> set[Any]:
    ...

class IndexedRowTableLinearize:
    def process_table(self, table_content: dict):  # -> LiteralString | str:
        ...
    def process_header(self, headers: list):  # -> LiteralString:
        ...
    def process_row(self, row: list, row_index: int):  # -> str:
        ...

class TapexTokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        vocab_file,
        merges_file,
        do_lower_case=...,
        errors=...,
        bos_token=...,
        eos_token=...,
        sep_token=...,
        cls_token=...,
        unk_token=...,
        pad_token=...,
        mask_token=...,
        add_prefix_space=...,
        max_cell_length=...,
        **kwargs,
    ) -> None: ...
    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = ...
    ) -> list[int]: ...
    def get_special_tokens_mask(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = ...,
        already_has_special_tokens: bool = ...,
    ) -> list[int]: ...
    def create_token_type_ids_from_sequences(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = ...
    ) -> list[int]: ...
    def prepare_for_tokenization(
        self, text, is_split_into_words=..., **kwargs
    ):  # -> tuple[str, dict[str, Any]]:
        ...
    @property
    def vocab_size(self):  # -> int:
        ...
    def get_vocab(self):  # -> dict[str, Any]:
        ...
    def bpe(self, token):  # -> LiteralString:
        ...
    def convert_tokens_to_string(self, tokens):  # -> str:
        ...
    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = ...
    ) -> tuple[str]: ...
    @add_end_docstrings(
        ENCODE_KWARGS_DOCSTRING, TAPEX_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING
    )
    def __call__(
        self,
        table: pd.DataFrame | list[pd.DataFrame] = ...,
        query: TextInput | list[TextInput] | None = ...,
        answer: str | list[str] | None = ...,
        add_special_tokens: bool = ...,
        padding: bool | str | PaddingStrategy = ...,
        truncation: bool | str | TruncationStrategy = ...,
        max_length: int | None = ...,
        stride: int = ...,
        pad_to_multiple_of: int | None = ...,
        return_tensors: str | TensorType | None = ...,
        return_token_type_ids: bool | None = ...,
        return_attention_mask: bool | None = ...,
        return_overflowing_tokens: bool = ...,
        return_special_tokens_mask: bool = ...,
        return_offsets_mapping: bool = ...,
        return_length: bool = ...,
        verbose: bool = ...,
        **kwargs,
    ) -> BatchEncoding: ...
    def source_call_func(
        self,
        table: pd.DataFrame | list[pd.DataFrame],
        query: TextInput | list[TextInput] | None = ...,
        answer: str | list[str] | None = ...,
        add_special_tokens: bool = ...,
        padding: bool | str | PaddingStrategy = ...,
        truncation: bool | str | TruncationStrategy = ...,
        max_length: int | None = ...,
        stride: int = ...,
        pad_to_multiple_of: int | None = ...,
        return_tensors: str | TensorType | None = ...,
        return_token_type_ids: bool | None = ...,
        return_attention_mask: bool | None = ...,
        return_overflowing_tokens: bool = ...,
        return_special_tokens_mask: bool = ...,
        return_offsets_mapping: bool = ...,
        return_length: bool = ...,
        verbose: bool = ...,
        **kwargs,
    ) -> BatchEncoding: ...
    @add_end_docstrings(
        ENCODE_KWARGS_DOCSTRING, TAPEX_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING
    )
    def batch_encode_plus(
        self,
        table: pd.DataFrame | list[pd.DataFrame],
        query: list[TextInput] | None = ...,
        answer: list[str] | None = ...,
        add_special_tokens: bool = ...,
        padding: bool | str | PaddingStrategy = ...,
        truncation: bool | str | None = ...,
        max_length: int | None = ...,
        pad_to_multiple_of: int | None = ...,
        return_tensors: str | TensorType | None = ...,
        return_token_type_ids: bool | None = ...,
        return_attention_mask: bool | None = ...,
        return_overflowing_tokens: bool = ...,
        return_special_tokens_mask: bool = ...,
        return_offsets_mapping: bool = ...,
        return_length: bool = ...,
        verbose: bool = ...,
        **kwargs,
    ) -> BatchEncoding: ...
    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING)
    def encode(
        self,
        table: pd.DataFrame,
        query: TextInput | None = ...,
        answer: str | None = ...,
        add_special_tokens: bool = ...,
        padding: bool | str | PaddingStrategy = ...,
        truncation: bool | str | TruncationStrategy | TapexTruncationStrategy = ...,
        max_length: int | None = ...,
        return_tensors: str | TensorType | None = ...,
        **kwargs,
    ) -> list[int]: ...
    @add_end_docstrings(
        ENCODE_KWARGS_DOCSTRING, TAPEX_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING
    )
    def encode_plus(
        self,
        table: pd.DataFrame,
        query: TextInput | None = ...,
        answer: str | None = ...,
        add_special_tokens: bool = ...,
        padding: bool | str | PaddingStrategy = ...,
        truncation: bool | str | None = ...,
        max_length: int | None = ...,
        pad_to_multiple_of: int | None = ...,
        return_tensors: str | TensorType | None = ...,
        return_token_type_ids: bool | None = ...,
        return_attention_mask: bool | None = ...,
        return_special_tokens_mask: bool = ...,
        return_offsets_mapping: bool = ...,
        return_length: bool = ...,
        verbose: bool = ...,
        **kwargs,
    ) -> BatchEncoding: ...
    def target_call_func(
        self,
        answer: str | list[str],
        add_special_tokens: bool = ...,
        padding: bool | str | PaddingStrategy = ...,
        truncation: bool | str | TruncationStrategy = ...,
        max_length: int | None = ...,
        stride: int = ...,
        pad_to_multiple_of: int | None = ...,
        return_tensors: str | TensorType | None = ...,
        return_token_type_ids: bool | None = ...,
        return_attention_mask: bool | None = ...,
        return_overflowing_tokens: bool = ...,
        return_special_tokens_mask: bool = ...,
        return_offsets_mapping: bool = ...,
        return_length: bool = ...,
        verbose: bool = ...,
        **kwargs,
    ) -> BatchEncoding: ...
    def target_batch_encode_plus(
        self,
        answer: list[str],
        add_special_tokens: bool = ...,
        padding: bool | str | PaddingStrategy = ...,
        truncation: bool | str | None = ...,
        max_length: int | None = ...,
        pad_to_multiple_of: int | None = ...,
        return_tensors: str | TensorType | None = ...,
        return_token_type_ids: bool | None = ...,
        return_attention_mask: bool | None = ...,
        return_overflowing_tokens: bool = ...,
        return_special_tokens_mask: bool = ...,
        return_offsets_mapping: bool = ...,
        return_length: bool = ...,
        verbose: bool = ...,
        **kwargs,
    ) -> BatchEncoding: ...
    def target_encode(
        self,
        answer: str,
        add_special_tokens: bool = ...,
        padding: bool | str | PaddingStrategy = ...,
        truncation: bool | str | TruncationStrategy | TapexTruncationStrategy = ...,
        max_length: int | None = ...,
        return_tensors: str | TensorType | None = ...,
        **kwargs,
    ) -> list[int]: ...
    def target_encode_plus(
        self,
        answer: str,
        add_special_tokens: bool = ...,
        padding: bool | str | PaddingStrategy = ...,
        truncation: bool | str | None = ...,
        max_length: int | None = ...,
        pad_to_multiple_of: int | None = ...,
        return_tensors: str | TensorType | None = ...,
        return_token_type_ids: bool | None = ...,
        return_attention_mask: bool | None = ...,
        return_special_tokens_mask: bool = ...,
        return_offsets_mapping: bool = ...,
        return_length: bool = ...,
        verbose: bool = ...,
        **kwargs,
    ) -> BatchEncoding: ...
    def prepare_table_query(
        self, table, query, answer=..., truncation_strategy=..., max_length=...
    ):  # -> LiteralString | str:
        ...
    def truncate_table_cells(
        self, table_content: dict, question: str, answer: list
    ):  # -> None:
        ...
    def truncate_cell(self, cell_value):  # -> int | float | str | None:
        ...
    def truncate_table_rows(
        self,
        table_content: dict,
        question: str,
        answer: str | list[str] | None = ...,
        max_length=...,
    ):  # -> None:
        ...
    def estimate_delete_ratio(
        self, table_content: dict, question: str, max_length=...
    ):  # -> tuple[float, Any] | tuple[Any, Any]:
        ...
    def delete_unrelated_rows(
        self, table_content: dict, question: str, answer: list, delete_ratio: float
    ):  # -> None:
        ...

__all__ = ["TapexTokenizer"]
