from collections.abc import Callable
from dataclasses import dataclass

import enum

import pandas as pd

from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import (
    ENCODE_KWARGS_DOCSTRING,
    BatchEncoding,
    EncodedInput,
    PreTokenizedInput,
    TextInput,
)
from ...utils import (
    ExplicitEnum,
    PaddingStrategy,
    TensorType,
    add_end_docstrings,
)

"""Tokenization class for TAPAS model."""
logger = ...
VOCAB_FILES_NAMES = ...

class TapasTruncationStrategy(ExplicitEnum):
    DROP_ROWS_TO_FIT = ...
    DO_NOT_TRUNCATE = ...

TableValue = ...

@dataclass(frozen=True)
class TokenCoordinates:
    column_index: int
    row_index: int
    token_index: int

@dataclass
class TokenizedTable:
    rows: list[list[list[str]]]
    selected_tokens: list[TokenCoordinates]

@dataclass(frozen=True)
class SerializedExample:
    tokens: list[str]
    column_ids: list[int]
    row_ids: list[int]
    segment_ids: list[int]

def load_vocab(vocab_file):  # -> OrderedDict[Any, Any]:
    ...
def whitespace_tokenize(text):  # -> list[Any]:
    ...

TAPAS_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING = ...

class TapasTokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        vocab_file,
        do_lower_case=...,
        do_basic_tokenize=...,
        never_split=...,
        unk_token=...,
        sep_token=...,
        pad_token=...,
        cls_token=...,
        mask_token=...,
        empty_token=...,
        tokenize_chinese_chars=...,
        strip_accents=...,
        cell_trim_length: int = ...,
        max_column_id: int | None = ...,
        max_row_id: int | None = ...,
        strip_column_names: bool = ...,
        update_answer_coordinates: bool = ...,
        min_question_length=...,
        max_question_length=...,
        model_max_length: int = ...,
        additional_special_tokens: list[str] | None = ...,
        clean_up_tokenization_spaces=...,
        **kwargs,
    ) -> None: ...
    @property
    def do_lower_case(self):  # -> bool:
        ...
    @property
    def vocab_size(self):  # -> int:
        ...
    def get_vocab(self):  # -> dict[str, int]:
        ...
    def convert_tokens_to_string(self, tokens):  # -> str:
        ...
    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = ...
    ) -> tuple[str]: ...
    def create_attention_mask_from_sequences(
        self, query_ids: list[int], table_values: list[TableValue]
    ) -> list[int]: ...
    def create_segment_token_type_ids_from_sequences(
        self, query_ids: list[int], table_values: list[TableValue]
    ) -> list[int]: ...
    def create_column_token_type_ids_from_sequences(
        self, query_ids: list[int], table_values: list[TableValue]
    ) -> list[int]: ...
    def create_row_token_type_ids_from_sequences(
        self, query_ids: list[int], table_values: list[TableValue]
    ) -> list[int]: ...
    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = ...
    ) -> list[int]: ...
    def get_special_tokens_mask(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = ...,
        already_has_special_tokens: bool = ...,
    ) -> list[int]: ...
    @add_end_docstrings(TAPAS_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def __call__(
        self,
        table: pd.DataFrame,
        queries: TextInput
        | PreTokenizedInput
        | EncodedInput
        | list[TextInput]
        | list[PreTokenizedInput]
        | list[EncodedInput]
        | None = ...,
        answer_coordinates: list[tuple] | list[list[tuple]] | None = ...,
        answer_text: list[TextInput] | list[list[TextInput]] | None = ...,
        add_special_tokens: bool = ...,
        padding: bool | str | PaddingStrategy = ...,
        truncation: bool | str | TapasTruncationStrategy = ...,
        max_length: int | None = ...,
        pad_to_multiple_of: int | None = ...,
        padding_side: str | None = ...,
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
        ENCODE_KWARGS_DOCSTRING, TAPAS_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING
    )
    def batch_encode_plus(
        self,
        table: pd.DataFrame,
        queries: list[TextInput]
        | list[PreTokenizedInput]
        | list[EncodedInput]
        | None = ...,
        answer_coordinates: list[list[tuple]] | None = ...,
        answer_text: list[list[TextInput]] | None = ...,
        add_special_tokens: bool = ...,
        padding: bool | str | PaddingStrategy = ...,
        truncation: bool | str | TapasTruncationStrategy = ...,
        max_length: int | None = ...,
        pad_to_multiple_of: int | None = ...,
        padding_side: str | None = ...,
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
        query: TextInput | PreTokenizedInput | EncodedInput | None = ...,
        add_special_tokens: bool = ...,
        padding: bool | str | PaddingStrategy = ...,
        truncation: bool | str | TapasTruncationStrategy = ...,
        max_length: int | None = ...,
        return_tensors: str | TensorType | None = ...,
        **kwargs,
    ) -> list[int]: ...
    @add_end_docstrings(
        ENCODE_KWARGS_DOCSTRING, TAPAS_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING
    )
    def encode_plus(
        self,
        table: pd.DataFrame,
        query: TextInput | PreTokenizedInput | EncodedInput | None = ...,
        answer_coordinates: list[tuple] | None = ...,
        answer_text: list[TextInput] | None = ...,
        add_special_tokens: bool = ...,
        padding: bool | str | PaddingStrategy = ...,
        truncation: bool | str | TapasTruncationStrategy = ...,
        max_length: int | None = ...,
        pad_to_multiple_of: int | None = ...,
        padding_side: str | None = ...,
        return_tensors: str | TensorType | None = ...,
        return_token_type_ids: bool | None = ...,
        return_attention_mask: bool | None = ...,
        return_special_tokens_mask: bool = ...,
        return_offsets_mapping: bool = ...,
        return_length: bool = ...,
        verbose: bool = ...,
        **kwargs,
    ) -> BatchEncoding: ...
    @add_end_docstrings(
        ENCODE_KWARGS_DOCSTRING, TAPAS_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING
    )
    def prepare_for_model(
        self,
        raw_table: pd.DataFrame,
        raw_query: TextInput | PreTokenizedInput | EncodedInput,
        tokenized_table: TokenizedTable | None = ...,
        query_tokens: TokenizedTable | None = ...,
        answer_coordinates: list[tuple] | None = ...,
        answer_text: list[TextInput] | None = ...,
        add_special_tokens: bool = ...,
        padding: bool | str | PaddingStrategy = ...,
        truncation: bool | str | TapasTruncationStrategy = ...,
        max_length: int | None = ...,
        pad_to_multiple_of: int | None = ...,
        padding_side: str | None = ...,
        return_tensors: str | TensorType | None = ...,
        return_token_type_ids: bool | None = ...,
        return_attention_mask: bool | None = ...,
        return_special_tokens_mask: bool = ...,
        return_offsets_mapping: bool = ...,
        return_length: bool = ...,
        verbose: bool = ...,
        prepend_batch_axis: bool = ...,
        **kwargs,
    ) -> BatchEncoding: ...
    def get_answer_ids(
        self,
        column_ids,
        row_ids,
        tokenized_table,
        answer_texts_question,
        answer_coordinates_question,
    ):  # -> list[int]:
        ...
    def convert_logits_to_predictions(
        self, data, logits, logits_agg=..., cell_classification_threshold=...
    ):  # -> tuple[list[Any], Any] | tuple[list[Any]]:
        ...

class BasicTokenizer:
    def __init__(
        self,
        do_lower_case=...,
        never_split=...,
        tokenize_chinese_chars=...,
        strip_accents=...,
        do_split_on_punc=...,
    ) -> None: ...
    def tokenize(self, text, never_split=...):  # -> list[Any] | list[LiteralString]:
        ...

class WordpieceTokenizer:
    def __init__(self, vocab, unk_token, max_input_chars_per_word=...) -> None: ...
    def tokenize(self, text):  # -> list[Any]:
        ...

class Relation(enum.Enum):
    HEADER_TO_CELL = ...
    CELL_TO_HEADER = ...
    QUERY_TO_HEADER = ...
    QUERY_TO_CELL = ...
    ROW_TO_CELL = ...
    CELL_TO_ROW = ...
    EQ = ...
    LT = ...
    GT = ...

@dataclass
class Date:
    year: int | None = ...
    month: int | None = ...
    day: int | None = ...

@dataclass
class NumericValue:
    float_value: float | None = ...
    date: Date | None = ...

@dataclass
class NumericValueSpan:
    begin_index: int | None = ...
    end_index: int | None = ...
    values: list[NumericValue] = ...

@dataclass
class Cell:
    text: str
    numeric_value: NumericValue | None = ...

@dataclass
class Question:
    original_text: str
    text: str
    numeric_spans: list[NumericValueSpan] | None = ...

_DateMask = ...
_YEAR = ...
_YEAR_MONTH = ...
_YEAR_MONTH_DAY = ...
_MONTH = ...
_MONTH_DAY = ...
_DATE_PATTERNS = ...
_FIELD_TO_REGEX = ...
_PROCESSED_DATE_PATTERNS = ...
_MAX_DATE_NGRAM_SIZE = ...
_NUMBER_WORDS = ...
_ORDINAL_WORDS = ...
_ORDINAL_SUFFIXES = ...
_NUMBER_PATTERN = ...
_MIN_YEAR = ...
_MAX_YEAR = ...
_INF = ...

def get_all_spans(text, max_ngram_length):  # -> Generator[tuple[Any, int], Any, None]:
    ...
def normalize_for_match(text):  # -> str:
    ...
def format_text(text):  # -> str:
    ...
def parse_text(text):  # -> list[Any]:
    ...

type _PrimitiveNumericValue = float | tuple[float | None, float | None, float | None]
type _SortKeyFn = Callable[[NumericValue], tuple[float, Ellipsis]]
_DATE_TUPLE_SIZE = ...
EMPTY_TEXT = ...
NUMBER_TYPE = ...
DATE_TYPE = ...

def get_numeric_sort_key_fn(
    numeric_values,
):  # -> Callable[..., Any | tuple[None, ...]]:
    ...
def get_numeric_relation(
    value, other_value, sort_key_fn
):  # -> Literal[Relation.EQ, Relation.LT, Relation.GT] | None:
    ...
def add_numeric_values_to_question(question):  # -> Question:
    ...
def filter_invalid_unicode(
    text,
):  # -> tuple[Literal[''], Literal[True]] | tuple[Any, Literal[False]]:
    ...
def filter_invalid_unicode_from_table(table):  # -> None:
    ...
def add_numeric_table_values(table, min_consolidation_fraction=..., debug_info=...): ...

__all__ = ["TapasTokenizer"]
