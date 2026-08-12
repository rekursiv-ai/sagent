from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import (
    ENCODE_KWARGS_DOCSTRING,
    ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING,
    BatchEncoding,
    PaddingStrategy,
    TensorType,
    TruncationStrategy,
)
from ...utils import add_end_docstrings

"""Tokenization classes for RoCBert."""
logger = ...
VOCAB_FILES_NAMES = ...

def load_vocab(vocab_file):  # -> OrderedDict[Any, Any]:
    ...
def whitespace_tokenize(text):  # -> list[Any]:
    ...

class RoCBertTokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        vocab_file,
        word_shape_file,
        word_pronunciation_file,
        do_lower_case=...,
        do_basic_tokenize=...,
        never_split=...,
        unk_token=...,
        sep_token=...,
        pad_token=...,
        cls_token=...,
        mask_token=...,
        tokenize_chinese_chars=...,
        strip_accents=...,
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
    @add_end_docstrings(
        ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING
    )
    def prepare_for_model(
        self,
        ids: list[int],
        shape_ids: list[int],
        pronunciation_ids: list[int],
        pair_ids: list[int] | None = ...,
        pair_shape_ids: list[int] | None = ...,
        pair_pronunciation_ids: list[int] | None = ...,
        add_special_tokens: bool = ...,
        padding: bool | str | PaddingStrategy = ...,
        truncation: bool | str | TruncationStrategy = ...,
        max_length: int | None = ...,
        stride: int = ...,
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
        prepend_batch_axis: bool = ...,
        **kwargs,
    ) -> BatchEncoding: ...
    def convert_tokens_to_shape_ids(
        self, tokens: str | list[str]
    ) -> int | list[int]: ...
    def convert_tokens_to_pronunciation_ids(
        self, tokens: str | list[str]
    ) -> int | list[int]: ...
    def convert_tokens_to_string(self, tokens):  # -> str:
        ...
    def build_inputs_with_special_tokens(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = ...,
        cls_token_id: int | None = ...,
        sep_token_id: int | None = ...,
    ) -> list[int]: ...
    def get_special_tokens_mask(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = ...,
        already_has_special_tokens: bool = ...,
    ) -> list[int]: ...
    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = ...
    ) -> tuple[str, str, str]: ...

class RoCBertBasicTokenizer:
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

class RoCBertWordpieceTokenizer:
    def __init__(self, vocab, unk_token, max_input_chars_per_word=...) -> None: ...
    def tokenize(self, text):  # -> list[Any]:
        ...

__all__ = ["RoCBertTokenizer"]
