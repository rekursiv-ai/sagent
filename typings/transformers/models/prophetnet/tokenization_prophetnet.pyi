from collections.abc import Iterable

from ...tokenization_utils import PreTrainedTokenizer

logger = ...
VOCAB_FILES_NAMES = ...

def whitespace_tokenize(text):  # -> list[Any]:
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

def load_vocab(vocab_file):  # -> OrderedDict[Any, Any]:
    ...

class ProphetNetTokenizer(PreTrainedTokenizer):
    vocab_files_names = ...
    model_input_names: list[str] = ...
    def __init__(
        self,
        vocab_file: str,
        do_lower_case: bool | None = ...,
        do_basic_tokenize: bool | None = ...,
        never_split: Iterable | None = ...,
        unk_token: str | None = ...,
        sep_token: str | None = ...,
        x_sep_token: str | None = ...,
        pad_token: str | None = ...,
        mask_token: str | None = ...,
        tokenize_chinese_chars: bool | None = ...,
        strip_accents: bool | None = ...,
        clean_up_tokenization_spaces: bool = ...,
        **kwargs,
    ) -> None: ...
    @property
    def vocab_size(self):  # -> int:
        ...
    def get_vocab(self):  # -> dict[str, int]:
        ...
    def convert_tokens_to_string(self, tokens: str):  # -> str:
        ...
    def get_special_tokens_mask(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = ...,
        already_has_special_tokens: bool | None = ...,
    ) -> list[int]: ...
    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = ...
    ) -> tuple[str]: ...
    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = ...
    ) -> list[int]: ...

__all__ = ["ProphetNetTokenizer"]
