from typing import Any

from ...tokenization_utils import PreTrainedTokenizer

"""Tokenization class for VITS."""
logger = ...
VOCAB_FILES_NAMES = ...

def has_non_roman_characters(input_string):  # -> bool:
    ...

class VitsTokenizer(PreTrainedTokenizer):
    vocab_files_names = ...
    model_input_names = ...
    def __init__(
        self,
        vocab_file,
        pad_token=...,
        unk_token=...,
        language=...,
        add_blank=...,
        normalize=...,
        phonemize=...,
        is_uroman=...,
        **kwargs,
    ) -> None: ...
    @property
    def vocab_size(self):  # -> int:
        ...
    def get_vocab(self):  # -> dict[str, int]:
        ...
    def normalize_text(self, input_string):  # -> Any | Literal['']:
        ...
    def prepare_for_tokenization(
        self,
        text: str,
        is_split_into_words: bool = ...,
        normalize: bool | None = ...,
        **kwargs,
    ) -> tuple[str, dict[str, Any]]: ...
    def convert_tokens_to_string(self, tokens: list[str]) -> str: ...
    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = ...
    ) -> tuple[str] | None: ...

__all__ = ["VitsTokenizer"]
