from functools import lru_cache

from ...tokenization_utils import PreTrainedTokenizer

"""Tokenization classes for Qwen2."""
logger = ...
VOCAB_FILES_NAMES = ...
MAX_MODEL_INPUT_SIZES = ...
PRETOKENIZE_REGEX = ...

@lru_cache
def bytes_to_unicode():  # -> dict[int, str]:
    ...
def get_pairs(word):  # -> set[Any]:
    ...

class Qwen2Tokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        vocab_file,
        merges_file,
        errors=...,
        unk_token=...,
        bos_token=...,
        eos_token=...,
        pad_token=...,
        clean_up_tokenization_spaces=...,
        split_special_tokens=...,
        **kwargs,
    ) -> None: ...
    @property
    def vocab_size(self) -> int: ...
    def get_vocab(self):  # -> dict[str, Any]:
        ...
    def bpe(self, token):  # -> LiteralString:
        ...
    def convert_tokens_to_string(self, tokens):  # -> str:
        ...
    def decode(
        self,
        token_ids,
        skip_special_tokens: bool = ...,
        clean_up_tokenization_spaces: bool | None = ...,
        spaces_between_special_tokens: bool = ...,
        **kwargs,
    ) -> str: ...
    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = ...
    ) -> tuple[str]: ...
    def prepare_for_tokenization(
        self, text, **kwargs
    ):  # -> tuple[str, dict[str, Any]]:
        ...

__all__ = ["Qwen2Tokenizer"]
