from functools import lru_cache

from ...tokenization_utils import PreTrainedTokenizer

"""Tokenization class for CLVP."""
logger = ...
VOCAB_FILES_NAMES = ...

@lru_cache
def bytes_to_unicode():  # -> dict[int, str]:
    ...
def get_pairs(word):  # -> set[Any]:
    ...

class ClvpTokenizer(PreTrainedTokenizer):
    vocab_files_names = ...
    model_input_names = ...
    def __init__(
        self,
        vocab_file,
        merges_file,
        errors=...,
        unk_token=...,
        bos_token=...,
        eos_token=...,
        pad_token=...,
        add_prefix_space=...,
        add_bos_token=...,
        add_eos_token=...,
        **kwargs,
    ) -> None: ...
    @property
    def vocab_size(self):  # -> int:
        ...
    @property
    def normalizer(self):  # -> EnglishNormalizer:
        ...
    def get_vocab(self):  # -> dict[str, Any]:
        ...
    def bpe(self, token):  # -> LiteralString:
        ...
    def build_inputs_with_special_tokens(
        self, token_ids_0, token_ids_1=...
    ):  # -> list[str | list[str] | Any | None]:
        ...
    def get_special_tokens_mask(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = ...,
        already_has_special_tokens: bool = ...,
    ) -> list[int]: ...
    def convert_tokens_to_string(self, tokens):  # -> str:
        ...
    def clean_up_tokenization(self, text):  # -> str:
        ...
    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = ...
    ) -> tuple[str]: ...

__all__ = ["ClvpTokenizer"]
