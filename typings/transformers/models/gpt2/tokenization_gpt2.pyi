from functools import lru_cache

from ...tokenization_utils import PreTrainedTokenizer

"""Tokenization classes for OpenAI GPT."""
logger = ...
VOCAB_FILES_NAMES = ...

@lru_cache
def bytes_to_unicode():  # -> dict[int, str]:
    ...
def get_pairs(word):  # -> set[Any]:
    ...

class GPT2Tokenizer(PreTrainedTokenizer):
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
        **kwargs,
    ) -> None: ...
    @property
    def vocab_size(self):  # -> int:
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
    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = ...
    ) -> tuple[str]: ...
    def prepare_for_tokenization(
        self, text, is_split_into_words=..., **kwargs
    ):  # -> tuple[str, dict[str, Any]]:
        ...

__all__ = ["GPT2Tokenizer"]
