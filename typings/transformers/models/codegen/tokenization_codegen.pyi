from functools import lru_cache

import numpy as np

from ...tokenization_utils import PreTrainedTokenizer

"""Tokenization classes for CodeGen"""
logger = ...
VOCAB_FILES_NAMES = ...

@lru_cache
def bytes_to_unicode():  # -> dict[int, str]:
    ...
def get_pairs(word):  # -> set[Any]:
    ...

class CodeGenTokenizer(PreTrainedTokenizer):
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
        return_token_type_ids=...,
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
    def convert_tokens_to_string(self, tokens):  # -> str:
        ...
    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = ...
    ) -> tuple[str]: ...
    def prepare_for_tokenization(
        self, text, is_split_into_words=..., **kwargs
    ):  # -> tuple[str, dict[str, Any]]:
        ...
    def decode(
        self,
        token_ids: int | list[int] | np.ndarray | torch.Tensor | tf.Tensor,
        skip_special_tokens: bool = ...,
        clean_up_tokenization_spaces: bool | None = ...,
        truncate_before_pattern: list[str] | None = ...,
        **kwargs,
    ) -> str: ...
    def truncate(self, completion, truncate_before_pattern): ...

__all__ = ["CodeGenTokenizer"]
