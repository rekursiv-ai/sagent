from dataclasses import dataclass
from typing import Any

import numpy as np

from ...tokenization_utils import PreTrainedTokenizer
from ...utils import ModelOutput

"""Tokenization class for Wav2Vec2Phoneme."""
logger = ...
VOCAB_FILES_NAMES = ...
type ListOfDict = list[dict[str, int | str]]

@dataclass
class Wav2Vec2PhonemeCTCTokenizerOutput(ModelOutput):
    text: list[str] | str
    char_offsets: list[ListOfDict] | ListOfDict = ...

class Wav2Vec2PhonemeCTCTokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        vocab_file,
        bos_token=...,
        eos_token=...,
        unk_token=...,
        pad_token=...,
        phone_delimiter_token=...,
        word_delimiter_token=...,
        do_phonemize=...,
        phonemizer_lang=...,
        phonemizer_backend=...,
        **kwargs,
    ) -> None: ...
    @property
    def vocab_size(self) -> int: ...
    def get_vocab(self) -> dict: ...
    def init_backend(self, phonemizer_lang: str):  # -> None:
        ...
    def prepare_for_tokenization(
        self,
        text: str,
        is_split_into_words: bool = ...,
        phonemizer_lang: str | None = ...,
        do_phonemize: bool | None = ...,
    ) -> tuple[str, dict[str, Any]]: ...
    def phonemize(self, text: str, phonemizer_lang: str | None = ...) -> str: ...
    @property
    def word_delimiter_token(self) -> str: ...
    @property
    def word_delimiter_token_id(self) -> int | None: ...
    @word_delimiter_token.setter
    def word_delimiter_token(self, value):  # -> None:
        ...
    @word_delimiter_token_id.setter
    def word_delimiter_token_id(self, value):  # -> None:
        ...
    @property
    def phone_delimiter_token(self) -> str: ...
    @property
    def phone_delimiter_token_id(self) -> int | None: ...
    @phone_delimiter_token.setter
    def phone_delimiter_token(self, value):  # -> None:
        ...
    @phone_delimiter_token_id.setter
    def phone_delimiter_token_id(self, value):  # -> None:
        ...
    def convert_tokens_to_string(
        self,
        tokens: list[str],
        group_tokens: bool = ...,
        spaces_between_special_tokens: bool = ...,
        filter_word_delimiter_token: bool = ...,
        output_char_offsets: bool = ...,
    ) -> str: ...
    def decode(
        self,
        token_ids: int | list[int] | np.ndarray | torch.Tensor | tf.Tensor,
        skip_special_tokens: bool = ...,
        clean_up_tokenization_spaces: bool | None = ...,
        output_char_offsets: bool = ...,
        **kwargs,
    ) -> str: ...
    def batch_decode(
        self,
        sequences: list[int] | list[list[int]] | np.ndarray | torch.Tensor | tf.Tensor,
        skip_special_tokens: bool = ...,
        clean_up_tokenization_spaces: bool | None = ...,
        output_char_offsets: bool = ...,
        **kwargs,
    ) -> list[str]: ...
    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = ...
    ) -> tuple[str]: ...

__all__ = ["Wav2Vec2PhonemeCTCTokenizer"]
