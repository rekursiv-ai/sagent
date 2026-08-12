from dataclasses import dataclass

import numpy as np

from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import BatchEncoding
from ...utils import ModelOutput, PaddingStrategy, TensorType, add_end_docstrings

"""Tokenization class for Wav2Vec2."""
logger = ...
VOCAB_FILES_NAMES = ...
WAV2VEC2_KWARGS_DOCSTRING = ...
type ListOfDict = list[dict[str, int | str]]

@dataclass
class Wav2Vec2CTCTokenizerOutput(ModelOutput):
    text: list[str] | str
    char_offsets: list[ListOfDict] | ListOfDict = ...
    word_offsets: list[ListOfDict] | ListOfDict = ...

class Wav2Vec2CTCTokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        vocab_file,
        bos_token=...,
        eos_token=...,
        unk_token=...,
        pad_token=...,
        word_delimiter_token=...,
        replace_word_delimiter_char=...,
        do_lower_case=...,
        target_lang=...,
        **kwargs,
    ) -> None: ...
    def set_target_lang(self, target_lang: str):  # -> None:
        ...
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
    def vocab_size(self) -> int: ...
    def get_vocab(self) -> dict: ...
    def convert_tokens_to_string(
        self,
        tokens: list[str],
        group_tokens: bool = ...,
        spaces_between_special_tokens: bool = ...,
        output_char_offsets: bool = ...,
        output_word_offsets: bool = ...,
    ) -> dict[str, str | float]: ...
    def prepare_for_tokenization(
        self, text, is_split_into_words=..., **kwargs
    ):  # -> tuple[str, dict[str, Any]]:
        ...
    def batch_decode(
        self,
        sequences: list[int] | list[list[int]] | np.ndarray | torch.Tensor | tf.Tensor,
        skip_special_tokens: bool = ...,
        clean_up_tokenization_spaces: bool | None = ...,
        output_char_offsets: bool = ...,
        output_word_offsets: bool = ...,
        **kwargs,
    ) -> list[str]: ...
    def decode(
        self,
        token_ids: int | list[int] | np.ndarray | torch.Tensor | tf.Tensor,
        skip_special_tokens: bool = ...,
        clean_up_tokenization_spaces: bool | None = ...,
        output_char_offsets: bool = ...,
        output_word_offsets: bool = ...,
        **kwargs,
    ) -> str: ...
    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = ...
    ) -> tuple[str]: ...

class Wav2Vec2Tokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        vocab_file,
        bos_token=...,
        eos_token=...,
        unk_token=...,
        pad_token=...,
        word_delimiter_token=...,
        do_lower_case=...,
        do_normalize=...,
        return_attention_mask=...,
        **kwargs,
    ) -> None: ...
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
    @add_end_docstrings(WAV2VEC2_KWARGS_DOCSTRING)
    def __call__(
        self,
        raw_speech: np.ndarray | list[float] | list[np.ndarray] | list[list[float]],
        padding: bool | str | PaddingStrategy = ...,
        max_length: int | None = ...,
        pad_to_multiple_of: int | None = ...,
        padding_side: str | None = ...,
        return_tensors: str | TensorType | None = ...,
        verbose: bool = ...,
        **kwargs,
    ) -> BatchEncoding: ...
    @property
    def vocab_size(self) -> int: ...
    def get_vocab(self) -> dict: ...
    def convert_tokens_to_string(self, tokens: list[str]) -> str: ...
    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = ...
    ) -> tuple[str]: ...

__all__ = ["Wav2Vec2CTCTokenizer", "Wav2Vec2Tokenizer"]
