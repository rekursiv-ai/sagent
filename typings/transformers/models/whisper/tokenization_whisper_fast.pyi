from functools import lru_cache

from .tokenization_whisper import WhisperTokenizer
from ...tokenization_utils_fast import PreTrainedTokenizerFast

"""Tokenization classes for Whisper."""
logger = ...
VOCAB_FILES_NAMES = ...

class WhisperTokenizerFast(PreTrainedTokenizerFast):
    slow_tokenizer_class = WhisperTokenizer
    def __init__(
        self,
        vocab_file=...,
        merges_file=...,
        normalizer_file=...,
        tokenizer_file=...,
        unk_token=...,
        bos_token=...,
        eos_token=...,
        add_prefix_space=...,
        language=...,
        task=...,
        predict_timestamps=...,
        **kwargs,
    ) -> None: ...
    @lru_cache  # noqa: B019 -- mirrors upstream @lru_cache on a method; not executed in a stub
    def timestamp_ids(self, time_precision=...):  # -> int | list[int]:
        ...
    def decode(
        self,
        token_ids,
        skip_special_tokens: bool = ...,
        clean_up_tokenization_spaces: bool | None = ...,
        output_offsets: bool = ...,
        time_precision: float = ...,
        decode_with_timestamps: bool = ...,
        normalize: bool = ...,
        basic_normalize: bool = ...,
        remove_diacritics: bool = ...,
        **kwargs,
    ) -> str: ...
    def normalize(self, text):  # -> str:
        ...
    @staticmethod
    def basic_normalize(text, remove_diacritics=...):  # -> str:
        ...
    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = ...
    ) -> tuple[str]: ...
    def set_prefix_tokens(
        self,
        language: str | None = ...,
        task: str | None = ...,
        predict_timestamps: bool | None = ...,
    ):  # -> None:
        ...
    @property
    def prefix_tokens(self) -> list[int]: ...
    def build_inputs_with_special_tokens(
        self, token_ids_0, token_ids_1=...
    ) -> list[int]: ...
    def get_special_tokens_mask(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = ...,
        already_has_special_tokens: bool = ...,
    ) -> list[int]: ...
    def get_decoder_prompt_ids(
        self, task=..., language=..., no_timestamps=...
    ):  # -> list[tuple[int, int]]:
        ...
    def get_prompt_ids(self, text: str, return_tensors=...):  # -> Encoding:
        ...

__all__ = ["WhisperTokenizerFast"]
