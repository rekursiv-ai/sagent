from typing import Any

from ...tokenization_utils import PreTrainedTokenizer
from ...utils.import_utils import requires

"""Tokenization class for SpeechT5."""
logger = ...
VOCAB_FILES_NAMES = ...

@requires(backends=("sentencepiece",))
class SpeechT5Tokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        vocab_file,
        bos_token=...,
        eos_token=...,
        unk_token=...,
        pad_token=...,
        normalize=...,
        sp_model_kwargs: dict[str, Any] | None = ...,
        **kwargs,
    ) -> None: ...
    def prepare_for_tokenization(
        self, text, is_split_into_words=..., **kwargs
    ):  # -> tuple[str | Any, dict[str, Any]]:
        ...
    @property
    def vocab_size(self): ...
    @property
    def normalizer(self):  # -> EnglishNumberNormalizer:
        ...
    @normalizer.setter
    def normalizer(self, value):  # -> None:
        ...
    def get_vocab(self):  # -> dict[str, int]:
        ...
    def __getstate__(self):  # -> dict[str, Any]:
        ...
    def __setstate__(self, d):  # -> None:
        ...
    def convert_tokens_to_string(self, tokens): ...
    def build_inputs_with_special_tokens(
        self, token_ids_0, token_ids_1=...
    ) -> list[int]: ...
    def get_special_tokens_mask(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = ...,
        already_has_special_tokens: bool = ...,
    ) -> list[int]: ...
    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = ...
    ) -> tuple[str]: ...

__all__ = ["SpeechT5Tokenizer"]
