from typing import Any

from ...tokenization_utils import PreTrainedTokenizer
from ...utils.import_utils import requires

"""Tokenization class for model BertGeneration."""
logger = ...
VOCAB_FILES_NAMES = ...

@requires(backends=("sentencepiece",))
class BertGenerationTokenizer(PreTrainedTokenizer):
    prefix_tokens: list[int] = ...
    def __init__(
        self,
        vocab_file,
        bos_token=...,
        eos_token=...,
        unk_token=...,
        pad_token=...,
        sep_token=...,
        sp_model_kwargs: dict[str, Any] | None = ...,
        **kwargs,
    ) -> None: ...
    @property
    def vocab_size(self): ...
    def get_vocab(self):  # -> dict[str, int]:
        ...
    def __getstate__(self):  # -> dict[str, Any]:
        ...
    def __setstate__(self, d):  # -> None:
        ...
    def convert_tokens_to_string(self, tokens): ...
    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = ...
    ) -> tuple[str]: ...

__all__ = ["BertGenerationTokenizer"]
