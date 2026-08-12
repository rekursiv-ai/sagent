from .tokenization_qwen2 import Qwen2Tokenizer
from ...tokenization_utils_fast import PreTrainedTokenizerFast

"""Tokenization classes for Qwen2."""
logger = ...
VOCAB_FILES_NAMES = ...
MAX_MODEL_INPUT_SIZES = ...

class Qwen2TokenizerFast(PreTrainedTokenizerFast):
    slow_tokenizer_class = Qwen2Tokenizer
    def __init__(
        self,
        vocab_file=...,
        merges_file=...,
        tokenizer_file=...,
        unk_token=...,
        bos_token=...,
        eos_token=...,
        pad_token=...,
        **kwargs,
    ) -> None: ...
    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = ...
    ) -> tuple[str]: ...

__all__ = ["Qwen2TokenizerFast"]
