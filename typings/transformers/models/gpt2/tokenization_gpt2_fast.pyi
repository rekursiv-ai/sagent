from .tokenization_gpt2 import GPT2Tokenizer
from ...tokenization_utils_fast import PreTrainedTokenizerFast

"""Tokenization classes for OpenAI GPT."""
logger = ...
VOCAB_FILES_NAMES = ...

class GPT2TokenizerFast(PreTrainedTokenizerFast):
    slow_tokenizer_class = GPT2Tokenizer
    def __init__(
        self,
        vocab_file=...,
        merges_file=...,
        tokenizer_file=...,
        unk_token=...,
        bos_token=...,
        eos_token=...,
        add_prefix_space=...,
        **kwargs,
    ) -> None: ...
    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = ...
    ) -> tuple[str]: ...

__all__ = ["GPT2TokenizerFast"]
