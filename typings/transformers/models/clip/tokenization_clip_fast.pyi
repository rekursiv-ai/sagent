from .tokenization_clip import CLIPTokenizer
from ...tokenization_utils_fast import PreTrainedTokenizerFast

"""Tokenization classes for OpenAI GPT."""
logger = ...
VOCAB_FILES_NAMES = ...

class CLIPTokenizerFast(PreTrainedTokenizerFast):
    slow_tokenizer_class = CLIPTokenizer
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
    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = ...
    ) -> list[int]: ...
    def create_token_type_ids_from_sequences(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = ...
    ) -> list[int]: ...
    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = ...
    ) -> tuple[str]: ...

__all__ = ["CLIPTokenizerFast"]
