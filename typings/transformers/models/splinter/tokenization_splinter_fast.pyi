from .tokenization_splinter import SplinterTokenizer
from ...tokenization_utils_fast import PreTrainedTokenizerFast

"""Fast Tokenization classes for Splinter."""
logger = ...
VOCAB_FILES_NAMES = ...

class SplinterTokenizerFast(PreTrainedTokenizerFast):
    slow_tokenizer_class = SplinterTokenizer
    def __init__(
        self,
        vocab_file=...,
        tokenizer_file=...,
        do_lower_case=...,
        unk_token=...,
        sep_token=...,
        pad_token=...,
        cls_token=...,
        mask_token=...,
        question_token=...,
        tokenize_chinese_chars=...,
        strip_accents=...,
        **kwargs,
    ) -> None: ...
    @property
    def question_token_id(self):  # -> int | list[int]:
        ...
    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = ...
    ) -> list[int]: ...
    def create_token_type_ids_from_sequences(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = ...
    ) -> list[int]: ...
    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = ...
    ) -> tuple[str]: ...

__all__ = ["SplinterTokenizerFast"]
