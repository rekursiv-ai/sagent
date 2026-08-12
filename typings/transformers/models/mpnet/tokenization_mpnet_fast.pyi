from .tokenization_mpnet import MPNetTokenizer
from ...tokenization_utils_fast import PreTrainedTokenizerFast

"""Fast Tokenization classes for MPNet."""
logger = ...
VOCAB_FILES_NAMES = ...

class MPNetTokenizerFast(PreTrainedTokenizerFast):
    slow_tokenizer_class = MPNetTokenizer
    def __init__(
        self,
        vocab_file=...,
        tokenizer_file=...,
        do_lower_case=...,
        bos_token=...,
        eos_token=...,
        sep_token=...,
        cls_token=...,
        unk_token=...,
        pad_token=...,
        mask_token=...,
        tokenize_chinese_chars=...,
        strip_accents=...,
        **kwargs,
    ) -> None: ...
    @property
    def mask_token(self) -> str: ...
    @mask_token.setter
    def mask_token(self, value):  # -> None:
        ...
    def build_inputs_with_special_tokens(
        self, token_ids_0, token_ids_1=...
    ):  # -> list[str | list[str] | Any | None]:
        ...
    def create_token_type_ids_from_sequences(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = ...
    ) -> list[int]: ...
    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = ...
    ) -> tuple[str]: ...

__all__ = ["MPNetTokenizerFast"]
