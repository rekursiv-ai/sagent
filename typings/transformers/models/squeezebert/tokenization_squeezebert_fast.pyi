from .tokenization_squeezebert import SqueezeBertTokenizer
from ...tokenization_utils_fast import PreTrainedTokenizerFast

"""Tokenization classes for SqueezeBERT."""
logger = ...
VOCAB_FILES_NAMES = ...

class SqueezeBertTokenizerFast(PreTrainedTokenizerFast):
    slow_tokenizer_class = SqueezeBertTokenizer
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
        tokenize_chinese_chars=...,
        strip_accents=...,
        **kwargs,
    ) -> None: ...
    def build_inputs_with_special_tokens(
        self, token_ids_0, token_ids_1=...
    ):  # -> list[str | list[str] | Any | None]:
        ...
    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = ...
    ) -> tuple[str]: ...

__all__ = ["SqueezeBertTokenizerFast"]
