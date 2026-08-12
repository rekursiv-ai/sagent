from .tokenization_bert import BertTokenizer
from ...tokenization_utils_fast import PreTrainedTokenizerFast

"""Fast Tokenization classes for Bert."""
logger = ...
VOCAB_FILES_NAMES = ...

class BertTokenizerFast(PreTrainedTokenizerFast):
    slow_tokenizer_class = BertTokenizer
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

__all__ = ["BertTokenizerFast"]
