from .tokenization_distilbert import DistilBertTokenizer
from ...tokenization_utils_fast import PreTrainedTokenizerFast

"""Tokenization classes for DistilBERT."""
logger = ...
VOCAB_FILES_NAMES = ...

class DistilBertTokenizerFast(PreTrainedTokenizerFast):
    vocab_files_names = ...
    model_input_names = ...
    slow_tokenizer_class = DistilBertTokenizer
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

__all__ = ["DistilBertTokenizerFast"]
