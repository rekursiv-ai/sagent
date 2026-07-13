from .tokenization_rembert import RemBertTokenizer
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import is_sentencepiece_available

"""Tokenization classes for RemBERT model."""
if is_sentencepiece_available(): ...
else:
    RemBertTokenizer = ...
logger = ...
VOCAB_FILES_NAMES = ...
SPIECE_UNDERLINE = ...

class RemBertTokenizerFast(PreTrainedTokenizerFast):
    vocab_files_names = ...
    slow_tokenizer_class = ...
    def __init__(
        self,
        vocab_file=...,
        tokenizer_file=...,
        do_lower_case=...,
        remove_space=...,
        keep_accents=...,
        bos_token=...,
        eos_token=...,
        unk_token=...,
        sep_token=...,
        pad_token=...,
        cls_token=...,
        mask_token=...,
        **kwargs,
    ) -> None: ...
    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = ...
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

__all__ = ["RemBertTokenizerFast"]
