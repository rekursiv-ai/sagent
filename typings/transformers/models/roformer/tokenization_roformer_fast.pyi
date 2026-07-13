from .tokenization_roformer import RoFormerTokenizer
from ...tokenization_utils_fast import PreTrainedTokenizerFast

"""Tokenization classes for RoFormer."""
logger = ...
VOCAB_FILES_NAMES = ...

class RoFormerTokenizerFast(PreTrainedTokenizerFast):
    vocab_files_names = ...
    slow_tokenizer_class = RoFormerTokenizer
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
    def __getstate__(self):  # -> dict[str, Any]:
        ...
    def __setstate__(self, d):  # -> None:
        ...
    def build_inputs_with_special_tokens(
        self, token_ids_0, token_ids_1=...
    ):  # -> list[str | list[str] | Any | None]:
        ...
    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = ...
    ) -> tuple[str]: ...
    def save_pretrained(
        self,
        save_directory,
        legacy_format=...,
        filename_prefix=...,
        push_to_hub=...,
        **kwargs,
    ):  # -> tuple[str, ...]:
        ...

__all__ = ["RoFormerTokenizerFast"]
