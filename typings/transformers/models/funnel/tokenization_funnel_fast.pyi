from .tokenization_funnel import FunnelTokenizer
from ...tokenization_utils_fast import PreTrainedTokenizerFast

"""Tokenization class for Funnel Transformer."""
logger = ...
VOCAB_FILES_NAMES = ...
_model_names = ...

class FunnelTokenizerFast(PreTrainedTokenizerFast):
    slow_tokenizer_class = FunnelTokenizer
    cls_token_type_id: int = ...
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
        bos_token=...,
        eos_token=...,
        clean_text=...,
        tokenize_chinese_chars=...,
        strip_accents=...,
        wordpieces_prefix=...,
        **kwargs,
    ) -> None: ...
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

__all__ = ["FunnelTokenizerFast"]
