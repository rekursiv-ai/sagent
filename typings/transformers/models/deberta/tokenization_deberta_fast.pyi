from .tokenization_deberta import DebertaTokenizer
from ...tokenization_utils_fast import PreTrainedTokenizerFast

"""Fast Tokenization class for model DeBERTa."""
logger = ...
VOCAB_FILES_NAMES = ...

class DebertaTokenizerFast(PreTrainedTokenizerFast):
    slow_tokenizer_class = DebertaTokenizer
    def __init__(
        self,
        vocab_file=...,
        merges_file=...,
        tokenizer_file=...,
        errors=...,
        bos_token=...,
        eos_token=...,
        sep_token=...,
        cls_token=...,
        unk_token=...,
        pad_token=...,
        mask_token=...,
        add_prefix_space=...,
        **kwargs,
    ) -> None: ...
    @property
    def mask_token(self) -> str: ...
    @mask_token.setter
    def mask_token(self, value):  # -> None:
        ...
    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = ...
    ) -> list[int]: ...
    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = ...
    ) -> tuple[str]: ...

__all__ = ["DebertaTokenizerFast"]
