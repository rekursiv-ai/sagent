from .tokenization_deberta_v2 import DebertaV2Tokenizer
from ...file_utils import is_sentencepiece_available
from ...tokenization_utils_fast import PreTrainedTokenizerFast

"""Fast Tokenization class for model DeBERTa."""
if is_sentencepiece_available(): ...
else:
    DebertaV2Tokenizer = ...
logger = ...
VOCAB_FILES_NAMES = ...

class DebertaV2TokenizerFast(PreTrainedTokenizerFast):
    def __init__(
        self,
        vocab_file=...,
        tokenizer_file=...,
        do_lower_case=...,
        split_by_punct=...,
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
        self, token_ids_0, token_ids_1=...
    ):  # -> list[str | list[str] | Any | None]:
        ...
    def get_special_tokens_mask(
        self, token_ids_0, token_ids_1=..., already_has_special_tokens=...
    ):  # -> list[int]:
        ...
    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = ...
    ) -> tuple[str]: ...

__all__ = ["DebertaV2TokenizerFast"]
