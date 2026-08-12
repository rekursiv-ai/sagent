from .tokenization_pegasus import PegasusTokenizer
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import is_sentencepiece_available

"""Tokenization class for model PEGASUS."""
if is_sentencepiece_available(): ...
else:
    PegasusTokenizer = ...
logger = ...
SPIECE_UNDERLINE = ...
VOCAB_FILES_NAMES = ...

class PegasusTokenizerFast(PreTrainedTokenizerFast):
    def __init__(
        self,
        vocab_file=...,
        tokenizer_file=...,
        pad_token=...,
        eos_token=...,
        unk_token=...,
        mask_token=...,
        mask_token_sent=...,
        additional_special_tokens=...,
        offset=...,
        **kwargs,
    ) -> None: ...
    def get_special_tokens_mask(
        self,
        token_ids_0: list,
        token_ids_1: list | None = ...,
        already_has_special_tokens: bool = ...,
    ) -> list[int]: ...
    def build_inputs_with_special_tokens(
        self, token_ids_0, token_ids_1=...
    ) -> list[int]: ...
    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = ...
    ) -> tuple[str]: ...

__all__ = ["PegasusTokenizerFast"]
