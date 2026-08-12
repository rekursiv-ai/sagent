from .tokenization_gemma import GemmaTokenizer
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import is_sentencepiece_available

if is_sentencepiece_available(): ...
else:
    GemmaTokenizer = ...
logger = ...
VOCAB_FILES_NAMES = ...

class GemmaTokenizerFast(PreTrainedTokenizerFast):
    def __init__(
        self,
        vocab_file=...,
        tokenizer_file=...,
        clean_up_tokenization_spaces=...,
        unk_token=...,
        bos_token=...,
        eos_token=...,
        pad_token=...,
        add_bos_token=...,
        add_eos_token=...,
        **kwargs,
    ) -> None: ...
    def update_post_processor(self):  # -> None:
        ...
    @property
    def add_eos_token(self):  # -> bool:
        ...
    @property
    def add_bos_token(self):  # -> bool:
        ...
    @add_eos_token.setter
    def add_eos_token(self, value):  # -> None:
        ...
    @add_bos_token.setter
    def add_bos_token(self, value):  # -> None:
        ...
    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = ...
    ) -> tuple[str]: ...
    def build_inputs_with_special_tokens(
        self, token_ids_0, token_ids_1=...
    ):  # -> list[str | list[str] | Any | None]:
        ...

__all__ = ["GemmaTokenizerFast"]
