from .tokenization_mvp import MvpTokenizer
from ...tokenization_utils_fast import PreTrainedTokenizerFast

logger = ...
VOCAB_FILES_NAMES = ...

class MvpTokenizerFast(PreTrainedTokenizerFast):
    slow_tokenizer_class = MvpTokenizer
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
        trim_offsets=...,
        **kwargs,
    ) -> None: ...
    @property
    def mask_token(self) -> str: ...
    @mask_token.setter
    def mask_token(self, value):  # -> None:
        ...
    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = ...
    ) -> tuple[str]: ...
    def build_inputs_with_special_tokens(
        self, token_ids_0, token_ids_1=...
    ):  # -> list[str | list[str] | Any | None]:
        ...
    def create_token_type_ids_from_sequences(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = ...
    ) -> list[int]: ...

__all__ = ["MvpTokenizerFast"]
