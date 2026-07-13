from ...tokenization_utils_fast import PreTrainedTokenizerFast

"""Tokenization classes for GPTNeoX."""
logger = ...
VOCAB_FILES_NAMES = ...

class GPTNeoXTokenizerFast(PreTrainedTokenizerFast):
    vocab_files_names = ...
    model_input_names = ...
    def __init__(
        self,
        vocab_file=...,
        merges_file=...,
        tokenizer_file=...,
        unk_token=...,
        bos_token=...,
        eos_token=...,
        pad_token=...,
        add_bos_token=...,
        add_eos_token=...,
        add_prefix_space=...,
        **kwargs,
    ) -> None: ...
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
    def update_post_processor(self):  # -> None:
        ...
    def get_special_tokens_mask(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = ...,
        already_has_special_tokens: bool = ...,
    ) -> list[int]: ...
    def build_inputs_with_special_tokens(
        self, token_ids_0, token_ids_1=...
    ):  # -> list[str | list[str] | Any | None]:
        ...
    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = ...
    ) -> tuple[str]: ...

__all__ = ["GPTNeoXTokenizerFast"]
